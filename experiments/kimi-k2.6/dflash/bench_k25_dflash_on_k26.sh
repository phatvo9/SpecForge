#!/bin/bash
set -e

# Benchmark: K2.6 base vs K2.6 + K2.5 DFlash
# GPUs 0,3,4,6 (ROCR 1,0,5,6) — 4 GPUs for TP=4 (BF16 ~1TB)
export ROCR_VISIBLE_DEVICES=1,0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export PYTHONPATH="/workspace/SpecForge:/workspace/SpecForge/benchmarks:$PYTHONPATH"

# Use local SGLang for kimi_k25 support
export PYTHONPATH="/workspace/official_sglang/python:$PYTHONPATH"

pip install openai requests tqdm 2>&1 | tail -3

echo "=== GPU Info ==="
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

MODEL_PATH="moonshotai/Kimi-K2.6"
DFLASH_PATH="z-lab/Kimi-K2.5-DFlash"
PORT=30200
NUM_PROMPTS=20
MAX_TOKENS=256

# --- Benchmark 1: Base K2.6 (no DFlash) ---
echo ""
echo "=== Starting Base K2.6 Server (no DFlash) ==="
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp-size 4 \
    --attention-backend aiter \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port $PORT &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/health | grep -q "ok"; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""
echo "=== Running Base Benchmark ==="
python3 -c "
import openai, time, json

client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')

prompts = [
    'Explain the theory of relativity in simple terms.',
    'Write a Python function to find the longest common subsequence.',
    'What are the main differences between TCP and UDP?',
    'Describe the process of photosynthesis step by step.',
    'Write a short story about a robot learning to paint.',
    'Explain how neural networks learn through backpropagation.',
    'What are the pros and cons of microservices architecture?',
    'Derive the quadratic formula from ax^2 + bx + c = 0.',
    'Compare and contrast REST and GraphQL APIs.',
    'Explain the CAP theorem in distributed systems.',
    'Write a recursive solution for the Tower of Hanoi problem.',
    'What is the significance of the Turing test?',
    'Describe the water cycle and its importance to Earth.',
    'Explain how public-key cryptography works.',
    'Write a haiku about artificial intelligence.',
    'What are the key principles of object-oriented programming?',
    'Explain the difference between supervised and unsupervised learning.',
    'Describe how a compiler works from source to binary.',
    'What causes the seasons on Earth?',
    'Write a regular expression to validate email addresses.',
]

total_tokens = 0
total_time = 0
results = []

for i, prompt in enumerate(prompts[:$NUM_PROMPTS]):
    start = time.time()
    resp = client.chat.completions.create(
        model='$MODEL_PATH',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=$MAX_TOKENS,
        temperature=0.0,
    )
    elapsed = time.time() - start
    tokens = resp.usage.completion_tokens
    total_tokens += tokens
    total_time += elapsed
    tps = tokens / elapsed
    results.append({'prompt_idx': i, 'tokens': tokens, 'time': elapsed, 'tps': tps})
    print(f'  [{i+1}/$NUM_PROMPTS] {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)')

print()
print(f'=== BASE RESULTS ===')
print(f'Total tokens: {total_tokens}')
print(f'Total time: {total_time:.2f}s')
print(f'Avg throughput: {total_tokens/total_time:.1f} tokens/sec')
print(f'Avg latency: {total_time/$NUM_PROMPTS:.2f}s per request')

with open('/workspace/bench_base_results.json', 'w') as f:
    json.dump({'total_tokens': total_tokens, 'total_time': total_time, 'results': results}, f, indent=2)
"

# Kill base server
kill $SERVER_PID 2>/dev/null || true
sleep 10

# --- Benchmark 2: K2.6 + K2.5 DFlash ---
echo ""
echo "=== Starting K2.6 + K2.5 DFlash Server ==="
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path $DFLASH_PATH \
    --speculative-num-draft-tokens 8 \
    --tp-size 4 \
    --attention-backend aiter \
    --speculative-draft-attention-backend aiter \
    --mem-fraction-static 0.75 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port $PORT &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health | grep -q "ok"; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""
echo "=== Running DFlash Benchmark ==="
python3 -c "
import openai, time, json

client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')

prompts = [
    'Explain the theory of relativity in simple terms.',
    'Write a Python function to find the longest common subsequence.',
    'What are the main differences between TCP and UDP?',
    'Describe the process of photosynthesis step by step.',
    'Write a short story about a robot learning to paint.',
    'Explain how neural networks learn through backpropagation.',
    'What are the pros and cons of microservices architecture?',
    'Derive the quadratic formula from ax^2 + bx + c = 0.',
    'Compare and contrast REST and GraphQL APIs.',
    'Explain the CAP theorem in distributed systems.',
    'Write a recursive solution for the Tower of Hanoi problem.',
    'What is the significance of the Turing test?',
    'Describe the water cycle and its importance to Earth.',
    'Explain how public-key cryptography works.',
    'Write a haiku about artificial intelligence.',
    'What are the key principles of object-oriented programming?',
    'Explain the difference between supervised and unsupervised learning.',
    'Describe how a compiler works from source to binary.',
    'What causes the seasons on Earth?',
    'Write a regular expression to validate email addresses.',
]

total_tokens = 0
total_time = 0
results = []

for i, prompt in enumerate(prompts[:$NUM_PROMPTS]):
    start = time.time()
    resp = client.chat.completions.create(
        model='$MODEL_PATH',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=$MAX_TOKENS,
        temperature=0.0,
    )
    elapsed = time.time() - start
    tokens = resp.usage.completion_tokens
    total_tokens += tokens
    total_time += elapsed
    tps = tokens / elapsed
    results.append({'prompt_idx': i, 'tokens': tokens, 'time': elapsed, 'tps': tps})
    print(f'  [{i+1}/$NUM_PROMPTS] {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)')

print()
print(f'=== DFLASH RESULTS ===')
print(f'Total tokens: {total_tokens}')
print(f'Total time: {total_time:.2f}s')
print(f'Avg throughput: {total_tokens/total_time:.1f} tokens/sec')
print(f'Avg latency: {total_time/$NUM_PROMPTS:.2f}s per request')

with open('/workspace/bench_dflash_results.json', 'w') as f:
    json.dump({'total_tokens': total_tokens, 'total_time': total_time, 'results': results}, f, indent=2)

# Compare
with open('/workspace/bench_base_results.json') as f:
    base = json.load(f)
base_tps = base['total_tokens'] / base['total_time']
dflash_tps = total_tokens / total_time
print()
print(f'=== COMPARISON ===')
print(f'Base:   {base_tps:.1f} tokens/sec')
print(f'DFlash: {dflash_tps:.1f} tokens/sec')
print(f'Speedup: {dflash_tps/base_tps:.2f}x')
"

kill $SERVER_PID 2>/dev/null || true
echo ""
echo "=== Benchmark Complete ==="
