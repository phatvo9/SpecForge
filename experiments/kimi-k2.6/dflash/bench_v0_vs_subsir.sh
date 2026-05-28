#!/bin/bash
set -e

# Benchmark: v0 DFlash vs SubSir DFlash on Kimi-K2.6
export ROCR_VISIBLE_DEVICES=1,2,5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai requests tqdm 2>&1 | tail -3

# Fix tokenizer
OLD_SNAP="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/2755962d07cb42aa2d988a35bcb65cd4a9c2de82"
NEW_SNAP="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/81bcaaa7947338ce2641983d98947bca0cc1a4d4"
REFS="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/refs/main"
echo "2755962d07cb42aa2d988a35bcb65cd4a9c2de82" > "$REFS"
for f in tokenizer.json tokenization_kimi_fast.py tool_declaration_ts.py; do
    if [ ! -e "$OLD_SNAP/$f" ] && [ -e "$NEW_SNAP/$f" ]; then
        cp "$NEW_SNAP/$f" "$OLD_SNAP/$f"
        echo "Copied $f into old snapshot"
    fi
done
rm -rf /root/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi_hyphen_K2_dot_6
echo "Cleared transformers module cache"

MODEL_PATH="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/2755962d07cb42aa2d988a35bcb65cd4a9c2de82"
V0_PATH="/workspace/checkpoints/dflash/kimi-k2.6-finetune-subsir/epoch_0_step_0"
SUBSIR_PATH="/workspace/checkpoints/dflash/kimi-k2.6-ft-mixed-41k/epoch_0_step_24000"
PORT=30200
NUM_PROMPTS=10
MAX_TOKENS=1000

# Generate prompts
python3 -c "
import json
prompts = [
    'You are a senior software architect reviewing a complex distributed system. The system consists of multiple microservices communicating via gRPC and Apache Kafka. The main services include: (1) An API Gateway that handles authentication, rate limiting, and request routing using JWT tokens and OAuth2.0 flows. It processes approximately 50,000 requests per second at peak load. (2) A User Service that manages user profiles, preferences, and session data, backed by PostgreSQL with read replicas and Redis caching. (3) An Order Processing Service that handles the complete order lifecycle from creation to fulfillment, using event sourcing with Apache Kafka as the event store. (4) A Payment Service that integrates with Stripe, PayPal, and various regional payment providers, implementing the saga pattern for distributed transactions. (5) An Inventory Service that tracks real-time stock levels across 500 warehouses, using CRDT-based conflict resolution for concurrent updates. (6) A Notification Service that sends emails, SMS, and push notifications through multiple providers with intelligent routing and fallback mechanisms. (7) A Search Service built on Elasticsearch with custom analyzers for product search, supporting faceted search, autocomplete, and personalized ranking. (8) An Analytics Service that processes clickstream data using Apache Flink for real-time dashboards and Apache Spark for batch processing. Please provide a comprehensive review covering architecture improvements, scalability recommendations, and cost optimization strategies.',
    'Explain in detail how modern CPU architectures achieve instruction-level parallelism through multiple mechanisms. Start with the basics of pipelining, pipeline hazards, and how they are resolved. Then discuss superscalar execution, out-of-order execution with reorder buffers, register renaming, modern branch predictors, the memory hierarchy, SIMD extensions, and simultaneous multithreading.',
    'Write a comprehensive guide to implementing a production-grade machine learning pipeline for a recommendation system. Cover data collection, feature engineering, training pipeline, model architectures, serving infrastructure, A/B testing methodology, MLOps aspects, and ethical considerations.',
    'Provide a thorough analysis of the Byzantine Generals Problem and its implications for distributed consensus. Cover Paxos, Raft, PBFT, modern developments like HotStuff, the CAP theorem, and blockchain consensus protocols.',
    'Design a complete compiler for a statically-typed functional programming language with algebraic data types, pattern matching, type inference, first-class functions, tail call optimization, and a garbage collector. Walk through each compiler phase.',
    'Analyze the complete history and technical evolution of cryptographic hash functions from MD5 through SHA-3 and BLAKE3.',
    'Describe the complete architecture of a modern web browser engine from HTML bytes to rendered pixels.',
    'Explain the mathematical foundations and practical implementation of public-key cryptography including RSA, ECC, and post-quantum schemes.',
    'Write a detailed technical analysis of how large language models work including Transformer architecture, training, scaling laws, RLHF, and inference optimization.',
    'Provide a comprehensive overview of modern database internals covering storage engines, WAL, query optimization, MVCC, and distributed concepts.'
]
with open('/workspace/bench_prompts.json', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'Generated {len(prompts)} prompts')
"

run_benchmark() {
    local NAME=$1
    local DFLASH_PATH=$2
    local RESULT_FILE=$3

    local LOG_FILE="/workspace/bench_${NAME}_server.log"

    echo ""
    echo "=== Starting K2.6 + $NAME DFlash Server ==="
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
        --port $PORT > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    echo "Waiting for server..."
    for i in $(seq 1 600); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    echo ""
    echo "=== Running $NAME Benchmark (max_tokens=$MAX_TOKENS) ==="
    python3 -c "
import openai, time, json

client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')
with open('/workspace/bench_prompts.json') as f:
    prompts = json.load(f)

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
print(f'=== $NAME RESULTS ===')
print(f'Total tokens: {total_tokens}')
print(f'Total time: {total_time:.2f}s')
print(f'Avg throughput: {total_tokens/total_time:.1f} tokens/sec')
print(f'Avg latency: {total_time/$NUM_PROMPTS:.2f}s per request')

with open('$RESULT_FILE', 'w') as f:
    json.dump({'name': '$NAME', 'total_tokens': total_tokens, 'total_time': total_time, 'results': results}, f, indent=2)
"

    # Extract accept len/rate from server logs
    echo ""
    echo "=== $NAME Accept Rate ==="
    python3 -c "
import re
with open('$LOG_FILE') as f:
    lines = f.read()
matches = re.findall(r'accept len: ([\d.]+), accept rate: ([\d.]+)', lines)
if matches:
    lens = [float(m[0]) for m in matches]
    rates = [float(m[1]) for m in matches]
    print(f'  Decode batches: {len(matches)}')
    print(f'  Avg accept len:  {sum(lens)/len(lens):.2f}  (min={min(lens):.2f}, max={max(lens):.2f})')
    print(f'  Avg accept rate: {sum(rates)/len(rates):.2f}  (min={min(rates):.2f}, max={max(rates):.2f})')
else:
    print('  No accept data found in logs')
"

    kill $SERVER_PID 2>/dev/null || true
    sleep 10
}

# --- Benchmark 1: v0 (our finetune) ---
run_benchmark "ft-mixed-41k-step24k" "$SUBSIR_PATH" "/workspace/bench_subsir_results.json"

# --- Compare ---
echo ""
echo "=== COMPARISON ==="
python3 -c "
import json

with open('/workspace/bench_v0_results.json') as f:
    v0 = json.load(f)
with open('/workspace/bench_subsir_results.json') as f:
    subsir = json.load(f)

v0_tps = v0['total_tokens'] / v0['total_time']
subsir_tps = subsir['total_tokens'] / subsir['total_time']

print(f'v0-ft-step5k:     {v0_tps:.1f} tokens/sec  (avg latency: {v0[\"total_time\"]/len(v0[\"results\"]):.2f}s)')
print(f'SubSir-tmp-long:  {subsir_tps:.1f} tokens/sec  (avg latency: {subsir[\"total_time\"]/len(subsir[\"results\"]):.2f}s)')
print(f'')
if v0_tps > subsir_tps:
    print(f'v0 is {v0_tps/subsir_tps:.2f}x faster')
else:
    print(f'SubSir is {subsir_tps/v0_tps:.2f}x faster')
"

# Save all results to checkpoints dir (host-mounted, survives container removal)
cp /workspace/bench_v0_results.json /workspace/checkpoints/ 2>/dev/null
cp /workspace/bench_subsir_results.json /workspace/checkpoints/ 2>/dev/null
cp /workspace/bench_*_server.log /workspace/checkpoints/ 2>/dev/null

echo ""
echo "=== Benchmark Complete ==="
