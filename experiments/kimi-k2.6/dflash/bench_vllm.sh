#!/bin/bash
set -e

# Benchmark: Kimi-K2.6 + DFlash on vLLM
# GPUs 1,2,5,7 (other 4 GPUs not used by training)
export ROCR_VISIBLE_DEVICES=1,2,5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

MODEL_PATH="moonshotai/Kimi-K2.6"
DFLASH_PATH="/workspace/checkpoints/dflash/kimi-k2.6-pretrain/epoch_0_step_100000"
PORT=30200
NUM_PROMPTS=10
MAX_TOKENS=1000

# Generate prompts
python3 -c "
import json
prompts = [
    'You are a senior software architect reviewing a complex distributed system. The system consists of multiple microservices communicating via gRPC and Apache Kafka. The main services include: (1) An API Gateway that handles authentication, rate limiting, and request routing using JWT tokens and OAuth2.0 flows. It processes approximately 50,000 requests per second at peak load. (2) A User Service that manages user profiles, preferences, and session data, backed by PostgreSQL with read replicas and Redis caching. (3) An Order Processing Service that handles the complete order lifecycle from creation to fulfillment, using event sourcing with Apache Kafka as the event store. (4) A Payment Service that integrates with Stripe, PayPal, and various regional payment providers, implementing the saga pattern for distributed transactions. (5) An Inventory Service that tracks real-time stock levels across 500 warehouses, using CRDT-based conflict resolution for concurrent updates. (6) A Notification Service that sends emails, SMS, and push notifications through multiple providers with intelligent routing and fallback mechanisms. (7) A Search Service built on Elasticsearch with custom analyzers for product search, supporting faceted search, autocomplete, and personalized ranking. (8) An Analytics Service that processes clickstream data using Apache Flink for real-time dashboards and Apache Spark for batch processing. Please provide a comprehensive review covering architecture improvements, scalability recommendations, and cost optimization strategies.',
    'Explain in detail how modern CPU architectures achieve instruction-level parallelism through multiple mechanisms. Start with the basics of pipelining - how a simple 5-stage pipeline works, what pipeline hazards exist, and how they are resolved through forwarding, stalling, and branch prediction. Then discuss superscalar execution, out-of-order execution with reorder buffers and reservation stations, register renaming, modern branch predictors from 2-bit counters to TAGE, the memory hierarchy, SIMD extensions, and simultaneous multithreading. Include specific examples from Intel and AMD processor families.',
    'Write a comprehensive guide to implementing a production-grade machine learning pipeline for a recommendation system. Cover data collection and feature engineering, training pipeline with temporal splits, model architectures from matrix factorization to deep learning approaches, serving infrastructure with approximate nearest neighbor search, A/B testing methodology, MLOps aspects, and ethical considerations including filter bubbles and fairness.',
    'Provide a thorough analysis of the Byzantine Generals Problem and its implications for distributed consensus. Cover the original formulation, impossibility results, Paxos, Raft, PBFT, modern developments like HotStuff and DAG-based protocols, the CAP theorem, and blockchain consensus protocols.',
    'Design a complete compiler for a statically-typed functional programming language with algebraic data types, pattern matching, type inference using Algorithm W, first-class functions with closures, tail call optimization, and a garbage collector. Walk through each compiler phase: lexical analysis, parsing, type checking, desugaring, optimization passes, code generation targeting LLVM IR, and runtime implementation.',
    'Analyze the complete history and technical evolution of cryptographic hash functions. Cover formal definitions, the birthday paradox, Merkle-Damgard construction, MD5 and its vulnerabilities, SHA-1 and SHAttered, SHA-2, the SHA-3 competition and Keccak sponge construction, and BLAKE2/BLAKE3.',
    'Describe the complete architecture of a modern web browser engine from receiving HTML bytes to rendering pixels. Cover networking, HTML parsing, CSS cascade, layout algorithms, painting, compositing, JavaScript engine integration, and accessibility.',
    'Explain the mathematical foundations and practical implementation of public-key cryptography. Cover number theory prerequisites, RSA, elliptic curve cryptography, Diffie-Hellman key exchange, digital signatures, and post-quantum cryptography.',
    'Write a detailed technical analysis of how large language models work. Cover the Transformer architecture, training pipeline, scaling laws, post-training with RLHF/DPO, and inference optimization including speculative decoding and quantization.',
    'Provide a comprehensive overview of modern database internals covering B-tree and LSM-tree storage engines, buffer pool management, WAL and ARIES recovery, query processing and optimization, concurrency control with MVCC, and distributed database concepts.'
]
with open('/workspace/bench_prompts_vllm.json', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'Generated {len(prompts)} prompts')
"

# --- Benchmark 1: Base K2.6 (no DFlash) ---
echo ""
echo "=== Starting vLLM Base K2.6 Server (no DFlash) ==="
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 32768 &
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
echo "=== Running Base Benchmark (max_tokens=$MAX_TOKENS) ==="
python3 -c "
import openai, time, json

client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')
with open('/workspace/bench_prompts_vllm.json') as f:
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
print(f'=== VLLM BASE RESULTS ===')
print(f'Total tokens: {total_tokens}')
print(f'Total time: {total_time:.2f}s')
print(f'Avg throughput: {total_tokens/total_time:.1f} tokens/sec')
print(f'Avg latency: {total_time/$NUM_PROMPTS:.2f}s per request')

with open('/workspace/bench_vllm_base_results.json', 'w') as f:
    json.dump({'total_tokens': total_tokens, 'total_time': total_time, 'results': results}, f, indent=2)
"

kill $SERVER_PID 2>/dev/null || true
sleep 10

# --- Benchmark 2: K2.6 + DFlash ---
echo ""
echo "=== Starting vLLM K2.6 + DFlash Server ==="
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 32768 \
    --speculative-config '{"method": "dflash", "model": "'"$DFLASH_PATH"'", "num_speculative_tokens": 8}' &
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
echo "=== Running DFlash Benchmark (max_tokens=$MAX_TOKENS) ==="
python3 -c "
import openai, time, json

client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')
with open('/workspace/bench_prompts_vllm.json') as f:
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
print(f'=== VLLM DFLASH RESULTS ===')
print(f'Total tokens: {total_tokens}')
print(f'Total time: {total_time:.2f}s')
print(f'Avg throughput: {total_tokens/total_time:.1f} tokens/sec')
print(f'Avg latency: {total_time/$NUM_PROMPTS:.2f}s per request')

with open('/workspace/bench_vllm_dflash_results.json', 'w') as f:
    json.dump({'total_tokens': total_tokens, 'total_time': total_time, 'results': results}, f, indent=2)

# Compare
with open('/workspace/bench_vllm_base_results.json') as f:
    base = json.load(f)
base_tps = base['total_tokens'] / base['total_time']
dflash_tps = total_tokens / total_time
print()
print(f'=== COMPARISON (vLLM) ===')
print(f'Base:   {base_tps:.1f} tokens/sec')
print(f'DFlash: {dflash_tps:.1f} tokens/sec')
print(f'Speedup: {dflash_tps/base_tps:.2f}x')
"

kill $SERVER_PID 2>/dev/null || true
echo ""
echo "=== vLLM Benchmark Complete ==="
