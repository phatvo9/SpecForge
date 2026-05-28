#!/bin/bash
set -e

# Benchmark: K2.6 base vs K2.6 + current DFlash checkpoint
# GPUs 1,2,5,7 (the other 4 GPUs not used by training)
export ROCR_VISIBLE_DEVICES=1,2,5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai requests tqdm 2>&1 | tail -3

# Fix tokenizer: old snapshot (2755) has weights but missing tokenizer.json
# New snapshot (81bc) has tokenizer but no weights. Symlink missing files.
OLD_SNAP="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/2755962d07cb42aa2d988a35bcb65cd4a9c2de82"
NEW_SNAP="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/81bcaaa7947338ce2641983d98947bca0cc1a4d4"
REFS="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/refs/main"
# Point main ref to old snapshot that has weights
echo "2755962d07cb42aa2d988a35bcb65cd4a9c2de82" > "$REFS"
# Copy missing tokenizer files from new snapshot
for f in tokenizer.json tokenization_kimi_fast.py tool_declaration_ts.py; do
    if [ ! -e "$OLD_SNAP/$f" ] && [ -e "$NEW_SNAP/$f" ]; then
        cp "$NEW_SNAP/$f" "$OLD_SNAP/$f"
        echo "Copied $f into old snapshot"
    fi
done
# Clear cached transformers module so it re-resolves with the old snapshot
rm -rf /root/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi_hyphen_K2_dot_6
echo "Cleared transformers module cache"

echo "=== GPU Info ==="
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

# Use resolved local snapshot path (has both weights + tokenizer after our fix)
MODEL_PATH="/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/2755962d07cb42aa2d988a35bcb65cd4a9c2de82"
DFLASH_PATH="/workspace/checkpoints/dflash/kimi-k2.6-finetune-v0/epoch_0_step_5000"
PORT=30200
NUM_PROMPTS=10
MAX_TOKENS=1000

# Generate 10 prompts of ~2k tokens each
python3 -c "
import json

prompts = [
    '''You are a senior software architect reviewing a complex distributed system. The system consists of multiple microservices communicating via gRPC and Apache Kafka. The main services include: (1) An API Gateway that handles authentication, rate limiting, and request routing using JWT tokens and OAuth2.0 flows. It processes approximately 50,000 requests per second at peak load. (2) A User Service that manages user profiles, preferences, and session data, backed by PostgreSQL with read replicas and Redis caching. (3) An Order Processing Service that handles the complete order lifecycle from creation to fulfillment, using event sourcing with Apache Kafka as the event store. (4) A Payment Service that integrates with Stripe, PayPal, and various regional payment providers, implementing the saga pattern for distributed transactions. (5) An Inventory Service that tracks real-time stock levels across 500 warehouses, using CRDT-based conflict resolution for concurrent updates. (6) A Notification Service that sends emails, SMS, and push notifications through multiple providers with intelligent routing and fallback mechanisms. (7) A Search Service built on Elasticsearch with custom analyzers for product search, supporting faceted search, autocomplete, and personalized ranking. (8) An Analytics Service that processes clickstream data using Apache Flink for real-time dashboards and Apache Spark for batch processing. The system currently handles 2 million daily active users and processes $50M in transactions monthly. Recent issues include: increased latency in the order processing pipeline during flash sales, occasional data inconsistencies between the inventory service and order service, and growing infrastructure costs. Please provide a comprehensive review covering: architecture improvements, scalability recommendations, cost optimization strategies, and a migration plan for moving from the current monolithic database to a fully distributed data architecture. Include specific technology recommendations and implementation timelines.''',

    '''Explain in detail how modern CPU architectures achieve instruction-level parallelism through multiple mechanisms. Start with the basics of pipelining - how a simple 5-stage pipeline (fetch, decode, execute, memory, writeback) works, what pipeline hazards exist (data hazards, control hazards, structural hazards), and how they are resolved through forwarding, stalling, and branch prediction. Then discuss superscalar execution, where multiple instructions are issued per cycle, and the challenges of dependency detection and resource allocation. Cover out-of-order execution in depth: the role of the reorder buffer (ROB), reservation stations, register renaming to eliminate false dependencies (WAR and WAW hazards), and the commit stage that ensures precise exceptions. Explain how modern branch predictors work, from simple 2-bit saturating counters to sophisticated TAGE predictors that use multiple history lengths. Discuss the memory hierarchy and how it interacts with ILP: store buffers, load-store queues, memory disambiguation, and speculative loads. Cover SIMD extensions (SSE, AVX, AVX-512) and how they provide data-level parallelism. Discuss simultaneous multithreading (SMT/Hyper-Threading) and how it improves throughput by sharing execution resources between threads. Finally, explain the limitations of ILP exploitation: the power wall, the memory wall, and Amdahl's law, and how these have driven the shift toward multicore architectures. Include specific examples from Intel and AMD processor families where relevant, discussing microarchitectural features like Intel's decoded instruction cache (DSB), loop stream detector, and AMD's op cache.''',

    '''Write a comprehensive guide to implementing a production-grade machine learning pipeline for a recommendation system. The system should handle both collaborative filtering and content-based approaches, with a hybrid model that combines both. Start with data collection and feature engineering: what user interaction signals to collect (clicks, purchases, time spent, scroll depth, explicit ratings), how to handle implicit feedback, and feature extraction from item metadata (text embeddings using BERT/sentence-transformers, image features using ResNet/ViT, categorical encoding). Discuss the training pipeline: data splitting strategies for temporal data (time-based splits, not random), handling cold-start problems for new users and items, negative sampling strategies, and evaluation metrics (NDCG, MAP, Hit Rate, Coverage, Diversity). Cover model architectures: matrix factorization (ALS, BPR), deep learning approaches (Neural Collaborative Filtering, Wide & Deep, DeepFM, two-tower models), and sequential models (SASRec, BERT4Rec, GRU4Rec). Explain the serving infrastructure: candidate generation using approximate nearest neighbor search (FAISS, ScaNN), ranking models, real-time feature stores (Feast, Tecton), model serving with low latency (TensorRT, ONNX Runtime, Triton Inference Server). Discuss A/B testing methodology: proper randomization, metric selection, statistical significance testing, guardrail metrics, and how to handle network effects. Cover MLOps aspects: experiment tracking (MLflow, Weights & Biases), model versioning, automated retraining pipelines, monitoring for data drift and concept drift, and alerting. Finally, discuss ethical considerations: filter bubbles, popularity bias, fairness across user groups, and privacy-preserving techniques like federated learning.''',

    '''Provide a thorough analysis of the Byzantine Generals Problem and its implications for distributed consensus. Begin with the original problem formulation by Lamport, Shostak, and Pease: how n generals surrounding a city must agree on a common plan of attack or retreat, when up to f of them may be traitors sending conflicting messages. Prove that consensus is impossible with 3f+1 or fewer generals (the impossibility result), and show the oral messages algorithm that achieves consensus with 3f+1 generals using f+1 rounds of message exchange. Then trace the evolution to practical consensus protocols: explain Paxos (single-decree and multi-Paxos) with its phases (prepare, promise, accept, accepted), the role of proposers, acceptors, and learners, and how it handles leader failures. Discuss Raft as a more understandable alternative, covering leader election, log replication, and safety guarantees. Explain how PBFT (Practical Byzantine Fault Tolerance) adapted these ideas for Byzantine faults with 3f+1 replicas, its three-phase protocol (pre-prepare, prepare, commit), view changes, and its O(n^2) message complexity. Cover modern developments: HotStuff's linear communication complexity and its influence on blockchain consensus (Facebook's LibraBFT), Tendermint's approach used in Cosmos, and the DAG-based protocols like Narwhal/Tusk and Bullshark. Discuss the CAP theorem and how it relates to consensus: why you cannot simultaneously guarantee consistency, availability, and partition tolerance, and how different systems make different tradeoffs (CP systems like ZooKeeper vs AP systems like Cassandra). Finally, explain how blockchain consensus protocols (Nakamoto consensus, proof-of-stake, etc.) relate to classical BFT.''',

    '''Design a complete compiler for a statically-typed functional programming language with the following features: algebraic data types (sum types and product types), pattern matching, type inference using Algorithm W (Hindley-Milner), first-class functions with closures, tail call optimization, and a garbage collector. Walk through each compiler phase in detail. Lexical analysis: define the token types, handle string escaping, nested comments, significant whitespace rules, and unicode identifiers. Parsing: design an LL(1) or PEG grammar, handle operator precedence and associativity using Pratt parsing, parse let bindings, lambda expressions, match expressions, and type annotations. Type checking and inference: implement Algorithm W with constraint generation and unification, handle let-polymorphism (generalization and instantiation), type classes or traits for ad-hoc polymorphism, and provide good error messages for type errors including expected vs actual type formatting. Desugaring: transform pattern matching into decision trees or backtracking automata, desugar do-notation or for-comprehensions, expand type class dictionaries. Optimization passes: constant folding, dead code elimination, inlining (with cost model), common subexpression elimination, lambda lifting, closure conversion, and defunctionalization as an alternative. Code generation: targeting either LLVM IR or a custom bytecode VM, implement the calling convention for closures, represent algebraic data types efficiently (tagged unions), implement tail calls using trampolining or direct jumps, generate efficient pattern matching code. Runtime: implement a copying garbage collector (Cheney's algorithm) or a mark-and-sweep collector, handle weak references, finalization, and interaction with FFI.''',

    '''Analyze the complete history and technical evolution of cryptographic hash functions, from their theoretical foundations to modern constructions. Start with the formal definition: a hash function H maps arbitrary-length inputs to fixed-length outputs, and a cryptographic hash function must satisfy three properties - preimage resistance (given h, hard to find m such that H(m)=h), second preimage resistance (given m1, hard to find m2≠m1 such that H(m1)=H(m2)), and collision resistance (hard to find any m1≠m2 such that H(m1)=H(m2)). Explain the birthday paradox and how it gives the birthday bound of 2^(n/2) for collision finding in an n-bit hash. Cover the Merkle-Damgård construction: the compression function, initialization vector, padding scheme (including the length extension attack vulnerability), and why it preserves collision resistance from the compression function to the full hash. Discuss MD5: its design by Rivest, the 4 rounds of 16 operations each using different nonlinear functions (F, G, H, I), the Wang et al. differential attack that found practical collisions in 2004, and the Flame malware that exploited MD5 collisions in Microsoft certificates. Cover SHA-1: its design based on MD4/MD5, the theoretical attack by Wang et al. in 2005, and Google's SHAttered practical collision in 2017. Explain SHA-2 (SHA-256, SHA-512): the design improvements over SHA-1, the compression function with 8 working variables, the message schedule, and why no practical attacks exist. Discuss the SHA-3 competition and Keccak: the sponge construction as an alternative to Merkle-Damgård (absorb and squeeze phases), the state permutation, and the security proof based on the indifferentiability framework. Cover BLAKE2 and BLAKE3: their design based on ChaCha, performance characteristics, and tree hashing for parallelism.''',

    '''Describe the complete architecture and implementation details of a modern web browser engine, from receiving HTML bytes to rendering pixels on screen. Start with the networking layer: DNS resolution (recursive resolvers, DNS-over-HTTPS), TCP connection establishment, TLS 1.3 handshake (1-RTT with supported cipher suites), HTTP/2 multiplexing and server push, HTTP/3 with QUIC protocol. Then cover HTML parsing: the tokenizer state machine (handling script tags, entity references, character encoding detection), tree construction algorithm (the stack of open elements, formatting elements, adoption agency algorithm), error recovery and quirks mode. CSS parsing: the cascade algorithm (specificity calculation, origin sorting, !important handling), selector matching (right-to-left evaluation for efficiency), the box model, and computed style resolution. Layout (reflow): the formatting context model (block, inline, flex, grid), float positioning, containing blocks, margin collapsing, percentage resolution, intrinsic sizing, and text layout (line breaking using the Unicode Line Breaking Algorithm, bidirectional text with the Unicode Bidi Algorithm, font shaping with HarfBuzz, font fallback chains). Painting: creating display lists, paint order (backgrounds, borders, floats, foreground, outlines), stacking contexts and z-index, opacity and blend modes. Compositing: layer tree construction, promotion to compositor layers (will-change, transforms, opacity), GPU texture uploading, tile-based rendering. JavaScript engine integration: the event loop, microtasks vs macrotasks, layout thrashing prevention, requestAnimationFrame scheduling, Intersection Observer and Resize Observer. Accessibility: the accessibility tree, ARIA roles and properties, screen reader interaction.''',

    '''Explain the mathematical foundations and practical implementation of public-key cryptography systems. Begin with the number theory prerequisites: modular arithmetic, Euler's totient function and Euler's theorem, the extended Euclidean algorithm for computing modular inverses, the Chinese Remainder Theorem and its application to efficient RSA operations. Explain the RSA cryptosystem: key generation (choosing large primes p and q, computing n=pq and φ(n)=(p-1)(q-1), selecting e and computing d=e^(-1) mod φ(n)), encryption (c=m^e mod n), decryption (m=c^d mod n), and the proof of correctness using Euler's theorem. Discuss RSA security: the relationship to integer factorization, the RSA problem vs the factoring problem, chosen-ciphertext attacks and the need for padding schemes (OAEP), side-channel attacks (timing attacks, power analysis), and the current state of factoring records. Cover elliptic curve cryptography: the group law on elliptic curves over finite fields (point addition and doubling), the discrete logarithm problem on elliptic curves (ECDLP), curve selection criteria (rigidity, twist security, cofactor), standard curves (NIST P-256, Curve25519, Ed448), and scalar multiplication algorithms (double-and-add, Montgomery ladder for constant-time implementation). Explain Diffie-Hellman key exchange in both classical and elliptic curve settings, the static vs ephemeral variants, and how it provides perfect forward secrecy in TLS. Discuss digital signatures: RSA signatures with PSS padding, ECDSA (the k-value reuse vulnerability that broke PlayStation 3), EdDSA (deterministic nonces), and Schnorr signatures. Cover post-quantum cryptography: the threat of Shor's algorithm, lattice-based schemes (NTRU, Kyber/ML-KEM), code-based schemes (Classic McEliece), hash-based signatures (SPHINCS+), and the NIST PQC standardization process.''',

    '''Write a detailed technical analysis of how large language models work, covering the complete architecture and training pipeline. Start with the Transformer architecture: the self-attention mechanism (query, key, value projections, scaled dot-product attention, the softmax bottleneck), multi-head attention (why multiple heads capture different relationship types), positional encoding (sinusoidal, learned, RoPE, ALiBi), feed-forward networks (the SwiGLU activation variant), layer normalization (pre-norm vs post-norm, RMSNorm), and residual connections. Explain the autoregressive language modeling objective: next-token prediction, the cross-entropy loss, teacher forcing during training, and causal masking to prevent attending to future tokens. Cover the training pipeline: data collection and curation (Common Crawl filtering, deduplication with MinHash, quality filtering), tokenization (BPE, WordPiece, SentencePiece, the vocabulary size tradeoff), distributed training strategies (data parallelism, tensor parallelism, pipeline parallelism, ZeRO optimization stages), mixed-precision training (BF16, loss scaling), gradient accumulation, and learning rate scheduling (warmup, cosine decay). Discuss scaling laws: the Chinchilla optimal compute allocation, the relationship between model size, dataset size, and compute budget, and emergent abilities at scale. Cover post-training: supervised fine-tuning (SFT) on instruction-following data, reinforcement learning from human feedback (RLHF) with the PPO algorithm and reward modeling, direct preference optimization (DPO) as an alternative, and constitutional AI (RLAIF). Explain inference optimization: KV caching, speculative decoding (how draft models accelerate generation), quantization (GPTQ, AWQ, GGML), flash attention, paged attention (vLLM), continuous batching, and tensor parallelism for serving.''',

    '''Provide a comprehensive overview of modern database internals, covering both traditional RDBMS and modern distributed databases. Start with storage engines: B-tree based engines (how B+ trees work for range queries, page splits and merges, write amplification), LSM-tree based engines (memtable, sorted string tables, compaction strategies - size-tiered vs leveled, bloom filters for read optimization, write amplification vs read amplification tradeoffs). Explain buffer pool management: page replacement policies (LRU, clock, LRU-K), dirty page flushing strategies, and how modern databases use direct I/O to bypass the OS page cache. Cover the write-ahead log (WAL): log record format, physiological logging, checkpointing (fuzzy checkpoints), and crash recovery using ARIES (analysis, redo, undo phases). Discuss query processing: parsing and semantic analysis, logical plan optimization (predicate pushdown, join reordering using dynamic programming or greedy algorithms, subquery decorrelation), physical plan selection (cost-based optimization, cardinality estimation using histograms and HyperLogLog sketches), and execution engines (volcano/iterator model vs vectorized/batch model vs compilation-based like HyPer). Cover concurrency control: two-phase locking (2PL) with deadlock detection, multi-version concurrency control (MVCC) with snapshot isolation and its write skew anomaly, serializable snapshot isolation (SSI), and optimistic concurrency control. Explain distributed database concepts: partitioning strategies (hash, range, consistent hashing), replication (synchronous vs asynchronous, chain replication), distributed transactions (2PC, 3PC, Calvin's deterministic execution, Spanner's TrueTime), and the architecture of systems like CockroachDB, TiDB, YugabyteDB, and FoundationDB.'''
]

with open('/workspace/bench_prompts.json', 'w') as f:
    json.dump(prompts, f, indent=2)
print(f'Generated {len(prompts)} prompts')
for i, p in enumerate(prompts):
    print(f'  Prompt {i}: ~{len(p.split())} words')
"

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
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; then
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

# --- Benchmark 2: K2.6 + trained DFlash ---
echo ""
echo "=== Starting K2.6 + DFlash Server (step 100k checkpoint) ==="
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
for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; then
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
