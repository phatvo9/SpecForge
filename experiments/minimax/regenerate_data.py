"""
Regenerate training data for MiniMax-M2.7 EAGLE3 training.
Takes user/system prompts from gpt-oss datasets and generates responses using MiniMax-M2.7.
"""
import json
import os
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm

BASE_URL = "http://localhost:30200"
MAX_TOKENS = 16000
TEMPERATURE = 0
OUTPUT_DIR = "/home/claudeuser/specforge_data/minimax2.7-spec-data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_prompts(dataset_name):
    """Extract user/system messages from a dataset."""
    ds = load_dataset(dataset_name, split="train")
    prompts = []
    for item in ds:
        convs = item["conversations"]
        messages = []
        for msg in convs:
            role = msg["role"]
            if role in ("user", "system"):
                messages.append({"role": role, "content": msg["content"]})
            elif role == "assistant_reasoning_effort":
                # Skip - MiniMax has its own reasoning
                continue
            else:
                # Stop at first assistant message - we'll regenerate
                break
        if messages and any(m["role"] == "user" for m in messages):
            prompts.append({"messages": messages, "source": dataset_name.split("/")[-1]})
    return prompts


def generate_response(messages, max_retries=3):
    """Send messages to MiniMax server and get response."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "MiniMaxAI/MiniMax-M2.7",
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
            usage = data["usage"]
            return {
                "content": content,
                "finish_reason": finish_reason,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
    return None


def process_single(prompt_data):
    """Process a single prompt and return the conversation."""
    messages = prompt_data["messages"]
    result = generate_response(messages)
    if result is None or result["finish_reason"] == "error":
        return None

    # Build full conversation for training
    conversation = []
    for msg in messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})
    conversation.append({"role": "assistant", "content": result["content"]})

    return {
        "conversations": conversation,
        "source": prompt_data["source"],
        "finish_reason": result["finish_reason"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    }


def main():
    # Check server is up
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Server health: {resp.status_code}")
    except:
        print("ERROR: Server not reachable at", BASE_URL)
        return

    # Extract prompts from all 4 datasets
    all_prompts = []
    datasets = [
        "phatv9/ultrachat_gpt-oss-120b-high",
        "phatv9/magpie-llama3.1-pro-300k-filtered_gpt-oss-120b-high",
        "phatv9/aa-benchmark-spec-gptoss-high",
        "phatv9/gptoss-high-3k-each",
    ]

    for ds_name in datasets:
        prompts = extract_prompts(ds_name)
        print(f"{ds_name}: {len(prompts)} prompts extracted")
        all_prompts.extend(prompts)

    print(f"\nTotal prompts: {len(all_prompts)}")
    random.seed(42)
    random.shuffle(all_prompts)

    # Generate responses with concurrent requests for throughput
    results = []
    failed = 0
    output_file = os.path.join(OUTPUT_DIR, "train_raw.jsonl")
    NUM_WORKERS = 64  # concurrent requests

    # Skip already generated samples
    existing = 0
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming: {existing} samples already generated, skipping...")
        all_prompts = all_prompts[existing:]

    with open(output_file, "a") as f:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            pbar = tqdm(total=len(all_prompts), desc="Generating")

            for i, prompt in enumerate(all_prompts):
                future = executor.submit(process_single, prompt)
                futures[future] = i

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    f.write(json.dumps(result) + "\n")
                    results.append(result)
                    if len(results) % 50 == 0:
                        f.flush()
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=len(results), fail=failed)

            pbar.close()

    print(f"\nDone! Generated {len(results)} samples, {failed} failed")
    print(f"Saved to {output_file}")

    # Summary stats
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    print(f"Total prompt tokens: {total_prompt_tokens:,}")
    print(f"Total completion tokens: {total_completion_tokens:,}")

    # Source distribution
    from collections import Counter

    sources = Counter(r["source"] for r in results)
    for src, count in sources.most_common():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
