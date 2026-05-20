"""
LLM-as-judge benchmark validation using Claude.
Evaluates all 150 pairs in counterfactual_eval_v3_150_draft.json.
Saves results to benchmark_validation_claude.json
"""
import json
import os
import time
import anthropic

DATA_PATH   = r"C:\nlp_project\data\counterfactual_eval_v3_150_draft.json"
OUTPUT_PATH = r"C:\nlp_project\data\benchmark_validation_claude.json"
BATCH_SIZE  = 15

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env

def evaluate_batch(pairs: list[dict]) -> list[dict]:
    lines = []
    for p in pairs:
        lines.append(
            f"ID={p['id']}\n"
            f"Original:       {p['original']}\n"
            f"Counterfactual: {p['counterfactual']}\n"
        )

    prompt = (
        "아래 한국어 문장 쌍들을 각각 평가해줘.\n"
        "각 문장이 혐오/공격적 표현이면 1, 아니면 0으로 판단해.\n"
        "Original과 Counterfactual 각각 독립적으로 판단해.\n\n"
        "출력 형식 (반드시 이 형식만, 다른 설명 없이):\n"
        "ID=숫자 orig=0또는1 cf=0또는1\n\n"
        + "\n".join(lines)
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()

    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("ID="):
            continue
        try:
            parts = line.split()
            pid   = int(parts[0].split("=")[1])
            orig  = int(parts[1].split("=")[1])
            cf    = int(parts[2].split("=")[1])
            results.append({"id": pid, "claude_orig": orig, "claude_cf": cf})
        except Exception:
            continue
    return results


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["pairs"]

    all_results = {}
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        print(f"Evaluating pairs {i+1}–{min(i+BATCH_SIZE, len(pairs))}...")
        results = evaluate_batch(batch)
        for r in results:
            all_results[r["id"]] = {"claude_orig": r["claude_orig"], "claude_cf": r["claude_cf"]}
        time.sleep(0.5)

    # Merge with original pairs and compute agreement
    final = []
    agree = 0
    disagree_ids = []

    for p in pairs:
        pid = p["id"]
        if pid not in all_results:
            print(f"  WARNING: no result for ID={pid}")
            continue
        cr = all_results[pid]
        orig_match = cr["claude_orig"] == p["expected_label"]
        cf_match   = cr["claude_cf"]   == p["expected_label"]
        both_agree = orig_match and cf_match

        if both_agree:
            agree += 1
        else:
            disagree_ids.append(pid)

        final.append({
            **p,
            "claude_orig":  cr["claude_orig"],
            "claude_cf":    cr["claude_cf"],
            "orig_agree":   orig_match,
            "cf_agree":     cf_match,
            "both_agree":   both_agree,
        })

    total = len(final)
    print(f"\n=== Validation Summary ===")
    print(f"Total evaluated: {total}")
    print(f"Both agree:      {agree} ({agree/total*100:.1f}%)")
    print(f"Disagreements:   {len(disagree_ids)}")
    print(f"Disagree IDs:    {disagree_ids}")

    from collections import Counter
    for cat in ["adversarial_clean", "stereotype_flip_strict", "subtle_stereotype"]:
        grp = [r for r in final if r["category"] == cat]
        ag  = sum(r["both_agree"] for r in grp)
        print(f"  {cat}: {ag}/{len(grp)} agree ({ag/len(grp)*100:.1f}%)")

    output = {
        "meta": {"validator": "claude-haiku-4-5", "total": total, "agree": agree, "agree_rate": agree/total},
        "pairs": final,
        "validated_ids": [r["id"] for r in final if r["both_agree"]],
        "rejected_ids":  disagree_ids,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
