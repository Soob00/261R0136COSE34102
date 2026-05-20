"""Hard Negative Evaluation for Dual-Head Orthogonality Korean Hate Speech Detection.

목적:
  - Baseline(klue/roberta-base)의 Shortcut Learning 취약점을 수치로 증명
  - 3가지 카테고리별 Counterfactual Consistency 측정
    1. adversarial_clean  : label=0, 정체성 키워드가 중립/긍정 맥락에 등장 -> FP 함정
    2. stereotype_flip    : label=1, 고정관념 공격 문맥에서 집단만 치환 -> 일관성 검증
    3. implicit_hate      : label=1, 명시적 키워드 없는 우회적 혐오 -> FN 함정

평가 지표:
  - Per-category Counterfactual Consistency (orig_pred == cf_pred 비율)
  - Per-category Accuracy (pred == expected_label 비율)
  - Overall comparison: Baseline vs Dual-head

Usage:
    cd c:/nlp_project
    python hard_negative_eval.py [--subset 400] [--epochs 5] [--lambda-ortho 0.05]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.real_model import BaselineClassifier, DualHeadClassifier, orthogonality_loss


# -- KOLD loading (same as real_pilot_run.py) --

def load_kold(path: Path, subset: int | None, seed: int = 42) -> list[tuple[str, int]]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    examples = [(d["comment"], int(d["OFF"])) for d in raw if d["comment"].strip()]
    if subset:
        rng = random.Random(seed)
        pos = [e for e in examples if e[1] == 1]
        neg = [e for e in examples if e[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        examples = pos[:subset // 2] + neg[:subset // 2]
        rng.shuffle(examples)
    return examples


def split_data(examples, val_ratio=0.2, seed=42):
    items = examples[:]
    random.Random(seed).shuffle(items)
    cut = int(len(items) * (1 - val_ratio))
    return items[:cut], items[cut:]


class KOLDDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        enc = self.tokenizer(text, max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# -- training --

def train_epoch(model, loader, optimizer, device, lambda_ortho):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_cls = total_ortho = 0.0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits, z_cls, z_sem = model(ids, mask)
        cls_l = criterion(logits, y)
        o_l = orthogonality_loss(z_cls, z_sem) if z_cls is not None else torch.tensor(0.0, device=device)
        loss = cls_l + lambda_ortho * o_l
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = ids.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_l.item() * bs
        total_ortho += o_l.item() * bs
        n += bs
    n = max(n, 1)
    return total_loss / n, total_cls / n, total_ortho / n


def evaluate_loader(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            logits, _, _ = model(ids, mask)
            total_loss += criterion(logits, y).item() * ids.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += ids.size(0)
    n = max(total, 1)
    return total_loss / n, correct / n


def train_model(name, model, train_loader, val_loader, device, epochs, lr, lambda_ortho):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_acc = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  epochs={epochs}  lr={lr}  lambda_ortho={lambda_ortho}")
    print(f"{'='*60}")
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'OrthoL':>8}  {'ValLoss':>8}  {'ValAcc':>7}")

    for epoch in range(1, epochs + 1):
        tr_loss, cls_l, ortho_l = train_epoch(model, train_loader, optimizer, device, lambda_ortho)
        val_loss, val_acc = evaluate_loader(model, val_loader, device)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {ortho_l:>8.4f}  {val_loss:>8.4f}  {val_acc:>7.3f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best val_acc={best_acc:.4f}")
    return model


# -- hard negative evaluation --

def predict_text(model, text, tokenizer, max_len, device):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits, _, _ = model(ids, mask)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)
    return pred, prob


def evaluate_hard_negatives(
    model: nn.Module,
    pairs: list[dict],
    tokenizer,
    max_len: int,
    device: torch.device,
    model_name: str,
) -> dict:
    """Evaluate model on hard negative pairs.

    Returns per-category and overall:
      - consistency: orig_pred == cf_pred
      - accuracy:    pred == expected_label (for both orig and cf)
    """
    results_by_cat = defaultdict(lambda: {"total": 0, "consistent": 0,
                                          "orig_correct": 0, "cf_correct": 0,
                                          "details": []})

    for pair in pairs:
        cat = pair["category"]
        expected = pair["expected_label"]
        orig_pred, orig_prob = predict_text(model, pair["original"], tokenizer, max_len, device)
        cf_pred, cf_prob = predict_text(model, pair["counterfactual"], tokenizer, max_len, device)

        consistent = (orig_pred == cf_pred)
        orig_correct = (orig_pred == expected)
        cf_correct = (cf_pred == expected)

        r = results_by_cat[cat]
        r["total"] += 1
        r["consistent"] += int(consistent)
        r["orig_correct"] += int(orig_correct)
        r["cf_correct"] += int(cf_correct)
        r["details"].append({
            "id": pair["id"],
            "expected": expected,
            "orig_pred": orig_pred,
            "cf_pred": cf_pred,
            "orig_prob": round(orig_prob, 3),
            "cf_prob": round(cf_prob, 3),
            "consistent": consistent,
            "orig_correct": orig_correct,
            "cf_correct": cf_correct,
            "identity_orig": pair.get("identity_orig", ""),
            "identity_cf": pair.get("identity_cf", ""),
            "trap": pair.get("trap", ""),
            "original_text": pair["original"][:50],
            "cf_text": pair["counterfactual"][:50],
        })

    # aggregate
    total_all = sum(r["total"] for r in results_by_cat.values())
    consistent_all = sum(r["consistent"] for r in results_by_cat.values())
    orig_correct_all = sum(r["orig_correct"] for r in results_by_cat.values())
    cf_correct_all = sum(r["cf_correct"] for r in results_by_cat.values())

    return {
        "model": model_name,
        "overall": {
            "total": total_all,
            "consistency": consistent_all / max(total_all, 1),
            "orig_accuracy": orig_correct_all / max(total_all, 1),
            "cf_accuracy": cf_correct_all / max(total_all, 1),
        },
        "by_category": {
            cat: {
                "total": r["total"],
                "consistency": r["consistent"] / max(r["total"], 1),
                "orig_accuracy": r["orig_correct"] / max(r["total"], 1),
                "cf_accuracy": r["cf_correct"] / max(r["total"], 1),
                "inconsistent_pairs": [d for d in r["details"] if not d["consistent"]],
            }
            for cat, r in results_by_cat.items()
        },
        "raw": dict(results_by_cat),
    }


def print_hard_negative_report(res: dict, category_meta: dict | None = None, verbose: bool = True) -> list[str]:
    lines = []
    w = lines.append

    w(f"\n{'='*65}")
    w(f"  Hard Negative Evaluation: [{res['model']}]")
    w(f"{'='*65}")

    ov = res["overall"]
    w(f"  Overall  total={ov['total']}  "
      f"consistency={ov['consistency']:.2f}  "
      f"orig_acc={ov['orig_accuracy']:.2f}  "
      f"cf_acc={ov['cf_accuracy']:.2f}")
    w("")

    category_meta = category_meta or {}
    for cat, c in res["by_category"].items():
        meta = category_meta.get(cat, {})
        expected = meta.get("expected_label", "?")
        desc = meta.get("description", cat)
        header = f"{cat} (label={expected})"
        if desc and desc != cat:
            header = f"{header}: {desc}"
        w(f"  [{header}]")
        w(f"    n={c['total']}  consistency={c['consistency']:.2f}  "
          f"orig_acc={c['orig_accuracy']:.2f}  cf_acc={c['cf_accuracy']:.2f}")

        inconsistent = c["inconsistent_pairs"]
        if inconsistent:
            w(f"    Inconsistent pairs ({len(inconsistent)}/{c['total']}):")
            for d in inconsistent[:10]:  # limit output
                w(f"      [FLIP] id={d['id']:>3} exp={d['expected']} "
                  f"orig={d['orig_pred']}({d['orig_prob']:.2f}) "
                  f"cf={d['cf_pred']}({d['cf_prob']:.2f}) "
                  f"| {d['identity_orig']} -> {d['identity_cf']}")
                if verbose:
                    w(f"             orig: {d['original_text']}")
                    w(f"             cf  : {d['cf_text']}")
            if len(inconsistent) > 10:
                w(f"      ... and {len(inconsistent)-10} more")
        else:
            w(f"    All consistent (no flips)")
        w("")

    return lines


# -- main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           type=Path,  default=ROOT / "data/kold_v1.json")
    parser.add_argument("--hard-negatives", type=Path,  default=ROOT / "data/hard_negative_testset.json")
    parser.add_argument("--subset",         type=int,   default=400)
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--max-len",        type=int,   default=128)
    parser.add_argument("--head-dim",       type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--lambda-ortho",   type=float, default=0.05)
    parser.add_argument("--model-name",     type=str,   default="klue/roberta-base")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--save-results",   type=Path,  default=ROOT / "results/hard_negative_eval.txt")
    parser.add_argument("--verbose",        action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_lines = []
    def log(line=""):
        print(line)
        output_lines.append(line)

    log(f"Hard Negative Evaluation")
    log(f"  device={device}  model={args.model_name}")
    log(f"  subset={args.subset}  epochs={args.epochs}  lambda_ortho={args.lambda_ortho}")

    # -- load KOLD --
    subset = args.subset if args.subset > 0 else None
    examples = load_kold(args.data, subset=subset, seed=args.seed)
    train_data, val_data = split_data(examples, seed=args.seed)
    log(f"\nKOLD: total={len(examples)}  train={len(train_data)}  val={len(val_data)}")

    # -- tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = KOLDDataset(train_data, tokenizer, args.max_len)
    val_ds   = KOLDDataset(val_data,   tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # -- load hard negatives --
    with args.hard_negatives.open(encoding="utf-8") as f:
        hn_data = json.load(f)
    pairs = hn_data["pairs"]
    log(f"\nHard Negative Set: {len(pairs)} pairs")
    category_meta = hn_data["meta"]["categories"]
    for cat, meta in category_meta.items():
        log(f"  {cat}: {meta['count']} pairs - {meta['description'][:60]}")

    # -- train baseline, evaluate, then free memory before loading dual-head --
    baseline = BaselineClassifier(model_name=args.model_name).to(device)
    baseline = train_model(
        "Baseline (Single-head)", baseline, train_loader, val_loader,
        device, args.epochs, args.lr, lambda_ortho=0.0
    )

    log("\nEvaluating baseline on hard negatives...")
    res_base = evaluate_hard_negatives(baseline, pairs, tokenizer, args.max_len, device, "Baseline")

    # free baseline from memory before loading dual-head
    del baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()
    log("Baseline freed from memory.")

    # -- train dual-head --
    torch.manual_seed(args.seed)
    dual = DualHeadClassifier(model_name=args.model_name, head_dim=args.head_dim).to(device)
    dual = train_model(
        f"Dual-head + Ortho (lambda={args.lambda_ortho})", dual, train_loader, val_loader,
        device, args.epochs, args.lr, lambda_ortho=args.lambda_ortho
    )

    log("\nEvaluating dual-head on hard negatives...")
    res_dual = evaluate_hard_negatives(dual, pairs, tokenizer, args.max_len, device, "Dual-head")

    del dual
    import gc; gc.collect()

    # -- print reports --
    log(f"\n{'='*65}")
    log(f"  HARD NEGATIVE EVALUATION RESULTS")
    log(f"{'='*65}")

    for res in [res_base, res_dual]:
        for line in print_hard_negative_report(res, category_meta=category_meta, verbose=args.verbose):
            log(line)


    # -- comparison table --
    log(f"\n{'='*65}")
    log(f"  COMPARISON TABLE")
    log(f"{'='*65}")
    log(f"  {'Category':<28} {'Baseline':>10} {'Dual-head':>10}  {'Delta':>6}")
    log(f"  {'-'*28} {'-'*10} {'-'*10}  {'-'*6}")

    cats = list(category_meta.keys())
    cat_labels = {
        cat: f"{cat} (n={category_meta[cat].get('count', res_base['by_category'].get(cat, {}).get('total', 0))})"
        for cat in cats
    }
    for cat in cats:
        bc = res_base["by_category"].get(cat, {}).get("consistency", 0)
        dc = res_dual["by_category"].get(cat, {}).get("consistency", 0)
        delta = dc - bc
        sign = "+" if delta >= 0 else ""
        log(f"  {cat_labels[cat]:<28} {bc:>10.2f} {dc:>10.2f}  {sign}{delta:>+.2f}")

    log(f"  {'-'*28} {'-'*10} {'-'*10}  {'-'*6}")
    bo = res_base["overall"]["consistency"]
    do_ = res_dual["overall"]["consistency"]
    delta_o = do_ - bo
    log(f"  {'Overall':<28} {bo:>10.2f} {do_:>10.2f}  {delta_o:>+.2f}")

    log(f"\n  Accuracy (avg orig+cf):")
    for cat in cats:
        ba = (res_base["by_category"].get(cat, {}).get("orig_accuracy", 0) +
              res_base["by_category"].get(cat, {}).get("cf_accuracy", 0)) / 2
        da = (res_dual["by_category"].get(cat, {}).get("orig_accuracy", 0) +
              res_dual["by_category"].get(cat, {}).get("cf_accuracy", 0)) / 2
        log(f"    {cat_labels[cat]:<26}  Baseline={ba:.2f}  Dual-head={da:.2f}")

    # -- interpretation --
    log(f"\n{'='*65}")
    log(f"  INTERPRETATION")
    log(f"{'='*65}")

    for cat in cats:
        meta = category_meta.get(cat, {})
        expected = meta.get("expected_label")
        desc = meta.get("description", cat)
        base_cons = res_base["by_category"].get(cat, {}).get("consistency", 1.0)
        base_acc = (
            res_base["by_category"].get(cat, {}).get("orig_accuracy", 0) +
            res_base["by_category"].get(cat, {}).get("cf_accuracy", 0)
        ) / 2
        log(f"\n  [{cat}] {desc}")
        log(f"    Baseline consistency={base_cons:.2f}  avg_acc={base_acc:.2f}  expected_label={expected}")
        if expected == 0 and base_acc < 0.80:
            log(f"    -> False-positive vulnerability likely remains in neutral identity contexts.")
        elif expected == 1 and base_cons < 0.85:
            log(f"    -> Group swap changes predictions often; shortcut reliance may still be present.")
        elif expected == 1 and base_acc < 0.70:
            log(f"    -> Hard positive examples are under-detected; false negatives remain substantial.")
        else:
            log(f"    -> Baseline is relatively stable on this category.")

    log(f"\n  [Dual-head vs Baseline]:")
    if do_ > bo:
        log(f"    Dual-head overall consistency={do_:.2f} > Baseline={bo:.2f} (+{do_-bo:.2f})")
        log(f"    -> Orthogonality constraint effective at reducing bias")
    elif do_ == bo:
        log(f"    Dual-head consistency same as Baseline ({bo:.2f})")
        log(f"    -> Orthogonality constraint has no effect; try adjusting lambda/head_dim")
    else:
        log(f"    Dual-head consistency={do_:.2f} < Baseline={bo:.2f} ({do_-bo:.2f})")
        log(f"    -> Dual-head underperforms Baseline -- review architecture/hyperparams")

    # -- save --
    args.save_results.parent.mkdir(parents=True, exist_ok=True)
    with args.save_results.open("w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"\n  Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
