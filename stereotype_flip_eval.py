"""Stereotype Flip v2 Evaluation.

Tests whether KLUE-RoBERTa trained on KOLD uses identity-term shortcuts.
All pairs: NO explicit slurs. Structural exclusion/concern framing only.

Usage:
    cd c:/nlp_project
    python stereotype_flip_eval.py [--subset 400] [--epochs 5]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from nlp_project.real_model import BaselineClassifier, DualHeadClassifier, orthogonality_loss


# ── data ──────────────────────────────────────────────────────────────────────

def load_kold(path, subset, seed=42):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    examples = [(d["comment"], int(d["OFF"])) for d in raw if d["comment"].strip()]
    if subset:
        rng = random.Random(seed)
        pos = [e for e in examples if e[1] == 1]
        neg = [e for e in examples if e[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        examples = pos[:subset//2] + neg[:subset//2]
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


# ── training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, lambda_ortho, accum_steps=1):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_ortho = n = 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits, z_cls, z_sem = model(ids, mask)
        cls_l = criterion(logits, y)
        o_l = orthogonality_loss(z_cls, z_sem) if z_cls is not None else torch.tensor(0.0, device=device)
        loss = (cls_l + lambda_ortho * o_l) / accum_steps
        loss.backward()
        bs = ids.size(0)
        total_loss += (cls_l.item() + lambda_ortho * o_l.item()) * bs
        total_ortho += o_l.item() * bs
        n += bs
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        del ids, mask, y, logits, z_cls, z_sem, loss, cls_l, o_l
    gc.collect()
    n = max(n, 1)
    return total_loss / n, total_ortho / n


def val_eval(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = correct = total = 0
    tp = fp = fn = tn = 0
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
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
    n = max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    macro_f1_neg = 2*(tn/max(tn+fn,1))*(tn/max(tn+fp,1)) / max((tn/max(tn+fn,1))+(tn/max(tn+fp,1)), 1e-8)
    macro_f1 = (f1 + macro_f1_neg) / 2
    return {"loss": total_loss/n, "acc": correct/n, "macro_f1": macro_f1, "f1_pos": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_model(name, model, train_loader, val_loader, device, epochs, lr, lambda_ortho, log, accum_steps=1):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01
    )
    best_f1 = 0.0
    best_state = None
    log(f"\n{'='*60}")
    log(f"  {name}")
    log(f"  epochs={epochs}  lr={lr}  lambda_ortho={lambda_ortho}  accum={accum_steps}")
    log(f"{'='*60}")
    log(f"{'Ep':>3}  {'TrLoss':>8}  {'OrthoL':>8}  {'ValLoss':>8}  {'MacroF1':>8}  {'Acc':>6}")
    for epoch in range(1, epochs + 1):
        tr_loss, ortho_l = train_epoch(model, train_loader, optimizer, device, lambda_ortho, accum_steps)
        m = val_eval(model, val_loader, device)
        if m["macro_f1"] >= best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log(f"{epoch:>3}  {tr_loss:>8.4f}  {ortho_l:>8.4f}  {m['loss']:>8.4f}  {m['macro_f1']:>8.3f}  {m['acc']:>6.3f}")
        gc.collect()
    if best_state:
        model.load_state_dict(best_state)
    final = val_eval(model, val_loader, device)
    log(f"  Best macro_F1={best_f1:.4f}  Final acc={final['acc']:.3f}  macro_F1={final['macro_f1']:.3f}")
    log(f"  Confusion: tp={final['tp']} fp={final['fp']} fn={final['fn']} tn={final['tn']}")
    return model, final


# ── stereotype flip evaluation ────────────────────────────────────────────────

def predict(model, text, tokenizer, max_len, device):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        logits, _, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        prob = torch.sigmoid(logits).item()
        return int(prob >= 0.5), round(prob, 3)


def eval_stereotype_flip(model, pairs, tokenizer, max_len, device, model_name, log):
    consistent = 0
    flips = []
    bias_type_stats = defaultdict(lambda: {"total": 0, "consistent": 0})

    for p in pairs:
        op, oprob = predict(model, p["original"], tokenizer, max_len, device)
        cp, cprob = predict(model, p["counterfactual"], tokenizer, max_len, device)
        ok = (op == cp)
        if ok:
            consistent += 1
        else:
            flips.append({**p, "orig_pred": op, "orig_prob": oprob,
                          "cf_pred": cp, "cf_prob": cprob})
        bt = p.get("bias_type", "unknown")
        bias_type_stats[bt]["total"] += 1
        bias_type_stats[bt]["consistent"] += int(ok)

    total = len(pairs)
    score = consistent / total

    log(f"\n  [{model_name}]  consistency={score:.2f}  ({consistent}/{total})")

    if flips:
        log(f"  Flipped pairs ({len(flips)}/{total}):")
        for f in flips:
            log(f"    [FLIP] id={f['id']:>3}  orig={f['orig_pred']}({f['orig_prob']:.2f})"
                f"  cf={f['cf_pred']}({f['cf_prob']:.2f})"
                f"  | {f['identity_orig']} -> {f['identity_cf']}")
            log(f"           bias_type: {f['bias_type']}")
            log(f"           orig: {f['original'][:60]}")
            log(f"           cf  : {f['counterfactual'][:60]}")
    else:
        log(f"  No flips detected.")

    # bias_type breakdown
    log(f"\n  Bias-type breakdown:")
    for bt, s in sorted(bias_type_stats.items()):
        c = s["consistent"]
        t = s["total"]
        log(f"    {bt:<40} {c}/{t}  ({c/t:.2f})")

    return {"model": model_name, "consistency": score,
            "consistent": consistent, "total": total, "flips": flips}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           type=Path,  default=ROOT / "data/kold_v1.json")
    parser.add_argument("--testset",        type=Path,  default=ROOT / "data/stereotype_flip_v2.json")
    parser.add_argument("--subset",         type=int,   default=400)
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--accum-steps",    type=int,   default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size * accum_steps.")
    parser.add_argument("--max-len",        type=int,   default=128)
    parser.add_argument("--head-dim",       type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--lambda-ortho",   type=float, default=0.05)
    parser.add_argument("--model-name",     type=str,   default="klue/roberta-base")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the transformer encoder; only train classification heads (probe mode).")
    parser.add_argument("--out",            type=Path,  default=ROOT / "results/stereotype_flip_v2_eval.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_lines = []
    def log(line=""):
        print(line)
        output_lines.append(str(line))

    log("Stereotype Flip v2 Evaluation")
    log(f"  device={device}  model={args.model_name}")
    log(f"  subset={args.subset}  epochs={args.epochs}  lambda_ortho={args.lambda_ortho}")
    log(f"  batch_size={args.batch_size}  accum_steps={args.accum_steps}"
        f"  (eff_batch={args.batch_size * args.accum_steps})  max_len={args.max_len}")
    log(f"  freeze_encoder={args.freeze_encoder}")

    # data
    subset = args.subset if args.subset > 0 else None
    examples = load_kold(args.data, subset, seed=args.seed)
    train_data, val_data = split_data(examples, seed=args.seed)
    log(f"\nKOLD: total={len(examples)}  train={len(train_data)}  val={len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = KOLDDataset(train_data, tokenizer, args.max_len)
    val_ds   = KOLDDataset(val_data,   tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # load stereotype flip testset
    with args.testset.open(encoding="utf-8") as f:
        hn = json.load(f)
    pairs = hn["pairs"]
    log(f"\nStereotype Flip v2: {len(pairs)} pairs")
    log(f"  Design: {hn['meta']['design_principle'][:80]}")
    log(f"  Hypothesis: {hn['meta']['hypothesis']}")

    def maybe_freeze(model):
        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            log(f"  Encoder frozen: trainable params = {trainable:,} / {total:,}")
        return model

    # ── Baseline ──
    baseline = maybe_freeze(BaselineClassifier(model_name=args.model_name).to(device))
    baseline, base_final = train_model(
        "Baseline (Single-head)", baseline, train_loader, val_loader,
        device, args.epochs, args.lr, 0.0, log, accum_steps=args.accum_steps
    )

    log(f"\n{'='*60}")
    log(f"  STEREOTYPE FLIP EVALUATION")
    log(f"{'='*60}")

    res_base = eval_stereotype_flip(
        baseline, pairs, tokenizer, args.max_len, device, "Baseline", log
    )

    del baseline; gc.collect()

    # ── Dual-head ──
    torch.manual_seed(args.seed)
    dual = maybe_freeze(DualHeadClassifier(model_name=args.model_name, head_dim=args.head_dim).to(device))
    dual, dual_final = train_model(
        f"Dual-head + Ortho (lambda={args.lambda_ortho})", dual, train_loader, val_loader,
        device, args.epochs, args.lr, args.lambda_ortho, log, accum_steps=args.accum_steps
    )

    res_dual = eval_stereotype_flip(
        dual, pairs, tokenizer, args.max_len, device, "Dual-head", log
    )

    del dual; gc.collect()

    # ── Summary ──
    log(f"\n{'='*60}")
    log(f"  SUMMARY")
    log(f"{'='*60}")
    log(f"\n  KOLD val performance:")
    log(f"    Baseline  : macro_F1={base_final['macro_f1']:.3f}  acc={base_final['acc']:.3f}")
    log(f"    Dual-head : macro_F1={dual_final['macro_f1']:.3f}  acc={dual_final['acc']:.3f}")
    log(f"\n  Stereotype Flip consistency:")
    log(f"    Baseline  : {res_base['consistency']:.2f}  ({res_base['consistent']}/{res_base['total']})")
    log(f"    Dual-head : {res_dual['consistency']:.2f}  ({res_dual['consistent']}/{res_dual['total']})")
    delta = res_dual["consistency"] - res_base["consistency"]
    log(f"    Delta     : {delta:+.2f}")

    log(f"\n  H1 check (Baseline shortcut exposure):")
    if res_base["consistency"] < 0.70:
        log(f"    CONFIRMED: Baseline consistency={res_base['consistency']:.2f} < 0.70")
        log(f"    -> Identity-term shortcut learning demonstrated.")
        h1_ok = True
    elif res_base["consistency"] < 0.85:
        log(f"    PARTIAL: Baseline consistency={res_base['consistency']:.2f} (target < 0.70)")
        log(f"    -> Some shortcut exposure but threshold not met.")
        h1_ok = False
    else:
        log(f"    NOT MET: Baseline consistency={res_base['consistency']:.2f} >= 0.85")
        log(f"    -> Structural framing still insufficient; need harder pairs.")
        h1_ok = False

    log(f"\n  H1 check (Dual-head recovery):")
    if delta > 0 and res_dual["consistency"] >= 0.80:
        log(f"    CONFIRMED: Dual-head consistency={res_dual['consistency']:.2f} >= 0.80 (+{delta:.2f})")
        log(f"    -> Orthogonality constraint recovers consistency while shortcut is exposed.")
    elif delta > 0:
        log(f"    PARTIAL: Dual-head improves by {delta:+.2f} but below 0.80 target.")
    else:
        log(f"    NOT MET: Dual-head did not improve over Baseline ({delta:+.2f}).")
        log(f"    -> Increase epochs, lambda, or head_dim.")

    log(f"\n  Flip pattern analysis:")
    if res_base["flips"]:
        id_terms = [f["identity_orig"] for f in res_base["flips"]]
        from collections import Counter
        term_counts = Counter(id_terms)
        log(f"    Most flipped identity terms (Baseline): {dict(term_counts.most_common(5))}")
    else:
        log(f"    No flips in Baseline.")

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"\n  Saved to: {args.out}")


if __name__ == "__main__":
    main()
