"""Real-data pilot for dual-head orthogonality Korean hate speech detection.

Uses:
  - Encoder: klue/roberta-base
  - Dataset: KOLD (kold_v1.json, 40k examples)
  - Pilot subset: first N examples (stratified by label)
  - Orthogonality loss on z_cls, z_sem (head_dim=128)
  - MLM: deferred

Usage:
    cd c:/nlp_project
    python real_pilot_run.py [--subset 400] [--epochs 5] [--lambda-ortho 0.05]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.real_model import BaselineClassifier, DualHeadClassifier, orthogonality_loss

# ── counterfactual pairs (manually constructed, identity-term substitution) ──
# 10 pairs: same sentence structure, only identity term swapped
# Purpose: check if predictions are consistent across identity-term substitutions
# These match the proposal's counterfactual consistency evaluation (H1)
COUNTERFACTUAL_PAIRS = [
    # (original, counterfactual, expected_label)
    # offensive — both should be predicted offensive
    ("여성들은 다 멍청하고 쓸모없다",       "남성들은 다 멍청하고 쓸모없다",       1),
    ("노인들은 빨리 죽어야 해",             "청년들은 빨리 죽어야 해",             1),
    ("외국인들이 우리나라에서 나가야 한다", "한국인들이 우리나라에서 나가야 한다", 1),
    ("장애인들은 정상인보다 못하다",        "비장애인들은 장애인보다 못하다",      1),
    ("여자들은 집에나 있어야 해",           "남자들은 집에나 있어야 해",           1),
    # non-offensive — both should be predicted non-offensive
    ("여성의 사회 참여가 늘어나는 것은 바람직하다",
     "남성의 사회 참여가 늘어나는 것은 바람직하다", 0),
    ("노인들의 경험과 지혜는 소중하다",
     "청년들의 경험과 열정은 소중하다", 0),
    ("외국인도 인권을 보장받아야 한다",
     "한국인도 인권을 보장받아야 한다", 0),
    ("장애인도 충분히 능력을 발휘할 수 있다",
     "비장애인도 충분히 능력을 발휘할 수 있다", 0),
    ("이슬람교도들도 우리 사회의 구성원이다",
     "기독교도들도 우리 사회의 구성원이다", 0),
]


# ── data ──────────────────────────────────────────────────────────────────────

def load_kold(path: Path, subset: int | None = None, seed: int = 42) -> list[tuple[str, int]]:
    """Load KOLD, return (comment_text, label) pairs.

    Stratified subsample if subset is given.
    Label: 1 = offensive (OFF=True), 0 = non-offensive.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    examples = [(d["comment"], int(d["OFF"])) for d in raw if d["comment"].strip()]

    if subset is not None:
        rng = random.Random(seed)
        pos = [e for e in examples if e[1] == 1]
        neg = [e for e in examples if e[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        half = subset // 2
        examples = pos[:half] + neg[:half]
        rng.shuffle(examples)

    return examples


def split_data(
    examples: list[tuple[str, int]], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list, list]:
    items = examples[:]
    random.Random(seed).shuffle(items)
    cut = int(len(items) * (1 - val_ratio))
    return items[:cut], items[cut:]


# ── dataset/loader ────────────────────────────────────────────────────────────

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class KOLDDataset(Dataset):
    def __init__(
        self,
        examples: list[tuple[str, int]],
        tokenizer,
        max_len: int = 128,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float32),
        }


# ── training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_ortho: float,
) -> tuple[float, float, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_cls = total_ortho = 0.0
    n = 0

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y    = batch["label"].to(device)

        optimizer.zero_grad()
        logits, z_cls, z_sem = model(ids, mask)

        cls_l = criterion(logits, y)
        if z_cls is not None:
            o_l = orthogonality_loss(z_cls, z_sem)
        else:
            o_l = torch.tensor(0.0, device=device)

        loss = cls_l + lambda_ortho * o_l
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = ids.size(0)
        total_loss  += loss.item() * bs
        total_cls   += cls_l.item() * bs
        total_ortho += o_l.item() * bs
        n += bs

    n = max(n, 1)
    return total_loss / n, total_cls / n, total_ortho / n


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = correct = total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y    = batch["label"].to(device)

            logits, _, _ = model(ids, mask)
            loss = criterion(logits, y)
            total_loss += loss.item() * ids.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total   += ids.size(0)
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
    return {
        "loss": total_loss / n,
        "acc": correct / n,
        "f1_pos": f1,
        "macro_f1": macro_f1,
        "prec": prec,
        "rec": rec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── counterfactual consistency ─────────────────────────────────────────────────

def predict_texts(
    model: nn.Module,
    texts: list[str],
    tokenizer,
    max_len: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, max_length=max_len, padding="max_length",
                            truncation=True, return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits, _, _ = model(ids, mask)
            preds.append(int((torch.sigmoid(logits) >= 0.5).item()))
    return preds


def run_counterfactual(
    model: nn.Module,
    pairs: list[tuple[str, str, int]],
    tokenizer,
    max_len: int,
    device: torch.device,
    name: str,
) -> dict:
    originals  = [p[0] for p in pairs]
    counters   = [p[1] for p in pairs]
    exp_labels = [p[2] for p in pairs]

    orig_preds  = predict_texts(model, originals, tokenizer, max_len, device)
    count_preds = predict_texts(model, counters,  tokenizer, max_len, device)

    consistent = sum(o == c for o, c in zip(orig_preds, count_preds))
    score = consistent / len(pairs)

    print(f"\n  [{name}] consistency = {score:.2f} ({consistent}/{len(pairs)})")
    for orig, counter, exp, op, cp in zip(originals, counters, exp_labels, orig_preds, count_preds):
        flag = "OK  " if op == cp else "FLIP"
        print(f"    [{flag}] expected={exp} orig={op} cf={cp} | '{orig[:28]}' -> '{counter[:28]}'")

    return {"score": score, "consistent": consistent, "total": len(pairs)}


# ── full experiment run ────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_ortho: float,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    best_f1 = 0.0
    best_state = None

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  lambda_ortho={lambda_ortho}  epochs={epochs}  lr={lr}")
    print(f"{'='*65}")
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'ClsL':>8}  {'OrthoL':>8}  {'VLoss':>8}  {'Acc':>6}  {'MacroF1':>8}  {'OrthoTrend':>10}")

    for epoch in range(1, epochs + 1):
        tr_loss, cls_l, ortho_l = train_epoch(
            model, train_loader, optimizer, device, lambda_ortho
        )
        val_m = evaluate(model, val_loader, device)

        if val_m["macro_f1"] >= best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "cls_loss": cls_l,
            "ortho_loss": ortho_l,
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        # trend arrow: is ortho decreasing from last epoch?
        if len(history) >= 2 and lambda_ortho > 0:
            prev = history[-2]["ortho_loss"]
            trend = "v" if ortho_l < prev else ("^" if ortho_l > prev else "=")
        else:
            trend = "-"

        print(
            f"{epoch:>3}  {tr_loss:>8.4f}  {cls_l:>8.4f}  {ortho_l:>8.4f}"
            f"  {val_m['loss']:>8.4f}  {val_m['acc']:>6.3f}  {val_m['macro_f1']:>8.3f}  {trend:>10}"
        )

    if best_state:
        model.load_state_dict(best_state)
    final = evaluate(model, val_loader, device)
    print(f"\n  Best macro_F1={best_f1:.4f}")
    print(f"  Final: acc={final['acc']:.3f}  macro_F1={final['macro_f1']:.3f}  f1_pos={final['f1_pos']:.3f}")
    print(f"  Confusion: tp={final['tp']} fp={final['fp']} fn={final['fn']} tn={final['tn']}")
    return {"name": name, "history": history, "best_f1": best_f1, "final": final, "model": model}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=Path,  default=ROOT / "data/kold_v1.json")
    parser.add_argument("--subset",       type=int,   default=400,
                        help="Stratified subset size (use 0 for full dataset)")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--max-len",      type=int,   default=128)
    parser.add_argument("--head-dim",     type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--lambda-ortho", type=float, default=0.05)
    parser.add_argument("--model-name",   type=str,   default="klue/roberta-base")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    # ── data ──
    subset = args.subset if args.subset > 0 else None
    examples = load_kold(args.data, subset=subset, seed=args.seed)
    train_data, val_data = split_data(examples, val_ratio=0.2, seed=args.seed)

    n_pos = sum(1 for _, l in examples if l == 1)
    n_neg = len(examples) - n_pos
    print(f"\nDataset subset: {len(examples)} examples (OFF={n_pos}, non-OFF={n_neg})")
    print(f"Train={len(train_data)}  Val={len(val_data)}")

    # ── tokenizer ──
    print(f"\nLoading tokenizer: {args.model_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = KOLDDataset(train_data, tokenizer, max_len=args.max_len)
    val_ds   = KOLDDataset(val_data,   tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── experiment 1: baseline ──
    print(f"\nLoading baseline model: {args.model_name}")
    baseline = BaselineClassifier(model_name=args.model_name).to(device)

    res_base = run_experiment(
        name="[1] Single-head Baseline (klue/roberta-base)",
        model=baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ortho=0.0,
    )

    # ── experiment 2: dual-head + ortho ──
    print(f"\nLoading dual-head model: {args.model_name}")
    torch.manual_seed(args.seed)
    dual = DualHeadClassifier(
        model_name=args.model_name,
        head_dim=args.head_dim,
    ).to(device)

    res_dual = run_experiment(
        name=f"[2] Dual-head + Orthogonality (lambda={args.lambda_ortho})",
        model=dual,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ortho=args.lambda_ortho,
    )

    # ── orthogonality loss saturation check ──
    print(f"\n{'='*65}")
    print("  Orthogonality Loss Saturation Check")
    print(f"{'='*65}")
    ortho_vals = [h["ortho_loss"] for h in res_dual["history"]]
    first2 = sum(ortho_vals[:2]) / min(2, len(ortho_vals))
    last2  = sum(ortho_vals[-2:]) / min(2, len(ortho_vals))
    drop_pct = (first2 - last2) / max(first2, 1e-8) * 100
    print(f"  ortho_loss epoch1={ortho_vals[0]:.4f}  epoch{len(ortho_vals)}={ortho_vals[-1]:.4f}")
    print(f"  Drop: {drop_pct:.1f}%  (>95% = likely trivial saturation)")
    if drop_pct > 95:
        print("  WARNING: ortho_loss collapsed early — constraint may not be providing")
        print("           meaningful decoupling pressure. Monitor with more epochs / higher lambda.")
    else:
        print("  OK: ortho_loss is active throughout training.")

    # ── counterfactual consistency ──
    print(f"\n{'='*65}")
    print("  Counterfactual Consistency Check (10 identity-term pairs)")
    print(f"{'='*65}")
    cf_base = run_counterfactual(
        res_base["model"], COUNTERFACTUAL_PAIRS, tokenizer, args.max_len, device, "Baseline"
    )
    cf_dual = run_counterfactual(
        res_dual["model"], COUNTERFACTUAL_PAIRS, tokenizer, args.max_len, device, "Dual-head"
    )

    # ── final summary ──
    print(f"\n{'='*65}")
    print("  REAL-DATA PILOT SUMMARY")
    print(f"{'='*65}")
    bm = res_base["final"]
    dm = res_dual["final"]
    print(f"\n  Dataset: KOLD (subset={len(examples)}, train={len(train_data)}, val={len(val_data)})")
    print(f"  Encoder: {args.model_name}")
    print(f"  head_dim: {args.head_dim}  lambda_ortho: {args.lambda_ortho}")
    print()
    print(f"  Performance:")
    print(f"    Baseline  : acc={bm['acc']:.3f}  macro_F1={bm['macro_f1']:.3f}  F1_pos={bm['f1_pos']:.3f}")
    print(f"    Dual-head : acc={dm['acc']:.3f}  macro_F1={dm['macro_f1']:.3f}  F1_pos={dm['f1_pos']:.3f}")
    print()
    print(f"  Counterfactual consistency:")
    print(f"    Baseline  : {cf_base['score']:.2f} ({cf_base['consistent']}/{cf_base['total']})")
    print(f"    Dual-head : {cf_dual['score']:.2f} ({cf_dual['consistent']}/{cf_dual['total']})")
    print()
    print(f"  Ortho loss: epoch1={ortho_vals[0]:.4f} -> final={ortho_vals[-1]:.4f} (drop={drop_pct:.1f}%)")
    print()
    print("  Assumptions:")
    print("    A1. Subset = stratified 400 examples from 40k KOLD for pilot speed")
    print("    A2. MLM objective deferred (baseline+dual-head ortho validated first)")
    print("    A3. max_len=128, lr=2e-5, AdamW, grad_clip=1.0")
    print("    A4. head_dim=128 (lower bound of proposal range)")
    print()


if __name__ == "__main__":
    main()
