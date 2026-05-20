"""Pilot experiment runner for dual-head orthogonality hate speech detection.

Usage:
    cd c:/nlp_project
    python pilot_run.py [--epochs 30] [--lambda-ortho 0.05] [--seed 42]

Pilot scope:
    1. Single-head baseline (BiLSTM)
    2. Dual-head + orthogonality loss (BiLSTM)
    3. Counterfactual consistency check on identity-term substitution pairs
    4. Report training stability, loss curves, metrics
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data import TextDataset, Vocab, build_vocab, make_loader, tokenize
from nlp_project.pilot_model import BaselineModel, DualHeadModel, orthogonality_loss

# ── counterfactual pairs (identity-term substitutions) ────────────────────────
# A small fixed set of sentence pairs where only the identity term differs.
# Purpose: check if prediction changes when identity term is swapped.
# These are pilot-level manual pairs — not a full counterfactual evaluation set.
COUNTERFACTUAL_PAIRS = [
    # (original, counterfactual, expected_label_preserved)
    # offensive pairs — both should be predicted as offensive (label=1)
    ("여성들은 다 멍청하고 쓸모없다",   "남성들은 다 멍청하고 쓸모없다",   1),
    ("여자들은 집에나 있어야 해",       "남자들은 집에나 있어야 해",       1),
    ("노인들은 빨리 죽어야 해",         "청년들은 빨리 죽어야 해",         1),
    ("늙은이들은 사회에 짐이 될 뿐이야","젊은이들은 사회에 짐이 될 뿐이야",1),
    ("외국인들이 우리나라에서 나가야 한다","한국인들이 우리나라에서 나가야 한다",1),
    ("장애인들은 정상인보다 못하다",    "비장애인들은 장애인보다 못하다",   1),
    # non-offensive pairs — both should be predicted as non-offensive (label=0)
    ("여성의 사회 참여가 늘어나는 것은 바람직하다",
     "남성의 사회 참여가 늘어나는 것은 바람직하다", 0),
    ("노인들의 경험과 지혜는 소중하다",
     "청년들의 경험과 열정은 소중하다", 0),
    ("외국인도 인권을 보장받아야 한다",
     "한국인도 인권을 보장받아야 한다", 0),
    ("장애인도 충분히 능력을 발휘할 수 있다",
     "비장애인도 충분히 능력을 발휘할 수 있다", 0),
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_pilot_csv(path: Path) -> list[tuple[str, int]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["text"], int(row["label"])))
    return items


def split_data(
    examples: list[tuple[str, int]], test_ratio: float, seed: int
) -> tuple[list, list]:
    items = examples[:]
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = max(1, int(len(items) * (1.0 - test_ratio)))
    cut = min(cut, len(items) - 1)
    return items[:cut], items[cut:]


# ── training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_ortho: float = 0.0,
) -> tuple[float, float, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_cls = total_ortho = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, z_cls, z_sem = model(x)

        cls_loss = criterion(logits, y)
        if z_cls is not None:
            o_loss = orthogonality_loss(z_cls, z_sem)
        else:
            o_loss = torch.tensor(0.0, device=device)

        loss = cls_loss + lambda_ortho * o_loss
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls  += cls_loss.item() * bs
        total_ortho += o_loss.item() * bs
        n += bs

    n = max(n, 1)
    return total_loss / n, total_cls / n, total_ortho / n


def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_total = correct = total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            loss = criterion(logits, y)
            loss_total += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += x.size(0)
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()

    n = max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "loss": loss_total / n,
        "acc":  correct / n,
        "f1":   f1,
        "prec": prec,
        "rec":  rec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── counterfactual consistency ────────────────────────────────────────────────

def predict_texts(
    model: nn.Module, texts: list[str], vocab: Vocab, max_len: int, device: torch.device
) -> list[int]:
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            ids = vocab.encode(text, max_len)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _, _ = model(x)
            pred = int((torch.sigmoid(logits) >= 0.5).item())
            preds.append(pred)
    return preds


def counterfactual_consistency(
    model: nn.Module,
    pairs: list[tuple[str, str, int]],
    vocab: Vocab,
    max_len: int,
    device: torch.device,
) -> dict:
    originals  = [p[0] for p in pairs]
    counters   = [p[1] for p in pairs]
    exp_labels = [p[2] for p in pairs]

    orig_preds  = predict_texts(model, originals, vocab, max_len, device)
    count_preds = predict_texts(model, counters,  vocab, max_len, device)

    consistent = sum(o == c for o, c in zip(orig_preds, count_preds))
    score = consistent / len(pairs)

    rows = []
    for i, (orig, counter, exp, op, cp) in enumerate(
        zip(originals, counters, exp_labels, orig_preds, count_preds)
    ):
        rows.append({
            "orig": orig, "counter": counter,
            "expected": exp, "pred_orig": op, "pred_counter": cp,
            "consistent": op == cp,
        })
    return {"score": score, "consistent": consistent, "total": len(pairs), "rows": rows}


# ── full training run ─────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_ortho: float,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_f1 = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  lambda_ortho={lambda_ortho}  epochs={epochs}  lr={lr}")
    print(f"{'='*60}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'ClsL':>8}  {'OrthoL':>8}  {'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>7}")

    for epoch in range(1, epochs + 1):
        tr_loss, cls_loss, ortho_loss_val = train_epoch(
            model, train_loader, optimizer, device, lambda_ortho
        )
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "cls_loss": cls_loss,
            "ortho_loss": ortho_loss_val,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"{epoch:>4}  {tr_loss:>8.4f}  {cls_loss:>8.4f}  {ortho_loss_val:>8.4f}"
            f"  {val_metrics['loss']:>8.4f}  {val_metrics['acc']:>7.3f}  {val_metrics['f1']:>7.3f}"
        )

    # restore best model for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate(model, val_loader, device)
    print(f"\n  Best val F1={best_f1:.4f}  Final: acc={final['acc']:.3f}  f1={final['f1']:.3f}")
    print(f"  confusion: tp={final['tp']} fp={final['fp']} fn={final['fn']} tn={final['tn']}")

    return {"name": name, "history": history, "best_f1": best_f1, "final": final, "model": model}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot experiment: dual-head hate speech detection")
    parser.add_argument("--data",         type=Path,  default=ROOT / "data/pilot_kold_synthetic.csv")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--max-len",      type=int,   default=32)
    parser.add_argument("--embed-dim",    type=int,   default=128)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--head-dim",     type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lambda-ortho", type=float, default=0.05)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──
    examples = load_pilot_csv(args.data)
    print(f"\nDataset: {len(examples)} examples loaded from {args.data}")
    n_pos = sum(1 for _, l in examples if l == 1)
    n_neg = sum(1 for _, l in examples if l == 0)
    print(f"  Label distribution: offensive={n_pos}, non-offensive={n_neg}")

    texts = [x for x, _ in examples]
    vocab = build_vocab(texts)
    print(f"  Vocab size: {len(vocab.itos)}")

    train_data, val_data = split_data(examples, test_ratio=0.2, seed=args.seed)
    print(f"  Train={len(train_data)}  Val={len(val_data)}")

    train_ds = TextDataset(train_data, vocab=vocab, max_len=args.max_len)
    val_ds   = TextDataset(val_data,   vocab=vocab, max_len=args.max_len)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── experiment 1: baseline ──
    baseline = BaselineModel(
        vocab_size=len(vocab.itos),
        pad_idx=vocab.pad_idx,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    res_base = run_experiment(
        name="[1] Single-head Baseline (BiLSTM)",
        model=baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ortho=0.0,
    )

    # ── experiment 2: dual-head + orthogonality ──
    torch.manual_seed(args.seed)
    dual = DualHeadModel(
        vocab_size=len(vocab.itos),
        pad_idx=vocab.pad_idx,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        head_dim=args.head_dim,
    ).to(device)

    res_dual = run_experiment(
        name=f"[2] Dual-head + Orthogonality (λ={args.lambda_ortho})",
        model=dual,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ortho=args.lambda_ortho,
    )

    # ── counterfactual consistency check ──
    print(f"\n{'='*60}")
    print("  Counterfactual Consistency Check")
    print(f"{'='*60}")
    print(f"  {len(COUNTERFACTUAL_PAIRS)} identity-term substitution pairs\n")

    for name, model in [("Baseline", res_base["model"]), ("Dual-head", res_dual["model"])]:
        cf = counterfactual_consistency(model, COUNTERFACTUAL_PAIRS, vocab, args.max_len, device)
        print(f"  [{name}] consistency={cf['score']:.2f}  ({cf['consistent']}/{cf['total']} pairs consistent)")
        for row in cf["rows"]:
            flag = "OK" if row["consistent"] else "FLIP"
            print(
                f"    [{flag}] label={row['expected']}"
                f"  orig_pred={row['pred_orig']}  counter_pred={row['pred_counter']}"
                f"  |  '{row['orig'][:30]}...' → '{row['counter'][:30]}...'"
            )

    # ── training stability check ──
    print(f"\n{'='*60}")
    print("  Training Stability Summary")
    print(f"{'='*60}")

    for res in [res_base, res_dual]:
        h = res["history"]
        losses = [e["train_loss"] for e in h]
        first3 = sum(losses[:3]) / 3
        last3  = sum(losses[-3:]) / 3
        decreasing = last3 < first3
        monotone_drops = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])

        print(f"\n  {res['name']}")
        print(f"    loss first-3 avg: {first3:.4f} -> last-3 avg: {last3:.4f}")
        print(f"    loss decreasing overall: {decreasing}")
        print(f"    epochs with loss decrease: {monotone_drops}/{len(losses)-1}")
        print(f"    best val F1: {res['best_f1']:.4f}")

        if res["name"].startswith("[2]"):
            ortho_vals = [e["ortho_loss"] for e in h]
            print(f"    ortho_loss first-3 avg: {sum(ortho_vals[:3])/3:.4f} "
                  f"-> last-3 avg: {sum(ortho_vals[-3:])/3:.4f}")

    # ── final summary ──
    print(f"\n{'='*60}")
    print("  PILOT RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Assumptions made:")
    print("    A1. Encoder = BiLSTM (not KLUE-RoBERTa; transformers not installed)")
    print("    A2. Dataset = synthetic ~75 examples (KOLD not available locally)")
    print("    A3. MLM auxiliary objective OMITTED (requires subword tokenizer + LM head)")
    print("    A4. head_dim=128 (proposal suggests d in {128, 256}; used lower bound)")
    print("    A5. lambda_ortho=0.05 (within proposal's suggested range {0.01, 0.05})")
    print("    A6. Tokenizer = regex character n-gram (proposal uses KLUE tokenizer)")
    print()
    print(f"  Results:")
    bm = res_base["final"]
    dm = res_dual["final"]
    print(f"    Baseline  : val acc={bm['acc']:.3f}  val F1={bm['f1']:.3f}")
    print(f"    Dual-head : val acc={dm['acc']:.3f}  val F1={dm['f1']:.3f}")

    base_cf = counterfactual_consistency(res_base["model"], COUNTERFACTUAL_PAIRS, vocab, args.max_len, device)
    dual_cf = counterfactual_consistency(res_dual["model"], COUNTERFACTUAL_PAIRS, vocab, args.max_len, device)
    print(f"    Baseline  counterfactual consistency: {base_cf['score']:.2f}")
    print(f"    Dual-head counterfactual consistency: {dual_cf['score']:.2f}")
    print()
    print("  Training stability: see above (both models trained, loss logged per epoch)")
    print("  Orthogonality loss: applied to z_cls and z_sem (head_dim representations)")
    print()


if __name__ == "__main__":
    main()
