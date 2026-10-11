#!/usr/bin/env python3
"""W3-01  Run one benchmark configuration and append the result.

This is contribution C2 in execution: the same models, the same training recipe
and the same hyper-parameters, evaluated under three protocols that differ ONLY
in how the data was split.  Any accuracy difference between them is therefore
attributable to the protocol, which is the entire argument of the paper.

Design constraints that shaped this file
----------------------------------------
* Kaggle gives 12 h per session and a 30 GPU-hour weekly budget, so one
  invocation runs exactly ONE (representation, backbone, protocol, seed) tuple
  and appends a row to results.csv.  Nothing is ever recomputed, and an
  interrupted sweep resumes where it stopped.
* Every run writes per-clip test predictions.  Without them we cannot compute a
  recording-level bootstrap confidence interval later, and a single point
  estimate is exactly the practice this paper is arguing against.
* The training recipe is deliberately plain and identical across backbones.
  Tuning per backbone would confound the comparison we are trying to make.

Usage
-----
    python w3_01_run_benchmark.py \
        --cache /kaggle/input/uwatr-logmel \
        --splits /kaggle/input/uwatr-splits/splits.csv.gz \
        --out /kaggle/working \
        --rep logmel --backbone resnet18 --protocol rec_random --seed 0

Backbones: logreg, cnn_small, resnet18, efficientnet_b0, ast
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    raise SystemExit("needs torch")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    """Serves (1, 128, 313) tensors straight out of the memory-mapped cache.

    Reading from the memmap per item keeps RAM flat: the full array never has to
    be resident, which matters because both representations together are ~5.4 GB.
    """

    def __init__(self, arrays: dict[str, np.ndarray], rows: np.ndarray, labels: np.ndarray):
        self.arrays = arrays
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        # both representations are cached; the caller picks which one to read
        x = np.stack([self.arrays[k][r] for k in sorted(self.arrays)], axis=0)
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[i])


def load_split(path: Path, protocol: str, seed: int) -> dict[str, set[str]]:
    """Return {split: {clip_id, ...}} for one protocol/seed."""
    out: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["protocol"] == protocol and int(row["seed"]) == seed:
                out[row["split"]].add(f"{row['rec_key']}#{row['clip_idx']}")
    return out


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """~150K-parameter compact CNN.

    Parameter count is the point: Soti et al. (2026) found an 11.17M-parameter
    ResNet18 gave no improvement over a 157K compact CNN under a recording-level
    split.  Including both lets us reproduce or contradict that on our protocol.
    """

    def __init__(self, n_classes: int, in_ch: int = 1, width: int = 24):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(in_ch, width),        # 64x156
            block(width, width * 2),    # 32x78
            block(width * 2, width * 4),  # 16x39
            block(width * 4, width * 4),  # 8x19
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(h)


def build_model(backbone: str, n_classes: int, in_ch: int):
    """Return (model, is_pretrained).  Pretrained weights are fetched if available."""
    if backbone == "cnn_small":
        return SmallCNN(n_classes, in_ch=in_ch), False

    if backbone in ("resnet18", "efficientnet_b0"):
        import torchvision.models as tvm

        try:
            weights = "IMAGENET1K_V1"
            if backbone == "resnet18":
                m = tvm.resnet18(weights=weights)
                m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
                m.fc = nn.Linear(m.fc.in_features, n_classes)
            else:
                m = tvm.efficientnet_b0(weights=weights)
                m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
            return m, True
        except Exception as exc:  # offline runtime -> fall back to random init
            print(f"[warn] pretrained weights unavailable ({exc}); training from scratch")
            if backbone == "resnet18":
                m = tvm.resnet18(weights=None)
                m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
                m.fc = nn.Linear(m.fc.in_features, n_classes)
            else:
                m = tvm.efficientnet_b0(weights=None)
                m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
            return m, False

    if backbone == "ast":
        from transformers import ASTConfig, ASTForAudioClassification

        try:
            model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                num_labels=n_classes, ignore_mismatched_sizes=True,
            )
            return model, True
        except Exception as exc:
            print(f"[warn] AST weights unavailable ({exc}); random init")
            cfg = ASTConfig(num_labels=n_classes)
            return ASTForAudioClassification(cfg), False

    raise ValueError(f"unknown backbone {backbone!r}")


def forward_logits(model, backbone: str, x: torch.Tensor) -> torch.Tensor:
    if backbone == "ast":
        # AST expects (B, T, F); our tensor is (B, C, F, T)
        return model(x.mean(dim=1)).logits
    return model(x)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 without sklearn, so the metric is defined right here."""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(model, backbone, loader, device, n_classes):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = forward_logits(model, backbone, x)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    p = np.concatenate(preds) if preds else np.zeros(0, dtype=int)
    t = np.concatenate(trues) if trues else np.zeros(0, dtype=int)
    acc = float((p == t).mean()) if len(t) else 0.0
    return acc, macro_f1(t, p, n_classes), t, p


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Run one UATR benchmark configuration.")
    ap.add_argument("--cache", type=Path, required=True, help="dir holding logmel.npy / spec.npy / index.csv")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rep", default="logmel", choices=["logmel", "spec"])
    ap.add_argument("--backbone", required=True,
                    choices=["logreg", "cnn_small", "resnet18", "efficientnet_b0", "ast"])
    ap.add_argument("--protocol", required=True, choices=["clip_random", "rec_random", "temporal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-train-clips", type=int, default=0, help="subsample train (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ data
    with (args.cache / "index.csv").open(encoding="utf-8") as fh:
        index = list(csv.DictReader(fh))
    clip_ids = [r["clip_id"] for r in index]
    classes = sorted({r["class_dir"] for r in index})
    class_to_i = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_to_i[r["class_dir"]] for r in index], dtype=np.int64)
    rec_of_clip = np.array([r["rec_key"] for r in index])
    id_to_row = {cid: i for i, cid in enumerate(clip_ids)}

    sp = load_split(args.splits, args.protocol, args.seed)
    def rows_for(split: str) -> np.ndarray:
        return np.array(sorted(id_to_row[c] for c in sp[split] if c in id_to_row), dtype=np.int64)

    train_rows, val_rows, test_rows = rows_for("train"), rows_for("val"), rows_for("test")
    if args.max_train_clips and len(train_rows) > args.max_train_clips:
        train_rows = np.random.RandomState(args.seed).choice(
            train_rows, args.max_train_clips, replace=False
        )

    rep_file = args.cache / f"{args.rep}.npy"
    arr = np.load(rep_file, mmap_mode="r")
    arrays = {args.rep: arr}

    n_classes = len(classes)
    run_id = f"{args.rep}__{args.backbone}__{args.protocol}__s{args.seed}"
    print(f"[run] {run_id}")
    print(f"[data] train={len(train_rows):,} val={len(val_rows):,} test={len(test_rows):,} "
          f"classes={n_classes} device={device}")
    # leakage is reported alongside every result so no number can be read without
    # the protocol that produced it
    train_recs = set(rec_of_clip[train_rows])
    test_recs = set(rec_of_clip[test_rows])
    leaked = len(test_recs & train_recs)
    print(f"[protocol] test recordings sharing a recording with train: {leaked}/{len(test_recs)}")

    # ---------------------------------------------------------------- model
    if args.backbone == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        t0 = time.time()
        Xtr = np.stack([arr[r] for r in train_rows]).reshape(len(train_rows), -1)
        Xte = np.stack([arr[r] for r in test_rows]).reshape(len(test_rows), -1)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial", n_jobs=-1)
        clf.fit(sc.transform(Xtr), labels[train_rows])
        t_pred = clf.predict(sc.transform(Xte))
        t_true = labels[test_rows]
        acc = float((t_pred == t_true).mean())
        f1 = macro_f1(t_true, t_pred, n_classes)
        n_params = int(Xtr.shape[1] * n_classes)
        wall = time.time() - t0
        best_val = float("nan")
        epochs_done = 1
    else:
        model, pretrained = build_model(args.backbone, n_classes, in_ch=1)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {args.backbone} params={n_params:,} pretrained={pretrained}")

        tr_ld = DataLoader(ClipDataset(arrays, train_rows, labels[train_rows]),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=False)
        va_ld = DataLoader(ClipDataset(arrays, val_rows, labels[val_rows]),
                           batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        te_ld = DataLoader(ClipDataset(arrays, test_rows, labels[test_rows]),
                           batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        best_val, best_state, epochs_done = -1.0, None, 0
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            tot = n_seen = 0
            for x, y in tr_ld:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = forward_logits(model, args.backbone, x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss) * len(y)
                n_seen += len(y)
            sched.step()
            v_acc, v_f1, _, _ = evaluate(model, args.backbone, va_ld, device, n_classes)
            epochs_done = ep
            marker = ""
            if v_f1 > best_val:
                best_val = v_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                marker = " *"
            print(f"  ep {ep:>3} loss={tot/max(n_seen,1):.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f}{marker}",
                  flush=True)
        wall = time.time() - t0

        if best_state is not None:
            model.load_state_dict(best_state)
        acc, f1, t_true, t_pred = evaluate(model, args.backbone, te_ld, device, n_classes)

    # ---------------------------------------------------------------- record
    args.out.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    # per-clip predictions + the recording each clip came from: needed for the
    # recording-level bootstrap in w3_02
    np.savez_compressed(
        pred_dir / f"{run_id}.npz",
        y_true=t_true, y_pred=t_pred,
        rec_key=np.array([rec_of_clip[r] for r in test_rows]),
        classes=np.array(classes),
    )

    result = {
        "run_id": run_id,
        "rep": args.rep,
        "backbone": args.backbone,
        "protocol": args.protocol,
        "seed": args.seed,
        "n_train_clips": int(len(train_rows)),
        "n_val_clips": int(len(val_rows)),
        "n_test_clips": int(len(test_rows)),
        "n_test_recordings": int(len(test_recs)),
        "test_recordings_leaked": int(leaked),
        "leak_pct": round(100 * leaked / max(len(test_recs), 1), 2),
        "n_classes": n_classes,
        "n_params": int(n_params),
        "epochs": int(epochs_done),
        "best_val_f1": None if np.isnan(best_val) else round(float(best_val), 4),
        "test_acc": round(float(acc), 4),
        "test_macro_f1": round(float(f1), 4),
        "wall_min": round(wall / 60, 2),
    }
    for c, i in class_to_i.items():
        m = t_true == i
        result[f"acc_{c}"] = round(float((t_pred[m] == i).mean()), 4) if m.any() else None
        result[f"n_{c}"] = int(m.sum())

    results_path = args.out / "results.csv"
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)

    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nappended to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
