# ============================================================================================== #
# W3-J — fine-tuned AST on the honest protocol (vessel_disjoint)                                  #
#                                                                                                 #
# Purpose: seal the last reviewer hole.  W3-H showed that FROZEN AudioSet-pretrained AST          #
# embeddings + linear probe decode the tail classes no better than our small CNN (tail top-2      #
# 0.116 vs 0.114).  A reviewer will say: "you only probed frozen features - fine-tuning could    #
# still unlock them."  This script fine-tunes the full encoder on the same frozen draws and      #
# re-measures the tail.                                                                           #
#                                                                                                 #
# Pre-registered verdict (written before running, thresholds identical to W3-H):                  #
#   tail-6 mean top-2 >= 0.30  -> fine-tuning DOES unlock the tail: the method door reopens       #
#                                 ("fine-tuned foundation models + honest evaluation" is a        #
#                                 positive-result paper).                                         #
#   tail-6 mean top-2 <= 0.15  -> the hole is SEALED: neither capacity, nor pretraining, nor      #
#                                 full fine-tuning encodes the tail.  The audit paper's claim     #
#                                 stands with the strongest possible form.                        #
#   in between                -> PARTIAL: report per-class which classes fine-tuning rescues.     #
#                                                                                                 #
# Budget model (P100/T4, AMP on): ~20-30 min per epoch over ~37k training clips.  Defaults        #
# (2 seeds x 8 epochs) fit one <=12 h Kaggle GPU session.  A session-budget guard stops before    #
# starting a seed that would not finish; rerun resumes by skipping completed seeds.               #
#                                                                                                 #
# Attach: under-water-features ONLY.  (W3I_data is NOT needed here.)                              #
# CPU smoke: set SMOKE_N_CLIP = 256 and EPOCHS = 1 at the top.                                    #
# ============================================================================================== #

import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

T0 = time.time()


def tick(msg):
    print(f"  [{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------- knobs -------- #
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1", help="frozen split seeds to fine-tune on")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr-backbone", type=float, default=2e-5)
ap.add_argument("--lr-head", type=float, default=1e-4)
ap.add_argument("--freeze-layers", type=int, default=0,
                help="freeze the first N of 12 encoder layers (0 = full fine-tune)")
ap.add_argument("--budget-min", type=float, default=660.0,
                help="stop before starting a seed that would exceed this many minutes")
ap.add_argument("--smoke", type=int, default=0,
                help="if >0, cap train/val/test to this many clips (wiring check)")
ap.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593",
                help="HF repo id, or a local directory with the model files")
ap.add_argument("--features-root", default="",
                help="path to uwatr-features (else autodetected under /kaggle/input)")
args, _ = ap.parse_known_args()

SEEDS = [int(s) for s in args.seeds.split(",")]
EPOCHS = args.epochs
BATCH = args.batch
LR_BB, LR_HEAD = args.lr_backbone, args.lr_head
FREEZE_LAYERS = args.freeze_layers
BUDGET_MIN = args.budget_min
SMOKE_N_CLIP = args.smoke
SESSION_BUDGET_MIN = 700.0          # hard ceiling of one Kaggle GPU session

# pre-registered tail set (recall < 0.10 in rec2 AND >= 5 train vessels)
TAIL6 = ["Fishing", "Passenger", "Pleasure Craft", "Sailing",
         "Search and Rescue vessel", "Tanker"]

# references measured earlier (rec2 pooled 5 seeds / W3-H frozen-AST 5 seeds)
REF = {
    "cnn_small":  dict(acc=0.3183, macro_f1=0.108, tail_top2=0.114),
    "frozen_AST": dict(acc=0.2677, macro_f1=0.112, tail_top2=0.116),
}
REF_PERCLASS_TOP2 = {                       # frozen AST (W3-H), per class
    "Cargo": 0.758, "Towing": 0.699, "Tug": 0.723, "Fishing": 0.296,
    "Passenger": 0.156, "Pleasure Craft": 0.208, "Port Tender": 0.039,
    "Sailing": 0.037, "Search and Rescue vessel": 0.002, "Tanker": 0.000,
    "Anti-pollution equipment": 0.000, "Dredging": 0.000,
}

# -------------------------------------------------------- discovery --------- #
feat = args.features_root or None
if feat is None:
    for cand in sorted(glob.glob("/kaggle/input/**/uwatr-features", recursive=True)):
        if os.path.isfile(os.path.join(cand, "oceanship", "index.csv")):
            feat = cand
            break
if feat is None:
    raise SystemExit("uwatr-features dataset not found - attach it and rerun")
tick(f"features: {feat}")

with open(os.path.join(feat, "oceanship", "index.csv"), encoding="utf-8",
          newline="") as fh:
    idx_rows = list(csv.DictReader(fh))
label_of_clip = {r["clip_id"]: r["label"] for r in idx_rows}

split_path = None
for name in ("oceanship_splits.csv.gz", "oceanship_splits.csv"):
    p = os.path.join(feat, "splits", name)
    if os.path.exists(p):
        split_path = p
        break
if split_path is None:
    raise SystemExit("no oceanship splits in the feature dataset")
opener = gzip.open if split_path.endswith(".gz") else open
splits = {}
with opener(split_path, "rt", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["protocol"] == "vessel_disjoint":
            splits.setdefault(int(r["seed"]), {})[r["clip_id"]] = r["split"]
tick(f"splits: {split_path}  seeds={sorted(splits)}")

arr = np.load(os.path.join(feat, "oceanship", "logmel.npy"), mmap_mode="r")
row_of = {r["clip_id"]: i for i, r in enumerate(idx_rows)}
N, FREQ, FRAMES = arr.shape
tick(f"array ({N}, {FREQ}, {FRAMES}) {arr.dtype}")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tick(f"device: {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

# --------------------------------------------- model + positional surgery --- #
from transformers import ASTModel

model = ASTModel.from_pretrained(args.model)
EMB_DIM = model.config.hidden_size


def patch_grid(m, freq, frames):
    """Empirical patch grid: run the patch conv on zeros and read the shape."""
    conv = m.embeddings.patch_embeddings.projection      # transformers 5.x
    with torch.no_grad():
        out = conv(torch.zeros(1, 1, freq, frames, device=dev))
    return out.shape[2], out.shape[3]


def adapt_pos_enc(m):
    """Interpolate the patch-position block of AST's positional table.

    Introspected facts (transformers 5.x): conv kernel 16x16 stride 10x10;
    trained grid (12, 101) from max_length 1024; position table (1, 1214) =
    1 cls + 12*101 patches + 1 distillation token.  Only the patch block is
    interpolated; cls and distill keep their trained vectors.
    """
    pe = m.embeddings.position_embeddings.data           # (1, 1+P, D)
    n_pos, D = pe.shape[1], pe.shape[-1]
    max_len = int(getattr(m.config, "max_length", 1024))
    oh, ow = patch_grid(m, FREQ, max_len)
    gh, gw = patch_grid(m, FREQ, FRAMES)
    if oh * ow != n_pos - 2:
        raise RuntimeError(f"pos table {n_pos} != 2 + grid {oh}x{ow}; "
                           f"conv convention unknown")
    special_head, special_tail = pe[:, :1], pe[:, -1:]
    grid = pe[:, 1:-1].reshape(1, oh, ow, D).permute(0, 3, 1, 2)
    new = torch.nn.functional.interpolate(grid, size=(gh, gw),
                                          mode="bilinear", align_corners=False)
    new = new.permute(0, 2, 3, 1).reshape(1, gh * gw, D)
    m.embeddings.position_embeddings = nn.Parameter(
        torch.cat([special_head, new, special_tail], dim=1))
    tick(f"positional table interpolated: patch grid {oh}x{ow} -> {gh}x{gw}")


model.to(dev)
# ASTModel.forward expects input_values of shape (batch, freq, frames) - the
# model adds the channel dimension itself (patch_embeddings.conv is 4D-only)
try:
    with torch.no_grad():
        model(torch.zeros(2, FREQ, FRAMES, device=dev))
    tick("native forward OK (unexpected for 313 frames)")
except Exception as exc:
    print(f"  [fallback] direct forward failed ({str(exc)[:90]}); adapting")
    adapt_pos_enc(model)
with torch.no_grad():
    model(torch.zeros(2, FREQ, FRAMES, device=dev))
tick("forward verified at native 313 frames")

head = nn.Sequential(nn.Dropout(0.1), nn.Linear(EMB_DIM, 12)).to(dev)

if FREEZE_LAYERS > 0:
    enc_layers = getattr(model.encoder, "layer",
                         getattr(model.encoder, "layers", None))
    if enc_layers is None:
        raise SystemExit("cannot locate encoder layers to freeze")
    for i, layer in enumerate(enc_layers):
        for p_ in layer.parameters():
            p_.requires_grad = i >= FREEZE_LAYERS
    tick(f"first {FREEZE_LAYERS} encoder layers frozen")

# snapshot the surgically-adapted PRETRAINED weights: every seed must start
# from the same pretrained state, not from the previous seed's fine-tune
BASE_STATE = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()}
tick("pretrained snapshot taken (per-seed reset)")


def standardize(x):
    x = x.astype(np.float32)
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)


def batches(ids, bs, shuffle, rng):
    order = np.arange(len(ids))
    if shuffle:
        rng.shuffle(order)
    for s in range(0, len(order), bs):
        chunk = [ids[i] for i in order[s:s + bs]]
        rows = np.array([row_of[c] for c in chunk])
        x = torch.from_numpy(standardize(np.array(arr[rows])))
        yield chunk, x.to(dev)


# --------------------------------------------------------------- data ------- #
def load_draw(seed):
    sp = splits[seed]
    ids = [c for c in sp if c in row_of and c in label_of_clip]
    classes = sorted({label_of_clip[c] for c in ids})
    tr = [c for c in ids if sp[c] == "train"]
    va = [c for c in ids if sp[c] == "val"]
    te = [c for c in ids if sp[c] == "test"]
    if SMOKE_N_CLIP:
        tr, va, te = tr[:SMOKE_N_CLIP], va[:SMOKE_N_CLIP // 2], te[:SMOKE_N_CLIP // 2]
    return classes, tr, va, te


def evaluate(split_ids, classes):
    model.eval()
    c2i = {c: i for i, c in enumerate(classes)}
    ys, ps = [], []
    with torch.no_grad():
        for _, x in batches(split_ids, BATCH, False, None):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                h = model(x).last_hidden_state.mean(dim=1)
                logits = head(h)
            ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.extend([c2i[label_of_clip[c]] for c in _])
    P = np.concatenate(ps)
    y = np.array(ys)
    order = np.argsort(-P, axis=1)
    yp = order[:, 0]
    per = {}
    for c in classes:
        i = c2i[c]
        m = y == i
        if not m.any():
            per[c] = dict(recall=0.0, top2=0.0, top3=0.0, n=0)
            continue
        per[c] = dict(recall=float((yp[m] == i).mean()),
                      top2=float((order[m][:, :2] == i).any(1).mean()),
                      top3=float((order[m][:, :3] == i).any(1).mean()),
                      n=int(m.sum()))
    acc = float((yp == y).mean())
    macro = float(np.mean([per[c]["recall"] for c in classes]))
    return acc, macro, per, P, y, split_ids


# --------------------------------------------------------------- train ------ #
summary = {"kind": "w3j-ft-ast", "seeds": {}, "config": dict(
    epochs=EPOCHS, batch=BATCH, lr_backbone=LR_BB, lr_head=LR_HEAD,
    freeze_layers=FREEZE_LAYERS, smoke=SMOKE_N_CLIP)}
tail_top2s, accs, macros = [], [], []

for seed in SEEDS:
    elapsed = (time.time() - T0) / 60
    remaining = SESSION_BUDGET_MIN - elapsed
    if remaining < EPOCHS * 35:          # 35 min/epoch is the pessimistic rate
        print(f"  [stop] seed {seed} skipped: {remaining:.0f} min left, "
              f"needs ~{EPOCHS * 35:.0f}.  Rerun with --seeds {seed} to resume.")
        continue

    classes, tr, va, te = load_draw(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    tick(f"seed {seed}: train {len(tr):,}  val {len(va):,}  test {len(te):,}  "
         f"classes {len(classes)}")

    # reset to the pretrained snapshot; re-init the head
    model.load_state_dict(BASE_STATE)
    for m_ in head.modules():
        if isinstance(m_, nn.Linear):
            nn.init.xavier_uniform_(m_.weight)
            nn.init.zeros_(m_.bias)

    optim = torch.optim.AdamW([
        {"params": [p_ for p_ in model.parameters() if p_.requires_grad],
         "lr": LR_BB},
        {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[LR_BB, LR_HEAD], total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    t_ep0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot_loss = seen = correct = 0
        rng = np.random.RandomState(1000 + seed)
        for chunk, x in batches(tr, BATCH, True, rng):
            optim.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                h = model(x).last_hidden_state.mean(dim=1)
                logits = head(h)
                yb = torch.tensor([c2i[label_of_clip[c]] for c in chunk],
                                  device=dev)
                loss = lossf(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad] +
                list(head.parameters()), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(chunk)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(chunk)
        va_acc, va_macro, _, _, _, _ = evaluate(va, classes)
        lr_now = optim.param_groups[0]["lr"]
        tick(f"seed {seed} ep {ep}/{EPOCHS}  loss {tot_loss / max(seen, 1):.4f}  "
             f"train_acc {correct / max(seen, 1):.4f}  val_acc {va_acc:.4f}  "
             f"lr {lr_now:.2e}")
        if va_acc > best_val:
            best_val = va_acc
            best_state = ({k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()},
                          {k: v.detach().cpu().clone()
                           for k, v in head.state_dict().items()})
        if ep == 1:
            per_epoch = time.time() - t_ep0
            proj = (EPOCHS - ep) * per_epoch / 60 + 8
            tick(f"    pace: {per_epoch / 60:.1f} min/epoch; "
                 f"projected finish for this seed in ~{proj:.0f} min")
    if best_state is None:
        raise SystemExit("training produced no checkpoint")

    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    acc, macro, per, P, y, te_ids = evaluate(te, classes)
    tail = [per[c]["top2"] for c in TAIL6
            if c in per and per[c]["n"] >= 20]
    tail_mean = statistics.mean(tail) if tail else 0.0
    accs.append(acc); macros.append(macro); tail_top2s.append(tail_mean)
    tick(f"seed {seed} TEST: acc {acc:.4f}  macro {macro:.4f}  "
         f"tail top-2 {tail_mean:.3f}")
    summary["seeds"][str(seed)] = dict(acc=acc, macro_f1=macro,
                                       tail_top2=tail_mean, per_class=per,
                                       best_val=best_val)
    np.savez_compressed(
        os.path.join(os.getcwd(), f"w3j_test_pred_s{seed}.npz"),
        clip_id=np.array(te_ids), y_true=y, proba=P.astype(np.float16))

print()
print("=" * 88)
print(f"fine-tuned AST, {len(accs)} seeds - per-class (mean over seeds where available)")
print("=" * 88)
classes_all = sorted(next(iter(summary["seeds"].values()))["per_class"])
print(f"  {'class':<28}{'recall':>8}{'top2':>8}{'frozen-AST top2':>17}")
print("  " + "-" * 64)
pooled = {}
for c in classes_all:
    rs = [summary["seeds"][s]["per_class"][c] for s in summary["seeds"]]
    rec = statistics.mean([r["recall"] for r in rs])
    t2 = statistics.mean([r["top2"] for r in rs])
    pooled[c] = dict(recall=rec, top2=t2)
    mark = "  <- tail6" if c in TAIL6 else ""
    print(f"  {c:<28}{rec:>8.3f}{t2:>8.3f}{REF_PERCLASS_TOP2[c]:>17.3f}{mark}")

print()
print("=" * 88)
print("VERDICT (pre-registered: GO >= 0.30, SEALED <= 0.15 tail mean top-2)")
print("=" * 88)
tail_final = statistics.mean([pooled[c]["top2"] for c in TAIL6
                              if c in pooled])
tail_classes_present = [c for c in TAIL6 if c in pooled]
if tail_classes_present:
    tail_final = statistics.mean([pooled[c]["top2"] for c in tail_classes_present])
else:
    tail_final = None
    print("  [note] no pre-registered tail class present in this draw - "
          "verdict thresholds cannot be evaluated (fixture run?)")
print(f"  cnn_small (rec2)   : acc {REF['cnn_small']['acc']:.4f}  "
      f"macro {REF['cnn_small']['macro_f1']:.3f}  tail top-2 "
      f"{REF['cnn_small']['tail_top2']:.3f}")
print(f"  frozen AST (W3-H)  : acc {REF['frozen_AST']['acc']:.4f}  "
      f"macro {REF['frozen_AST']['macro_f1']:.3f}  tail top-2 "
      f"{REF['frozen_AST']['tail_top2']:.3f}")
if accs:
    print(f"  fine-tuned AST     : acc {statistics.mean(accs):.4f}"
          f"+-{(statistics.stdev(accs) if len(accs) > 1 else 0):.4f}  "
          f"macro {statistics.mean(macros):.3f}  tail top-2 "
          f"{tail_final if tail_final is not None else float('nan'):.3f}")
summary["verdict"] = dict(accs=accs, macros=macros, tail_top2_mean=tail_final,
                          tail_per_class={c: pooled[c] for c in TAIL6
                                          if c in pooled})

if accs and tail_final is not None and tail_final >= 0.30:
    print("  -> GO: full fine-tuning DOES encode the tail that frozen")
    print("     embeddings and the small CNN both miss.  The method door")
    print("     reopens: fine-tuned foundation models + honest evaluation.")
elif accs and tail_final is not None and tail_final <= 0.15:
    print("  -> SEALED: neither capacity (ResNet18), nor pretraining (frozen")
    print("     AST), nor FULL FINE-TUNING encodes the tail.  The collapse is")
    print("     corpus-intrinsic at this operating point - the audit paper's")
    print("     claim now stands in its strongest form.")
else:
    print("  -> PARTIAL: fine-tuning moved some tail classes; report per class.")

for cand in (os.getcwd(),):
    with open(os.path.join(cand, "w3j_ft_ast.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
print()
tick("summary -> w3j_ft_ast.json  (+ w3j_test_pred_s<seed>.npz per seed)")
print("  -> Output tab -> Create Dataset to keep the predictions and summary")
