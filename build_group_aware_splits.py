"""
Fast Group-Aware Split Builder
================================
Optimized: numpy vectorized MinHash (64 perm), cross-split only (train vs val+test),
10k batch processing, max 50k per source.

Parametreler: num_perm=64, threshold=0.85, seed=42
Split: 70 / 15 / 15
"""

import sys, re, json, os, time, hashlib, zlib
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT       = Path(__file__).resolve().parent
SPLITS_IN  = ROOT / "data" / "splits" / "master_splits"
SPLITS_OUT = ROOT / "data" / "splits" / "group_aware_splits"
REPORT_OUT = ROOT / "results" / "group_aware_split_report.json"

NUM_PERM       = 64
THRESHOLD      = 0.85
NUM_BANDS      = 16          # 16 bands x 4 rows = 64 perms
ROWS_PER_BAND  = NUM_PERM // NUM_BANDS   # 4
SHINGLE_K      = 3
SEED           = 42
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
MAX_PER_SOURCE = 50_000
BATCH_SIZE     = 10_000

# ── Hash functions (fixed seed, uint64 arithmetic) ───────────────────────────
_rng = np.random.default_rng(seed=SEED)
_A   = _rng.integers(1, 2**31, size=NUM_PERM, dtype=np.uint64)
_B   = _rng.integers(0, 2**31, size=NUM_PERM, dtype=np.uint64)
_MAX = np.iinfo(np.uint64).max

# ── Normalization ─────────────────────────────────────────────────────────────
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_TR_MAP   = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                           "abcçdefgğhıijklmnoöprsştuüvyz")

def normalize(text: str) -> str:
    text = text.translate(_TR_MAP)
    text = _URL_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()

def md5hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def text_to_sig(text: str) -> np.ndarray:
    words  = text.split()
    k      = SHINGLE_K
    if len(words) < k:
        shingles = [text] if text else ["__empty__"]
    else:
        shingles = [" ".join(words[i:i+k]) for i in range(len(words) - k + 1)]

    sig = np.full(NUM_PERM, _MAX, dtype=np.uint64)
    for s in shingles:
        # crc32: 32-bit ve süreçler arası deterministik (hash() PYTHONHASHSEED ile her çalıştırmada değişir)
        h  = np.uint64(zlib.crc32(s.encode("utf-8")))
        hv = _A * h + _B                              # uint64 overflow = mod 2^64
        np.minimum(sig, hv, out=sig)
    return sig

def compute_sigs_batch(texts: list, label: str = "") -> np.ndarray:
    n    = len(texts)
    sigs = np.zeros((n, NUM_PERM), dtype=np.uint64)
    t0   = time.time()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        for i in range(start, end):
            sigs[i] = text_to_sig(texts[i])
        print(f"  {label} {end:,}/{n:,} ({end/n*100:.0f}%)  [{time.time()-t0:.1f}s]",
              flush=True)
    return sigs

# ── Union-Find ────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 — Veri Yükleme + Örnekleme
# ═════════════════════════════════════════════════════════════════════════════
t_total = time.time()
print("=" * 62, flush=True)
print("STEP 0 — Veri yükleniyor ve örnekleniyor...", flush=True)
print("=" * 62, flush=True)
t0 = time.time()

train_df = pd.read_parquet(SPLITS_IN / "train.parquet")
val_df   = pd.read_parquet(SPLITS_IN / "val.parquet")
test_df  = pd.read_parquet(SPLITS_IN / "test.parquet")

train_df["_orig_split"] = "train"
val_df  ["_orig_split"] = "val"
test_df ["_orig_split"] = "test"

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"  Ham toplam: {len(all_df):,} satır  "
      f"(train={len(train_df):,} val={len(val_df):,} test={len(test_df):,})",
      flush=True)

# Max per-source sampling
parts = []
for src, grp in all_df.groupby("source"):
    if len(grp) > MAX_PER_SOURCE:
        grp = grp.sample(n=MAX_PER_SOURCE, random_state=SEED)
        tag = f"→ kırpıldı {MAX_PER_SOURCE:,}"
    else:
        tag = "(tam)"
    parts.append(grp)
    print(f"  {src:<20} {len(grp):,}  {tag}", flush=True)

all_df = pd.concat(parts, ignore_index=True)
N      = len(all_df)
print(f"\n  Örnekleme sonrası toplam : {N:,}  [{time.time()-t0:.1f}s]", flush=True)

# Normalize
print("  Normalizasyon yapılıyor...", flush=True)
norm_texts = all_df["text"].fillna("").map(normalize).tolist()
print(f"  Normalizasyon tamamlandı.  [{time.time()-t0:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Train MinHash
# ═════════════════════════════════════════════════════════════════════════════
train_mask    = all_df["_orig_split"] == "train"
test_val_mask = ~train_mask

train_indices   = np.where(train_mask.values)[0]
testval_indices = np.where(test_val_mask.values)[0]

train_texts   = [norm_texts[i] for i in train_indices]
testval_texts = [norm_texts[i] for i in testval_indices]

print(f"\n{'='*62}", flush=True)
print(f"STEP 1 — Train MinHash ({len(train_texts):,} örnek, {NUM_PERM} perm)...", flush=True)
print(f"{'='*62}", flush=True)
t1 = time.time()
train_sigs = compute_sigs_batch(train_texts, label="Train:")
print(f"  Train sigs bitti.  [{time.time()-t1:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Test/Val MinHash
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print(f"STEP 2 — Val+Test MinHash ({len(testval_texts):,} örnek)...", flush=True)
print(f"{'='*62}", flush=True)
t2 = time.time()
testval_sigs = compute_sigs_batch(testval_texts, label="Val+Test:")
print(f"  Val+Test sigs bitti.  [{time.time()-t2:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-Split Near-Dup Detection (Band LSH)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print(f"STEP 3 — Cross-split LSH  "
      f"[{NUM_BANDS} band x {ROWS_PER_BAND} row, threshold={THRESHOLD}]", flush=True)
print(f"{'='*62}", flush=True)
t3 = time.time()

candidate_pairs: set[tuple[int, int]] = set()

for b in range(NUM_BANDS):
    r0 = b * ROWS_PER_BAND
    r1 = r0 + ROWS_PER_BAND

    # Build index from train band signatures
    band_idx: dict[bytes, list] = defaultdict(list)
    train_band = train_sigs[:, r0:r1]
    for i in range(len(train_sigs)):
        band_idx[train_band[i].tobytes()].append(i)

    # Query with val+test band signatures
    tv_band = testval_sigs[:, r0:r1]
    for j in range(len(testval_sigs)):
        key = tv_band[j].tobytes()
        if key in band_idx:
            for i in band_idx[key]:
                candidate_pairs.add((i, j))

    print(f"  Band {b+1:2d}/{NUM_BANDS}  candidates: {len(candidate_pairs):,}"
          f"  [{time.time()-t3:.1f}s]", flush=True)

print(f"\n  Toplam candidate çifti: {len(candidate_pairs):,}", flush=True)

# Verify similarity — filter false positives
print("  Jaccard doğrulaması yapılıyor...", flush=True)
t_v = time.time()
verified_pairs: list[tuple[int, int]] = []
for ci, (ti, vi) in enumerate(candidate_pairs):
    sim = float(np.mean(train_sigs[ti] == testval_sigs[vi]))
    if sim >= THRESHOLD:
        verified_pairs.append((int(train_indices[ti]), int(testval_indices[vi])))
    if (ci + 1) % 10_000 == 0:
        print(f"  Doğrulama {ci+1:,}/{len(candidate_pairs):,}  "
              f"geçen: {len(verified_pairs):,}  [{time.time()-t_v:.1f}s]", flush=True)

print(f"  Doğrulanmış near-dup çifti: {len(verified_pairs):,}  "
      f"[{time.time()-t3:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Union-Find Grouping
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("STEP 4 — Union-Find gruplama...", flush=True)
print(f"{'='*62}", flush=True)
t4 = time.time()

uf = UnionFind(N)
for gi, gj in verified_pairs:
    uf.union(gi, gj)

all_df["_group"] = [uf.find(i) for i in range(N)]

gc     = all_df["_group"].value_counts()
n_sing = int((gc == 1).sum())
n_ns   = int((gc > 1).sum())
n_in   = int(gc[gc > 1].sum())

print(f"  Toplam grup        : {len(gc):,}", flush=True)
print(f"  Singleton          : {n_sing:,}", flush=True)
print(f"  Near-dup grup (≥2) : {n_ns:,}", flush=True)
print(f"  Grupta olan örnek  : {n_in:,}", flush=True)
print(f"  Union-Find bitti.  [{time.time()-t4:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Group-Aware Stratified Split (70/15/15)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("STEP 5 — Group-aware stratified split (70/15/15)...", flush=True)
print(f"{'='*62}", flush=True)
t5 = time.time()

# Dominant label per group (use label_norm)
group_label = (
    all_df.groupby("_group")["label_norm"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"label_norm": "_glabel"})
)
all_df = all_df.merge(group_label, on="_group", how="left")

ug          = group_label.copy()
lc          = ug["_glabel"].value_counts()
rare_labels = lc[lc < 2].index.tolist()

if rare_labels:
    print(f"  Nadir sınıf (train'e eklenir): {rare_labels}", flush=True)

rare_mask     = ug["_glabel"].isin(rare_labels)
rare_groups   = ug[rare_mask]
common_groups = ug[~rare_mask]

g_ids    = common_groups["_group"].tolist()
g_labels = common_groups["_glabel"].tolist()

train_g, temp_g, _, temp_labels = train_test_split(
    g_ids, g_labels,
    test_size=1.0 - TRAIN_RATIO,
    random_state=SEED,
    stratify=g_labels,
)
val_g, test_g = train_test_split(
    temp_g,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_labels,
)

train_g  = list(train_g) + rare_groups["_group"].tolist()
train_set = set(train_g)
val_set   = set(val_g)
test_set  = set(test_g)

def assign_split(gid: int) -> str:
    if gid in train_set: return "train"
    if gid in val_set:   return "val"
    return "test"

all_df["_new_split"] = all_df["_group"].map(assign_split)
sizes = all_df["_new_split"].value_counts()
print(f"  Yeni train : {sizes.get('train', 0):,}", flush=True)
print(f"  Yeni val   : {sizes.get('val',   0):,}", flush=True)
print(f"  Yeni test  : {sizes.get('test',  0):,}", flush=True)
print(f"  Split bitti.  [{time.time()-t5:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Validation
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("STEP 6 — Doğrulama...", flush=True)
print(f"{'='*62}", flush=True)
t6 = time.time()

new_train = all_df[all_df["_new_split"] == "train"]
new_test  = all_df[all_df["_new_split"] == "test"]

# Group leak (should be 0 by construction)
train_grps = set(new_train["_group"].unique())
test_grps  = set(new_test["_group"].unique())
group_leak = len(train_grps & test_grps)
print(f"  Train-test ortak grup : {group_leak}  (0 = OK)", flush=True)

# Exact duplicate check
train_hashes = set(new_train["text"].fillna("").map(normalize).map(md5hex))
exact_dup    = int(new_test["text"].fillna("").map(normalize).map(md5hex).isin(train_hashes).sum())
print(f"  Exact dup (train-test): {exact_dup}", flush=True)

# Class distribution
print("\n  Sınıf dağılımı (test):", flush=True)
orig_test_dist = test_df["label_norm"].value_counts().to_dict()
new_test_dist  = new_test["label_norm"].value_counts().to_dict()
all_labels     = sorted(orig_test_dist.keys(), key=lambda x: -orig_test_dist.get(x, 0))
print(f"  {'Sınıf':<16} {'Orijinal':>10} {'Yeni':>10}", flush=True)
print("  " + "-" * 38, flush=True)
for lbl in all_labels:
    print(f"  {lbl:<16} {orig_test_dist.get(lbl,0):>10,} {new_test_dist.get(lbl,0):>10,}",
          flush=True)
print(f"  Doğrulama bitti.  [{time.time()-t6:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save Parquets
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("STEP 7 — Parquet dosyaları kaydediliyor...", flush=True)
print(f"{'='*62}", flush=True)
t7 = time.time()

SPLITS_OUT.mkdir(parents=True, exist_ok=True)
SAVE_COLS = ["text", "label_norm", "source", "label_id", "label"]

for split_name in ["train", "val", "test"]:
    out = SPLITS_OUT / f"{split_name}.parquet"
    df  = all_df[all_df["_new_split"] == split_name][SAVE_COLS].reset_index(drop=True)
    df.to_parquet(str(out), index=False)
    print(f"  {split_name}.parquet : {len(df):,} satır  → {out}", flush=True)

print(f"  Kayıt bitti.  [{time.time()-t7:.1f}s]", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save Report
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("STEP 8 — Rapor kaydediliyor...", flush=True)
print(f"{'='*62}", flush=True)

total_elapsed = round(time.time() - t_total, 1)

report = {
    "params": {
        "num_perm": NUM_PERM,
        "threshold": THRESHOLD,
        "num_bands": NUM_BANDS,
        "rows_per_band": ROWS_PER_BAND,
        "max_per_source": MAX_PER_SOURCE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    },
    "dataset": {
        "total_after_sampling": N,
        "orig_train": int(train_mask.sum()),
        "orig_val_test": int(test_val_mask.sum()),
    },
    "deduplication": {
        "candidate_pairs": len(candidate_pairs),
        "verified_pairs": len(verified_pairs),
        "total_groups": int(len(gc)),
        "singleton_groups": n_sing,
        "non_singleton_groups": n_ns,
        "examples_in_groups": n_in,
    },
    "split_result": {
        "train": int(sizes.get("train", 0)),
        "val":   int(sizes.get("val",   0)),
        "test":  int(sizes.get("test",  0)),
        "group_leak": group_leak,
        "exact_dup_train_test": exact_dup,
    },
    "class_dist_test_original": {k: int(v) for k, v in orig_test_dist.items()},
    "class_dist_test_new":      {k: int(v) for k, v in new_test_dist.items()},
    "total_elapsed_s": total_elapsed,
}

os.makedirs(str(REPORT_OUT.parent), exist_ok=True)
with open(str(REPORT_OUT), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"  Kaydedildi: {REPORT_OUT}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# ÖZET
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}", flush=True)
print("ÖZET", flush=True)
print(f"{'='*62}", flush=True)
print(f"  Toplam süre        : {total_elapsed:.1f}s", flush=True)
print(f"  Örnekleme sonrası  : {N:,} satır", flush=True)
print(f"  Cross near-dup çifti: {len(verified_pairs):,}", flush=True)
print(f"  Group leak         : {group_leak}  {'✓ OK' if group_leak == 0 else '✗ HATA'}", flush=True)
print(f"  Exact dup          : {exact_dup}  {'✓ OK' if exact_dup == 0 else '⚠ Var'}", flush=True)
print(f"  Yeni train         : {sizes.get('train',0):,}", flush=True)
print(f"  Yeni val           : {sizes.get('val',0):,}", flush=True)
print(f"  Yeni test          : {sizes.get('test',0):,}", flush=True)
print(f"  Çıktı dizini       : {SPLITS_OUT}", flush=True)
print(f"  Rapor              : {REPORT_OUT}", flush=True)
print(flush=True)
