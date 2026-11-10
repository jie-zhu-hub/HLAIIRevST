#!/usr/bin/env bash
# HLAIIRevST inference for the Natalizumab x McPAS-TCR shared-decoy ranking set.
#
# Keep this script in the project's code directory.
# Run from that directory with:
#   bash predict_Natalizumab_McPAS_enhanced_validation.sh
#
# Default model: epoch121 Test-1-PPVn-selected exploratory checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PEPTIDE_COL="${PEPTIDE_COL:-MT_pep}"
THRESHOLD="${THRESHOLD:-}"

INPUT="${INPUT:-$SCRIPT_DIR/../data/hlaii/Natalizumab_McPAS_enhanced_validation_input.csv}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/hlaii_lora/seed_20220909/3_lora_full_r8_frac-100_none_core-full_alpha_reverse_best_test1_ppvn_epoch121_PPVN0p550000.pkl}"
OUTPUT_CSV="${OUTPUT_CSV:-$SCRIPT_DIR/Natalizumab_McPAS_enhanced_validation_predictions.csv}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/Natalizumab_McPAS_enhanced_validation_predict.log}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "$OUTPUT_CSV" != /* ]]; then
  OUTPUT_CSV="$SCRIPT_DIR/$OUTPUT_CSV"
fi
if [[ "$LOG_FILE" != /* ]]; then
  LOG_FILE="$SCRIPT_DIR/$LOG_FILE"
fi

OUTPUT_DIR="$(dirname -- "$OUTPUT_CSV")"
OUTPUT_NAME="$(basename -- "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$LOG_FILE")"

[[ -f "$SCRIPT_DIR/predict_hlaiirevst.py" ]] || die "predict_hlaiirevst.py not found: $SCRIPT_DIR/predict_hlaiirevst.py"
[[ -f "$INPUT" ]] || die "Input CSV not found: $INPUT"
[[ -f "$CHECKPOINT" ]] || die "Checkpoint not found: $CHECKPOINT"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

export INPUT OUTPUT_CSV

printf '%s\n' \
  '================================================================================' \
  'HLAIIRevST Natalizumab shared-decoy ranking preflight' \
  "code directory : $SCRIPT_DIR" \
  "input          : $INPUT" \
  "checkpoint     : $CHECKPOINT" \
  "output CSV     : $OUTPUT_CSV" \
  "log            : $LOG_FILE" \
  "device         : $DEVICE" \
  "batch size     : $BATCH_SIZE" \
  "num workers    : $NUM_WORKERS" \
  "threshold      : ${THRESHOLD:-checkpoint default}" \
  '================================================================================'

"$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import numpy as np
    import pandas as pd
    import torch
except Exception as exc:
    raise SystemExit(
        "HLAIIRevST dependencies are unavailable. Activate the HLAIIRevST environment "
        "containing numpy, pandas and torch before running this script.\n"
        f"Original import error: {exc}"
    )

path = os.environ["INPUT"]
df = pd.read_csv(path)

required = [
    "CDR3",
    "HLA_type",
    "MT_pep",
    "HLA_alpha",
    "HLA_beta",
    "label",
    "peplen",
    "candidate_id",
    "pmhc_id",
    "candidate_type",
    "cdr3_len",
]
missing = [column for column in required if column not in df.columns]
if missing:
    raise SystemExit(f"Input CSV is missing required columns: {missing}")
if df[required].isna().any().any():
    raise SystemExit("Input CSV contains missing values in required columns")
if len(df) != 166:
    raise SystemExit(f"Expected 166 rows, found {len(df)}")

valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
for column in ["CDR3", "MT_pep", "HLA_alpha", "HLA_beta"]:
    invalid = [
        sequence
        for sequence in df[column].astype(str)
        if not sequence or not set(sequence).issubset(valid_aa)
    ]
    if invalid:
        raise SystemExit(
            f"Column {column} contains empty, lowercase, or non-standard sequences: "
            f"{invalid[:10]}"
        )

if not df["label"].astype(int).isin([0, 1]).all():
    raise SystemExit("label must contain only 0 and 1")
if not (df["peplen"].astype(int) == df["MT_pep"].str.len()).all():
    raise SystemExit("peplen does not equal the actual peptide length")
if not (df["cdr3_len"].astype(int) == df["CDR3"].str.len()).all():
    raise SystemExit("cdr3_len does not equal the actual CDR3 length")
if not df["HLA_alpha"].str.len().eq(15).all():
    raise SystemExit("HLA_alpha length must be 15")
if not df["HLA_beta"].str.len().eq(19).all():
    raise SystemExit("HLA_beta length must be 19")
if not df["MT_pep"].str.len().le(25).all():
    raise SystemExit("Peptide length exceeds HLAIIRevST maximum of 25")
if not df["CDR3"].str.len().le(30).all():
    raise SystemExit("CDR3 length exceeds HLAIIRevST maximum of 30")
if not df["HLA_type"].astype(str).eq("HLA-DRB1*07:01").all():
    raise SystemExit("All rows must use HLA-DRB1*07:01")
if not df["HLA_alpha"].astype(str).eq("MIADVAAGSAIFFEQ").all():
    raise SystemExit("The reversed HLA-alpha pseudosequence is incorrect")
if not df["HLA_beta"].astype(str).eq("WGYFELYVIDRQTVHVGFT").all():
    raise SystemExit("The HLA-beta pseudosequence is incorrect")
if df.duplicated(subset=["pmhc_id", "candidate_id"]).any():
    raise SystemExit("Duplicate (pmhc_id, candidate_id) rows detected")

expected = {
    "pMHC_A": {
        "peptide": "GKAPRLLIHYTSALQPGI",
        "rows": 84,
        "positives": 3,
        "decoys": 81,
    },
    "pMHC_B": {
        "peptide": "PRLLIHYTSALQPGI",
        "rows": 82,
        "positives": 1,
        "decoys": 81,
    },
}
if set(df["pmhc_id"]) != set(expected):
    raise SystemExit(f"Unexpected pMHC IDs: {sorted(df['pmhc_id'].unique())}")

decoy_sets = {}
for pmhc_id, specification in expected.items():
    group = df[df["pmhc_id"] == pmhc_id]
    positive = group[group["label"].astype(int) == 1]
    decoy = group[group["label"].astype(int) == 0]
    if len(group) != specification["rows"]:
        raise SystemExit(f"{pmhc_id}: expected {specification['rows']} rows, found {len(group)}")
    if len(positive) != specification["positives"]:
        raise SystemExit(
            f"{pmhc_id}: expected {specification['positives']} positives, found {len(positive)}"
        )
    if len(decoy) != specification["decoys"]:
        raise SystemExit(f"{pmhc_id}: expected 81 decoys, found {len(decoy)}")
    if set(group["MT_pep"]) != {specification["peptide"]}:
        raise SystemExit(f"{pmhc_id}: peptide assignment is incorrect")
    if not positive["candidate_type"].eq("known_positive").all():
        raise SystemExit(f"{pmhc_id}: positive candidate_type is incorrect")
    if not decoy["candidate_type"].eq("putative_non_cognate_decoy").all():
        raise SystemExit(f"{pmhc_id}: decoy candidate_type is incorrect")
    decoy_sets[pmhc_id] = set(zip(decoy["candidate_id"], decoy["CDR3"]))

if decoy_sets["pMHC_A"] != decoy_sets["pMHC_B"]:
    raise SystemExit("The two pMHC groups do not share the same decoy pool")

decoys = df[df["label"].astype(int) == 0]
unique_decoys = decoys[["candidate_id", "CDR3", "cdr3_len"]].drop_duplicates()
length_counts = unique_decoys["cdr3_len"].astype(int).value_counts().to_dict()
if len(unique_decoys) != 81 or length_counts != {14: 51, 15: 30}:
    raise SystemExit(
        f"Shared decoy composition is incorrect: n={len(unique_decoys)}, "
        f"lengths={length_counts}"
    )

print(f"Python : {sys.version.split()[0]}")
print(f"NumPy  : {np.__version__}")
print(f"pandas : {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Input QC passed:")
print("  166 inference rows")
print("  4 known positives")
print("  81 unique shared decoys (51 length-14; 30 length-15)")
print("  pMHC_A: 3 positives + 81 decoys")
print("  pMHC_B: 1 positive  + 81 decoys")
PY

COMMAND=(
  "$PYTHON_BIN" -u "$SCRIPT_DIR/predict_hlaiirevst.py"
  --input "$INPUT"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
  --device "$DEVICE"
  --peptide_col "$PEPTIDE_COL"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$THRESHOLD" ]]; then
  COMMAND+=(--threshold "$THRESHOLD")
fi

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"

[[ -s "$OUTPUT_CSV" ]] || die "Prediction finished without creating a non-empty output: $OUTPUT_CSV"

"$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
import pandas as pd

path = os.environ["OUTPUT_CSV"]
df = pd.read_csv(path)

required = [
    "candidate_id",
    "pmhc_id",
    "candidate_type",
    "CDR3",
    "cdr3_len",
    "label",
    "predicted_label",
    "predicted_score",
    "decision_threshold",
]
missing = [column for column in required if column not in df.columns]
if missing:
    raise SystemExit(f"Prediction output is missing columns: {missing}")
if len(df) != 166:
    raise SystemExit(f"Expected 166 prediction rows, found {len(df)}")
if not df["predicted_score"].between(0.0, 1.0).all():
    raise SystemExit("One or more predicted_score values fall outside [0, 1]")
if not df["predicted_label"].isin([0, 1]).all():
    raise SystemExit("predicted_label contains values other than 0 or 1")

positive = df[df["label"].astype(int) == 1].copy()
decoy = df[df["label"].astype(int) == 0].copy()
ranking_rows = []

for _, row in positive.iterrows():
    background = decoy[decoy["pmhc_id"] == row["pmhc_id"]]
    higher_or_equal = int((background["predicted_score"] >= row["predicted_score"]).sum())
    rank = 1 + higher_or_equal
    percentile = 100.0 * (1.0 - higher_or_equal / len(background))
    empirical_p = rank / (len(background) + 1.0)

    matched = background[
        background["cdr3_len"].astype(int) == int(row["cdr3_len"])
    ]
    matched_higher_or_equal = int(
        (matched["predicted_score"] >= row["predicted_score"]).sum()
    )
    matched_rank = 1 + matched_higher_or_equal
    matched_percentile = 100.0 * (
        1.0 - matched_higher_or_equal / len(matched)
    )
    matched_empirical_p = matched_rank / (len(matched) + 1.0)

    ranking_rows.append(
        {
            "candidate_id": row["candidate_id"],
            "pmhc_id": row["pmhc_id"],
            "CDR3": row["CDR3"],
            "score": row["predicted_score"],
            "all_decoy_rank": f"{rank}/{len(background) + 1}",
            "all_decoy_percentile": percentile,
            "all_decoy_empirical_p": empirical_p,
            "length_matched_rank": f"{matched_rank}/{len(matched) + 1}",
            "length_matched_percentile": matched_percentile,
            "length_matched_empirical_p": matched_empirical_p,
        }
    )

ranking = pd.DataFrame(ranking_rows).sort_values(["pmhc_id", "candidate_id"])

print("\nKnown-positive ranking against shared McPAS decoys:")
print(ranking.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

print("\nScore distribution by pMHC and candidate type:")
summary = (
    df.groupby(["pmhc_id", "candidate_type"])["predicted_score"]
    .agg(["count", "mean", "median", "min", "max"])
)
print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

print("\nImportant interpretation:")
print("  Rankings use continuous predicted_score and do not depend on threshold.")
print("  label=0 rows are putative non-cognate decoys, not confirmed experimental negatives.")
PY

printf '%s\n' \
  '================================================================================' \
  'Natalizumab shared-decoy prediction completed successfully.' \
  "Result: $OUTPUT_CSV" \
  "Log   : $LOG_FILE" \
  '================================================================================'
