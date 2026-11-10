#!/usr/bin/env bash
# Direct inference for the four-row Natalizumab MHC-II case study.
# Keep this file in the project's code directory.
# Then run:
#   bash predict_Natalizumab_case_study.sh
# Default model: epoch121 Test-1-PPVn-selected exploratory checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PEPTIDE_COL="${PEPTIDE_COL:-MT_pep}"

INPUT="${INPUT:-$SCRIPT_DIR/../data/hlaii/Natalizumab_case_study.csv}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/hlaii_lora/seed_20220909/3_lora_full_r8_frac-100_none_core-full_alpha_reverse_best_test1_ppvn_epoch121_PPVN0p550000.pkl}"
OUTPUT_CSV="${OUTPUT_CSV:-$SCRIPT_DIR/Natalizumab_case_study_predictions.csv}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/Natalizumab_case_study_predict.log}"

if [[ "$OUTPUT_CSV" != /* ]]; then
  OUTPUT_CSV="$SCRIPT_DIR/$OUTPUT_CSV"
fi
if [[ "$LOG_FILE" != /* ]]; then
  LOG_FILE="$SCRIPT_DIR/$LOG_FILE"
fi

OUTPUT_DIR="$(dirname -- "$OUTPUT_CSV")"
OUTPUT_NAME="$(basename -- "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$LOG_FILE")"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "$SCRIPT_DIR/predict_hlaiirevst.py" ]] || die "predict_hlaiirevst.py not found in $SCRIPT_DIR"
[[ -f "$INPUT" ]] || die "Input CSV not found: $INPUT"
[[ -f "$CHECKPOINT" ]] || die "Checkpoint not found: $CHECKPOINT"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

export INPUT CHECKPOINT OUTPUT_CSV

printf '%s\n' \
  '===============================================================================' \
  'HLAIIRevST Natalizumab direct-inference preflight' \
  "code directory : $SCRIPT_DIR" \
  "input          : $INPUT" \
  "checkpoint     : $CHECKPOINT" \
  "output CSV     : $OUTPUT_CSV" \
  "log            : $LOG_FILE" \
  "device         : $DEVICE" \
  '============================================================================='

"$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import numpy as np
    import pandas as pd
    import torch
except Exception as exc:
    raise SystemExit(
        "HLAIIRevST dependencies are unavailable. Activate the training environment "
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
]
missing = [column for column in required if column not in df.columns]
if missing:
    raise SystemExit(f"Input CSV is missing required columns: {missing}")
if len(df) != 4:
    raise SystemExit(f"Expected exactly 4 Natalizumab rows, found {len(df)}")
if df[required].isna().any().any():
    raise SystemExit("Input CSV contains missing values in required columns")

valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
for column in ["CDR3", "MT_pep", "HLA_alpha", "HLA_beta"]:
    invalid = [
        sequence
        for sequence in df[column].astype(str)
        if not sequence or not set(sequence).issubset(valid_aa)
    ]
    if invalid:
        raise SystemExit(
            f"Column {column} contains empty, lowercase, or non-standard sequences: {invalid}"
        )

checks = {
    "HLA_alpha length must be 15": df["HLA_alpha"].str.len().eq(15).all(),
    "HLA_beta length must be 19": df["HLA_beta"].str.len().eq(19).all(),
    "peptide length must not exceed 25": df["MT_pep"].str.len().le(25).all(),
    "CDR3 length must not exceed 30": df["CDR3"].str.len().le(30).all(),
    "peplen must equal the actual peptide length": (
        df["peplen"].astype(int) == df["MT_pep"].str.len()
    ).all(),
    "all four case-study labels must be 1": df["label"].astype(int).eq(1).all(),
    "HLA_type must be HLA-DRB1*07:01": (
        df["HLA_type"].astype(str) == "HLA-DRB1*07:01"
    ).all(),
    "The alpha-reverse model requires the reversed alpha pseudosequence": (
        df["HLA_alpha"].astype(str) == "MIADVAAGSAIFFEQ"
    ).all(),
    "HLA beta pseudosequence is incorrect": (
        df["HLA_beta"].astype(str) == "WGYFELYVIDRQTVHVGFT"
    ).all(),
    "triplets must be unique": not df.duplicated(
        subset=["CDR3", "HLA_type", "MT_pep"]
    ).any(),
}
failed = [message for message, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Input QC failed:\n- " + "\n- ".join(failed))

print(f"Python : {sys.version.split()[0]}")
print(f"NumPy  : {np.__version__}")
print(f"pandas : {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Input QC passed: 4 unique positive Natalizumab triplets.")
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

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"

[[ -s "$OUTPUT_CSV" ]] || die "Prediction finished without creating a non-empty output: $OUTPUT_CSV"

"$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

path = os.environ["OUTPUT_CSV"]
df = pd.read_csv(path)

required = ["predicted_label", "predicted_score", "decision_threshold"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise SystemExit(f"Prediction output is missing columns: {missing}")
if len(df) != 4:
    raise SystemExit(f"Expected 4 prediction rows, found {len(df)}")
if not df["predicted_score"].between(0.0, 1.0).all():
    raise SystemExit("One or more predicted_score values fall outside [0, 1]")
if not df["predicted_label"].isin([0, 1]).all():
    raise SystemExit("predicted_label contains values other than 0 or 1")

columns = [
    "CDR3",
    "MT_pep",
    "HLA_type",
    "predicted_score",
    "decision_threshold",
    "predicted_label",
]
print("\nNatalizumab predictions (descending score):")
print(
    df[columns]
    .sort_values("predicted_score", ascending=False)
    .to_string(index=False)
)
print(f"\nKnown-positive recovery: {int(df['predicted_label'].sum())}/4")
print(f"Recall at checkpoint threshold: {df['predicted_label'].mean():.3f}")
print(f"Mean predicted score: {df['predicted_score'].mean():.6f}")
print(f"Median predicted score: {df['predicted_score'].median():.6f}")
print(f"Minimum predicted score: {df['predicted_score'].min():.6f}")
print(f"Maximum predicted score: {df['predicted_score'].max():.6f}")
PY

printf '%s\n' \
  '===============================================================================' \
  'Natalizumab prediction completed successfully.' \
  "Result: $OUTPUT_CSV" \
  "Log   : $LOG_FILE" \
  '============================================================================='
