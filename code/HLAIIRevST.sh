#!/bin/bash
set -euo pipefail

# =========================
# HLAIIRevST LoRA Fine-tuning
# Base model: MHCI HLAI-peptide-TCR immunogenicity model
# Target model: MHCII HLAII-peptide-TCR immunogenicity model
# =========================

echo "Start HLAIIRevST LoRA fine-tuning"

# ===== environment =====
# Activate the intended Python environment before running this script.
PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ===== experiment =====
EXP=3
DEVICE="${DEVICE:-cuda}"
SEED=20220909

EPOCHS=200
BATCH_SIZE=256
LR="1e-4"
WEIGHT_DECAY="0.0"

# ===== dataset =====
DATA_DIR="../data/"
DATA_NAME="hlaii"
HLA_ORIENTATION="alpha_reverse"
OUTPUT_DIR="./checkpoints/hlaii_lora/"

# ===== pretrained base =====
PRETRAINED_CKPT="./checkpoints/base_checkpoint/HLAIIRevST_base_checkpoint.pkl"

# ===== sequence length =====
# HLAII alpha=15, beta=19, fused length=34
HLA_ALPHA_MAX_LEN=15
HLA_BETA_MAX_LEN=19
PEP_MAX_LEN=25
TCR_MAX_LEN=30

# ===== model =====
D_MODEL=64
E_LAYERS=3
D_LAYERS=1
DIM=64
N_HEADS=8
D_FF=512
INTERACT_LAYERS=1
SIGMA=1.0
WINDOW_SIZE="3"
DROPOUT=0.3
HLA_FUSION="concat+attention"

# ===== loss =====
ALPHA=0.3
GAMMA=3.0
THRESHOLD=0.5
OVERSAMPLE="hybrid"

# ===== LoRA =====
USE_LORA=true
ADAPTATION_MODE="lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_GROUP="full"
LORA_TARGETS="W_Q,W_K,W_V,W_Q_v,W_K_v,W_V_v,fc,linear1,linear2,out_proj,feed_forward.0,feed_forward.3,ffn.0,ffn.3,projection.0,projection.3,projection.5"
LORA_TRAIN_PROJECTION=true
LORA_TRAIN_BIAS="none"

NUM_WORKERS=4

# ===== optional =====
ABLATE_INPUT="none"
HLA_CORE_ABLATION="full"
CORE_SHUFFLE_SEED=20220909

CMD_ARGS=""
CMD_ARGS="${CMD_ARGS} --exp ${EXP}"
CMD_ARGS="${CMD_ARGS} --device ${DEVICE}"
CMD_ARGS="${CMD_ARGS} --seed ${SEED}"
CMD_ARGS="${CMD_ARGS} --epochs ${EPOCHS}"
CMD_ARGS="${CMD_ARGS} --batch_size ${BATCH_SIZE}"
CMD_ARGS="${CMD_ARGS} --lr ${LR}"
CMD_ARGS="${CMD_ARGS} --weight_decay ${WEIGHT_DECAY}"

CMD_ARGS="${CMD_ARGS} --data_dir ${DATA_DIR}"
CMD_ARGS="${CMD_ARGS} --data_name ${DATA_NAME}"
CMD_ARGS="${CMD_ARGS} --hla_orientation ${HLA_ORIENTATION}"
CMD_ARGS="${CMD_ARGS} --output_dir ${OUTPUT_DIR}"

CMD_ARGS="${CMD_ARGS} --pretrained_ckpt ${PRETRAINED_CKPT}"

CMD_ARGS="${CMD_ARGS} --pep_max_len ${PEP_MAX_LEN}"
CMD_ARGS="${CMD_ARGS} --hla_alpha_max_len ${HLA_ALPHA_MAX_LEN}"
CMD_ARGS="${CMD_ARGS} --hla_beta_max_len ${HLA_BETA_MAX_LEN}"
CMD_ARGS="${CMD_ARGS} --tcr_max_len ${TCR_MAX_LEN}"

CMD_ARGS="${CMD_ARGS} --d_model ${D_MODEL}"
CMD_ARGS="${CMD_ARGS} --e_layers ${E_LAYERS}"
CMD_ARGS="${CMD_ARGS} --d_layers ${D_LAYERS}"
CMD_ARGS="${CMD_ARGS} --dim ${DIM}"
CMD_ARGS="${CMD_ARGS} --n_heads ${N_HEADS}"
CMD_ARGS="${CMD_ARGS} --d_ff ${D_FF}"
CMD_ARGS="${CMD_ARGS} --interact_layers ${INTERACT_LAYERS}"
CMD_ARGS="${CMD_ARGS} --sigma ${SIGMA}"
CMD_ARGS="${CMD_ARGS} --window_size ${WINDOW_SIZE}"
CMD_ARGS="${CMD_ARGS} --dropout ${DROPOUT}"
CMD_ARGS="${CMD_ARGS} --HLA_fusion ${HLA_FUSION}"
CMD_ARGS="${CMD_ARGS} --num_workers ${NUM_WORKERS}"

CMD_ARGS="${CMD_ARGS} --alpha ${ALPHA}"
CMD_ARGS="${CMD_ARGS} --gamma ${GAMMA}"
CMD_ARGS="${CMD_ARGS} --threshold ${THRESHOLD}"
CMD_ARGS="${CMD_ARGS} --oversample ${OVERSAMPLE}"

CMD_ARGS="${CMD_ARGS} --adaptation_mode ${ADAPTATION_MODE}"
CMD_ARGS="${CMD_ARGS} --lora_r ${LORA_R}"
CMD_ARGS="${CMD_ARGS} --lora_alpha ${LORA_ALPHA}"
CMD_ARGS="${CMD_ARGS} --lora_dropout ${LORA_DROPOUT}"
CMD_ARGS="${CMD_ARGS} --lora_target_group ${LORA_TARGET_GROUP}"
CMD_ARGS="${CMD_ARGS} --lora_targets ${LORA_TARGETS}"
CMD_ARGS="${CMD_ARGS} --lora_train_bias ${LORA_TRAIN_BIAS}"

if [ "${USE_LORA}" = true ]; then
  CMD_ARGS="${CMD_ARGS} --use_lora"
fi

if [ "${LORA_TRAIN_PROJECTION}" = true ]; then
  CMD_ARGS="${CMD_ARGS} --lora_train_projection"
fi

CMD_ARGS="${CMD_ARGS} --ablate_input ${ABLATE_INPUT}"
CMD_ARGS="${CMD_ARGS} --hla_core_ablation ${HLA_CORE_ABLATION}"
CMD_ARGS="${CMD_ARGS} --core_shuffle_seed ${CORE_SHUFFLE_SEED}"

# ===== preflight =====
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--check" ) ]]; then
  echo "Usage: bash HLAIIRevST.sh [--check]" >&2
  exit 2
fi
command -v "${PYTHON}" >/dev/null || { echo "Python interpreter not found: ${PYTHON}" >&2; exit 1; }
for FILE in main.py "${PRETRAINED_CKPT}" '../data/hlaii/train_data(-).csv' '../data/hlaii/val_data(-).csv' '../data/hlaii/test_data(-).csv'; do
  [[ -f "${FILE}" ]] || { echo "Required file not found: ${FILE}" >&2; exit 1; }
done

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
"${PYTHON}" -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("WARNING: CUDA is unavailable; main.py will fall back to CPU, which differs from the reference runtime.") if not torch.cuda.is_available() else None'

if [[ "${1:-}" == "--check" ]]; then
  echo "Command: ${PYTHON} -u main.py ${CMD_ARGS}"
  echo "File and runtime checks passed. No training was started."
  exit 0
fi
if [[ -e check_out3.txt || -e check_err3.log ]]; then
  echo "Output logs already exist. Archive or rename them before starting another run." >&2
  exit 1
fi

# ============================================
# Run fine-tuning
# ============================================

echo "Command: ${PYTHON} -u main.py ${CMD_ARGS}"
echo "Redirect output to check_out3.txt and check_err3.log"

# CMD_ARGS uses whitespace-separated values, as in the reference launcher.
nohup "${PYTHON}" -u main.py $CMD_ARGS > check_out3.txt 2> check_err3.log < /dev/null &

pid=$!
echo "Process ID: $pid"
echo "Script execution completed"
