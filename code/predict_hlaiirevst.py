from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from Model.HLAIIRevST import Transformer
from Model.lora_utils import apply_lora_to_model


VOCAB = {
    'C': 1, 'W': 2, 'V': 3, 'A': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10,
    'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0,
}
VOCAB_SIZE = len(VOCAB)
_VOCAB_TABLE = np.zeros(128, dtype=np.int64)
for _ch, _id in VOCAB.items():
    _VOCAB_TABLE[ord(_ch)] = _id


class FourInputDataset(Data.Dataset):
    def __init__(self, hla_alpha: torch.Tensor, hla_beta: torch.Tensor, pep: torch.Tensor, tcr: torch.Tensor, labels: torch.Tensor):
        self.hla_alpha = hla_alpha
        self.hla_beta = hla_beta
        self.pep = pep
        self.tcr = tcr
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.hla_alpha[idx], self.hla_beta[idx], self.pep[idx], self.tcr[idx], self.labels[idx]


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    return new_state


def parse_keyword_string(keyword_string: Optional[str]) -> List[str]:
    if keyword_string is None or keyword_string == '':
        return []
    return [item.strip() for item in keyword_string.split(',') if item.strip()]


def _seq_col_to_ids(series: pd.Series, max_len: int, pad_char: str = '-') -> np.ndarray:
    pad_id = VOCAB[pad_char]
    n = len(series)
    result = np.full((n, max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(series.astype(str).fillna('')):
        chars = seq[:max_len]
        try:
            ids = _VOCAB_TABLE[[ord(c) for c in chars]]
        except Exception as exc:
            raise ValueError(f'Invalid sequence found: {seq!r}. Only amino acids in vocab are allowed.') from exc
        result[i, :len(ids)] = ids
    return result


def make_data(data: pd.DataFrame, peptide_col: str, pep_max_len: int, hla_alpha_max_len: int, hla_beta_max_len: int, tcr_max_len: int):
    required = ['HLA_alpha', 'HLA_beta', peptide_col, 'CDR3']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}. Input CSV must contain {required}.')

    if 'label' not in data.columns:
        data = data.copy()
        data['label'] = 0

    hla_alpha_arr = _seq_col_to_ids(data['HLA_alpha'], hla_alpha_max_len)
    hla_beta_arr = _seq_col_to_ids(data['HLA_beta'], hla_beta_max_len)
    pep_arr = _seq_col_to_ids(data[peptide_col], pep_max_len)
    tcr_arr = _seq_col_to_ids(data['CDR3'], tcr_max_len)
    labels_arr = data['label'].to_numpy(dtype=np.int64)

    return (
        torch.from_numpy(hla_alpha_arr),
        torch.from_numpy(hla_beta_arr),
        torch.from_numpy(pep_arr),
        torch.from_numpy(tcr_arr),
        torch.from_numpy(labels_arr),
    )


def read_predict_data(input_path: str, batch_size: int, peptide_col: str, pep_max_len: int, hla_alpha_max_len: int, hla_beta_max_len: int, tcr_max_len: int, num_workers: int = 0):
    data = pd.read_csv(input_path)
    hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels = make_data(
        data, peptide_col, pep_max_len, hla_alpha_max_len, hla_beta_max_len, tcr_max_len
    )
    use_cuda = torch.cuda.is_available()
    loader = Data.DataLoader(
        FourInputDataset(hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    return data, loader


def transfer(y_prob: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_prob) > threshold).astype(np.int64)


def predict_step(model: torch.nn.Module, loader: Data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_prob_list: List[float] = []
    y_true_list: List[int] = []
    with torch.no_grad():
        for hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels in loader:
            hla_alpha_inputs = hla_alpha_inputs.to(device)
            hla_beta_inputs = hla_beta_inputs.to(device)
            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            y_prob_list.extend(probs.tolist())
            y_true_list.extend(labels.detach().cpu().numpy().tolist())
    return np.array(y_true_list), np.array(y_prob_list)


def checkpoint_args_to_namespace(ckpt_args: Dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    merged = dict(ckpt_args or {})
    # Provide safe defaults when older checkpoints do not store some fields.
    defaults = {
        'pep_max_len': 25,
        'hla_alpha_max_len': 15,
        'hla_beta_max_len': 19,
        'tcr_max_len': 30,
        'd_model': 64,
        'e_layers': 3,
        'dim': 64,
        'n_heads': 8,
        'sigma': 1.0,
        'window_size': '3',
        'd_ff': 512,
        'interact_layers': 1,
        'd_layers': 1,
        'dropout': 0.3,
        'HLA_fusion': 'concat+attention',
        'debug': False,
    }
    for k, v in defaults.items():
        merged.setdefault(k, v)

    # CLI overrides when explicitly provided.
    for name in [
        'pep_max_len', 'hla_alpha_max_len', 'hla_beta_max_len', 'tcr_max_len',
        'd_model', 'e_layers', 'dim', 'n_heads', 'sigma', 'window_size',
        'd_ff', 'interact_layers', 'd_layers', 'dropout', 'HLA_fusion'
    ]:
        cli_val = getattr(cli_args, name)
        if cli_val is not None:
            merged[name] = cli_val

    merged['debug'] = bool(merged.get('debug', False))
    return SimpleNamespace(**merged)


def build_model(args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    if args.HLA_fusion == 'avg':
        if args.hla_alpha_max_len != args.hla_beta_max_len:
            raise ValueError(
                f"When HLA_fusion='avg', hla_alpha_max_len ({args.hla_alpha_max_len}) must equal "
                f"hla_beta_max_len ({args.hla_beta_max_len})."
            )
        fused_hla_len = args.hla_alpha_max_len
    elif args.HLA_fusion in ['concat', 'concat+conv', 'concat+attention']:
        if args.hla_alpha_max_len == args.hla_beta_max_len:
            fused_hla_len = args.hla_alpha_max_len
        else:
            fused_hla_len = args.hla_alpha_max_len + args.hla_beta_max_len
    else:
        raise ValueError(f'Unknown fusion method: {args.HLA_fusion}')

    if fused_hla_len != 34:
        raise ValueError(
            f'For the current DeepSlidingAttention-based ternary model, fused HLA length must be 34, got {fused_hla_len}. '
            'For MHCII use hla_alpha_max_len=15 and hla_beta_max_len=19.'
        )

    window_size = str(args.window_size)
    if window_size == 'default':
        window_size = '3'

    model = Transformer(
        device=device,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        e_layers=args.e_layers,
        d=args.dim,
        n_heads=args.n_heads,
        sigma=args.sigma,
        window_threshold=window_size,
        d_ff=args.d_ff,
        interact_layers=args.interact_layers,
        hla_alpha_max_len=args.hla_alpha_max_len,
        hla_beta_max_len=args.hla_beta_max_len,
        d_layers=args.d_layers,
        dropout=args.dropout,
        HLA_fusion=args.HLA_fusion,
        pep_max_len=args.pep_max_len,
        debug=getattr(args, 'debug', False),
    ).to(device)
    return model


def maybe_apply_lora(model: torch.nn.Module, checkpoint: Dict, cli_args: argparse.Namespace) -> None:
    use_lora = checkpoint.get('use_lora', getattr(cli_args, 'use_lora', False))
    if not use_lora:
        return

    lora_config = checkpoint.get('lora_config', {}) or {}
    r = cli_args.lora_r if cli_args.lora_r is not None else lora_config.get('r', 8)
    alpha = cli_args.lora_alpha if cli_args.lora_alpha is not None else lora_config.get('alpha', 16)
    dropout = cli_args.lora_dropout if cli_args.lora_dropout is not None else lora_config.get('dropout', 0.05)

    if cli_args.lora_targets is not None:
        targets = parse_keyword_string(cli_args.lora_targets)
    else:
        raw_targets = lora_config.get('targets', [])
        if isinstance(raw_targets, str):
            targets = parse_keyword_string(raw_targets)
        else:
            targets = list(raw_targets)

    apply_lora_to_model(
        model,
        target_keywords=targets,
        r=r,
        alpha=alpha,
        dropout=dropout,
        verbose=True,
    )


def load_checkpoint_weights(model: torch.nn.Module, checkpoint: Dict, device: torch.device) -> None:
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    state_dict = strip_module_prefix(state_dict)
    load_msg = model.load_state_dict(state_dict, strict=True)
    if hasattr(load_msg, 'missing_keys') and (load_msg.missing_keys or load_msg.unexpected_keys):
        print('[Warning] strict=True loaded with messages:')
        print('  missing_keys   :', load_msg.missing_keys)
        print('  unexpected_keys:', load_msg.unexpected_keys)
    print('=' * 80)
    print('Loaded inference checkpoint.')
    print('=' * 80)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='predict_hlaiirevst', description='Inference script for HLIAIIRevST fine-tuned checkpoints.')
    parser.add_argument('--input', required=True, type=str, help='Path to input CSV.')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to saved best checkpoint (.pkl/.pt).')
    parser.add_argument('--output', required=True, type=str, help='Directory for prediction outputs.')
    parser.add_argument('--output_name', default='hlaiirevst_predictions.csv', type=str)
    parser.add_argument('--device', default='auto', type=str, help='auto / cpu / cuda / cuda:0 ...')
    parser.add_argument('--threshold', default=None, type=float, help='Override checkpoint threshold. Default: use checkpoint threshold if available.')
    parser.add_argument('--peptide_col', default='MT_pep', type=str, help='Peptide column name. Usually MT_pep; can be WT_pep if needed.')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    # Optional manual overrides for architecture in case checkpoint does not contain args.
    parser.add_argument('--pep_max_len', default=None, type=int)
    parser.add_argument('--hla_alpha_max_len', default=None, type=int)
    parser.add_argument('--hla_beta_max_len', default=None, type=int)
    parser.add_argument('--tcr_max_len', default=None, type=int)
    parser.add_argument('--d_model', default=None, type=int)
    parser.add_argument('--e_layers', default=None, type=int)
    parser.add_argument('--dim', default=None, type=int)
    parser.add_argument('--n_heads', default=None, type=int)
    parser.add_argument('--sigma', default=None, type=float)
    parser.add_argument('--window_size', default=None, type=str)
    parser.add_argument('--d_ff', default=None, type=int)
    parser.add_argument('--interact_layers', default=None, type=int)
    parser.add_argument('--d_layers', default=None, type=int)
    parser.add_argument('--dropout', default=None, type=float)
    parser.add_argument('--HLA_fusion', default=None, type=str)

    # Optional LoRA overrides.
    parser.add_argument('--use_lora', action='store_true', help='Force-enable LoRA injection if checkpoint metadata is missing.')
    parser.add_argument('--lora_r', default=None, type=int)
    parser.add_argument('--lora_alpha', default=None, type=int)
    parser.add_argument('--lora_dropout', default=None, type=float)
    parser.add_argument('--lora_targets', default=None, type=str)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_dir(args.output)
    device = resolve_device(args.device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    model_args = checkpoint_args_to_namespace(ckpt_args, args)

    batch_size = args.batch_size if args.batch_size is not None else int(getattr(model_args, 'batch_size', 256))

    print('=' * 80)
    print('HLIAIIRevST inference configuration')
    print(f'checkpoint      : {args.checkpoint}')
    print(f'device          : {device}')
    print(f'peptide_col     : {args.peptide_col}')
    print(f'batch_size      : {batch_size}')
    print(f'use_lora_ckpt   : {checkpoint.get("use_lora", False)}')
    print(f'ckpt_threshold  : {checkpoint.get("threshold", 0.5)}')
    print('=' * 80)

    predict_data, predict_loader = read_predict_data(
        input_path=args.input,
        batch_size=batch_size,
        peptide_col=args.peptide_col,
        pep_max_len=model_args.pep_max_len,
        hla_alpha_max_len=model_args.hla_alpha_max_len,
        hla_beta_max_len=model_args.hla_beta_max_len,
        tcr_max_len=model_args.tcr_max_len,
        num_workers=args.num_workers,
    )

    model = build_model(model_args, device)
    maybe_apply_lora(model, checkpoint, args)
    load_checkpoint_weights(model, checkpoint, device)
    model.eval()

    _, y_prob = predict_step(model, predict_loader, device)
    threshold = checkpoint.get('threshold', 0.5) if args.threshold is None else args.threshold
    y_pred = transfer(y_prob, threshold)

    result_df = predict_data.copy()
    result_df['predicted_label'] = y_pred
    result_df['predicted_score'] = y_prob
    result_df['decision_threshold'] = threshold

    output_path = os.path.join(args.output, args.output_name)
    result_df.to_csv(output_path, index=False)

    print(f'Prediction completed. Saved to: {output_path}')
    print(result_df[['predicted_label', 'predicted_score']].head())


if __name__ == '__main__':
    main()
