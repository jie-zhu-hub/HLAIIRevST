"""Evaluate a fixed HLAIIRevST checkpoint on a labelled four-sequence CSV."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.core.multiarray import scalar as numpy_scalar
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score

from Dataset.dataloader import MyDataSet
from main import apply_hla_core_ablation, apply_input_ablation, build_model
from predict_hlaiirevst import (
    VOCAB,
    build_arg_parser as prediction_arg_parser,
    checkpoint_args_to_namespace,
    load_checkpoint_weights,
    maybe_apply_lora,
    predict_step,
    resolve_device,
    transfer,
)


def build_arg_parser():
    parser = prediction_arg_parser()
    parser.prog = 'HLAIIRevST-test'
    parser.description = 'Evaluate a saved HLAIIRevST model without training or threshold search.'
    parser.set_defaults(output_name='hlaiirevst_test_predictions.csv')
    return parser


def load_checkpoint(path, device):
    # Current training checkpoints can contain NumPy scalar metrics.
    if 'weights_only' not in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=device)
    if not hasattr(torch.serialization, 'safe_globals'):
        return torch.load(path, map_location=device, weights_only=True)
    allowed = [numpy_scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


def calculate_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError('Expected non-empty, equally sized label and score vectors.')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Evaluation labels must be 0 or 1.')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Predictions contain invalid probabilities; evaluation stopped.')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be a finite number between 0 and 1.')
    predicted = transfer(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    positives = int((labels == 1).sum())
    metrics = {
        'n_samples': int(len(labels)), 'n_positive': positives,
        'threshold': float(threshold), 'decision_rule': 'score > threshold',
        'roc_auc': None, 'aupr': None, 'ppvn': None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float((tp + tn) / len(labels)),
    }
    if len(np.unique(labels)) == 2:
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics['roc_auc'] = float(roc_auc_score(labels, scores))
        metrics['aupr'] = float(auc(recall, precision))
    if positives:
        order = np.argsort(-scores, kind='mergesort')
        metrics['ppvn'] = float(labels[order[:positives]].sum() / positives)
    return metrics


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch_size must be positive.')
    if args.num_workers < 0:
        parser.error('--num_workers must be non-negative.')
    input_path, checkpoint_path = Path(args.input), Path(args.checkpoint)
    for path in (input_path, checkpoint_path):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    output_dir = Path(args.output)
    if Path(args.output_name).name != args.output_name:
        parser.error('--output_name must be a filename within --output.')
    predictions_path = output_dir / args.output_name
    metrics_path = output_dir / 'hlaiirevst_test_metrics.json'
    protected = {input_path.resolve(), checkpoint_path.resolve()}
    if (predictions_path.resolve() in protected or metrics_path.resolve() in protected
            or predictions_path.resolve() == metrics_path.resolve()):
        parser.error('Output paths must not overwrite the input CSV, checkpoint, or each other.')

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a training checkpoint dictionary or a model state_dict.')
    model_args = checkpoint_args_to_namespace(checkpoint.get('args', {}), args)
    # The training attention module selects its spatial prior using string values.
    model_args.window_size = str(model_args.window_size)
    if model_args.window_size == 'default':
        model_args.window_size = '3'
    threshold = float(checkpoint.get('threshold', 0.5) if args.threshold is None else args.threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be a finite number between 0 and 1.')
    batch_size = args.batch_size if args.batch_size is not None else int(getattr(model_args, 'batch_size', 256))
    if batch_size <= 0:
        raise ValueError('Checkpoint batch_size must be positive.')

    data = pd.read_csv(input_path)
    dataset = MyDataSet.from_dataframe(
        data, VOCAB, peptide_col=args.peptide_col,
        pep_max_len=model_args.pep_max_len,
        hla_alpha_max_len=model_args.hla_alpha_max_len,
        hla_beta_max_len=model_args.hla_beta_max_len,
        tcr_max_len=model_args.tcr_max_len,
    )
    alpha, beta, peptide, tcr, labels = dataset.tensors
    core_ablation = getattr(model_args, 'hla_core_ablation', 'full')
    input_ablation = getattr(model_args, 'ablate_input', 'none')
    alpha, beta = apply_hla_core_ablation(
        alpha, beta, hla_core_ablation=core_ablation,
        core_shuffle_seed=getattr(model_args, 'core_shuffle_seed', 20220909), split_name='test',
    )
    alpha, beta, peptide, tcr = apply_input_ablation(alpha, beta, peptide, tcr, input_ablation)
    loader = torch.utils.data.DataLoader(
        MyDataSet(alpha, beta, peptide, tcr, labels), batch_size=batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda',
    )
    model = build_model(model_args, device)
    lora_checkpoint = dict(checkpoint, use_lora=bool(checkpoint.get('use_lora', False) or args.use_lora))
    maybe_apply_lora(model, lora_checkpoint, args)
    load_checkpoint_weights(model, checkpoint, device)
    y_true, scores = predict_step(model, loader, device)
    metrics = calculate_metrics(y_true, scores, threshold)
    metrics.update({
        'model': 'HLAIIRevST', 'input': str(input_path.resolve()),
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_selection_key': checkpoint.get('selection_key'),
        'threshold_source': 'cli' if args.threshold is not None else ('checkpoint' if 'threshold' in checkpoint else 'default_0.5'),
        'checkpoint_mdrt': bool(getattr(model_args, 'mdrt', False)),
        'hla_orientation': getattr(model_args, 'hla_orientation', None),
        'ablate_input': input_ablation, 'hla_core_ablation': core_ablation,
    })
    results = data.copy()
    results['predicted_label'] = transfer(scores, threshold)
    results['predicted_score'] = scores
    results['decision_threshold'] = threshold
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f'Predictions: {predictions_path}\nMetrics: {metrics_path}')


if __name__ == '__main__':
    main()
