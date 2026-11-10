import argparse
import os
import pickle
import random
import tempfile
import textwrap
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from Model.HLAIIRevST import Transformer
from Model.lora_utils import (
    apply_lora_to_model,
    extract_lora_state_dict,
    get_trainable_parameter_groups,
    mark_only_lora_and_selected_trainable,
    print_trainable_parameters,
)
from Dataset.HLIAIIRevST_input_dataloader import FourSequenceDataset

vocab = {
    'C': 1, 'W': 2, 'V': 3, 'A': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10,
    'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0
}
vocab_size = len(vocab)
f_mean = lambda l: sum(l) / len(l)

_VOCAB_TABLE = np.zeros(128, dtype=np.int64)
for _ch, _id in vocab.items():
    _VOCAB_TABLE[ord(_ch)] = _id


class WeightedFocalLoss(nn.Module):
    def __init__(self, device, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-bce_loss)
        f_loss = at * (1 - pt) ** self.gamma * bce_loss
        return f_loss.mean()


# Data augmentation
def random_mutation(seq, n_mutations=1, amino_acids=None):
    if amino_acids is None:
        amino_acids = [k for k in vocab.keys() if k != '-']
    seq_list = list(seq)
    for _ in range(n_mutations):
        if len(seq_list) == 0:
            break
        pos = np.random.randint(0, len(seq_list))
        original = seq_list[pos]
        if original in amino_acids:
            possible = [aa for aa in amino_acids if aa != original]
            if possible:
                seq_list[pos] = np.random.choice(possible)
    return ''.join(seq_list)


def random_deletion(seq, n_deletions=1):
    seq_list = list(seq)
    if len(seq_list) <= n_deletions:
        return seq
    for _ in range(n_deletions):
        pos = np.random.randint(0, len(seq_list))
        del seq_list[pos]
    return ''.join(seq_list)


def random_insertion(seq, n_insertions=1, amino_acids=None):
    if amino_acids is None:
        amino_acids = [k for k in vocab.keys() if k != '-']
    seq_list = list(seq)
    for _ in range(n_insertions):
        pos = np.random.randint(0, len(seq_list) + 1)
        aa = np.random.choice(amino_acids)
        seq_list.insert(pos, aa)
    return ''.join(seq_list)


def random_swap(seq, n_swaps=1):
    seq_list = list(seq)
    if len(seq_list) < 2:
        return seq
    for _ in range(n_swaps):
        pos1, pos2 = np.random.choice(len(seq_list), 2, replace=False)
        seq_list[pos1], seq_list[pos2] = seq_list[pos2], seq_list[pos1]
    return ''.join(seq_list)


def advanced_oversampling(data):
    positive_data = data[data.label == 1].copy()
    negative_data = data[data.label == 0].copy()

    pos_count = len(positive_data)
    neg_count = len(negative_data)
    needed = max(0, neg_count - pos_count)

    amino_acids = [k for k in vocab.keys() if k != '-']
    augmented_samples = []

    if needed > 0:
        repeat_samples = positive_data.sample(n=int(needed * 0.5), replace=True).reset_index(drop=True)
        augmented_samples.append(repeat_samples)

    if needed > 0 and len(positive_data) > 0:
        n_augment = needed - len(repeat_samples) if 'repeat_samples' in locals() else needed
        new_rows = []
        for _ in range(n_augment):
            sample = positive_data.iloc[np.random.randint(0, pos_count)].copy()
            aug_method = np.random.choice(['mutate', 'delete', 'insert', 'swap'])
            if aug_method == 'mutate':
                sample.MT_pep = random_mutation(sample.MT_pep, n_mutations=1, amino_acids=amino_acids)
            elif aug_method == 'delete':
                sample.MT_pep = random_deletion(sample.MT_pep, n_deletions=1)
            elif aug_method == 'insert':
                sample.MT_pep = random_insertion(sample.MT_pep, n_insertions=1, amino_acids=amino_acids)
            elif aug_method == 'swap':
                sample.MT_pep = random_swap(sample.MT_pep, n_swaps=1)
            new_rows.append(sample)
        if new_rows:
            augmented_samples.append(pd.DataFrame(new_rows))

    if augmented_samples:
        all_augmented = pd.concat(augmented_samples, ignore_index=True)
        positive_augmented = pd.concat([positive_data, all_augmented], ignore_index=True)
    else:
        positive_augmented = positive_data

    balanced_data = pd.concat([positive_augmented, negative_data], ignore_index=True)
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    print(f"Hybrid oversampling: original positive samples {pos_count} -> after augmentation {len(positive_augmented)}")
    return balanced_data


def oversample_sequences(data, method='simple'):
    positive_data = data[data.label == 1].copy()
    negative_data = data[data.label == 0].copy()

    pos_count = len(positive_data)
    neg_count = len(negative_data)
    target_count = neg_count

    if method == 'simple':
        if pos_count < target_count:
            repeat_times = target_count // pos_count
            remainder = target_count % pos_count
            oversampled_pos = pd.concat(
                [positive_data] * repeat_times + [positive_data.sample(n=remainder, replace=True)],
                ignore_index=True,
            )
            print(f"Simple oversampling: positive samples {pos_count} -> {len(oversampled_pos)}")
            balanced_data = pd.concat([oversampled_pos, negative_data], ignore_index=True)
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
            return balanced_data
    elif method == 'mutate':
        if pos_count < target_count:
            needed = target_count - pos_count
            amino_acids = [aa for aa in vocab.keys() if aa != '-']
            new_samples = []
            for _ in range(needed):
                sample = positive_data.iloc[np.random.randint(0, pos_count)].copy()
                if len(sample.MT_pep) > 0:
                    pep_list = list(sample.MT_pep)
                    mutate_pos = np.random.randint(0, len(pep_list))
                    original_aa = pep_list[mutate_pos]
                    possible_aas = [aa for aa in amino_acids if aa != original_aa]
                    if possible_aas:
                        pep_list[mutate_pos] = np.random.choice(possible_aas)
                        sample.MT_pep = ''.join(pep_list)
                new_samples.append(sample)
            oversampled_data = pd.concat([positive_data, pd.DataFrame(new_samples)], ignore_index=True)
            balanced_data = pd.concat([oversampled_data, negative_data], ignore_index=True)
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
            print(f"Mutation oversampling: positive samples {pos_count} -> {len(oversampled_data)}")
            return balanced_data
    elif method == 'hybrid':
        return advanced_oversampling(data)

    return data


# Data processing
def _seq_col_to_ids(series, max_len, pad_char='-'):
    pad_id = vocab[pad_char]
    n = len(series)
    result = np.full((n, max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(series):
        chars = seq[:max_len]
        ids = _VOCAB_TABLE[[ord(c) for c in chars]]
        result[i, :len(ids)] = ids
    return result


def make_data(data, pep_max_len, hla_alpha_max_len, hla_beta_max_len, tcr_max_len):
    hla_alpha_arr = _seq_col_to_ids(data.HLA_alpha, hla_alpha_max_len)
    hla_beta_arr = _seq_col_to_ids(data.HLA_beta, hla_beta_max_len)
    pep_arr = _seq_col_to_ids(data.MT_pep, pep_max_len)
    tcr_arr = _seq_col_to_ids(data.CDR3, tcr_max_len)
    labels_arr = data.label.to_numpy(dtype=np.int64, copy=True)
    return (
        torch.from_numpy(hla_alpha_arr),
        torch.from_numpy(hla_beta_arr),
        torch.from_numpy(pep_arr),
        torch.from_numpy(tcr_arr),
        torch.from_numpy(labels_arr),
    )


def apply_input_ablation(hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, ablate_input='none'):
    """
    Mask one input component with padding tokens while preserving tensor shapes.
    This isolates each input source without changing model architecture.
    """
    ablate_input = (ablate_input or 'none').lower()
    pad_id = vocab['-']

    if ablate_input == 'none':
        return hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs
    if ablate_input == 'hla':
        hla_alpha_inputs = torch.full_like(hla_alpha_inputs, pad_id)
        hla_beta_inputs = torch.full_like(hla_beta_inputs, pad_id)
    elif ablate_input == 'peptide':
        pep_inputs = torch.full_like(pep_inputs, pad_id)
    elif ablate_input == 'tcr':
        tcr_inputs = torch.full_like(tcr_inputs, pad_id)
    else:
        raise ValueError(
            f"Unsupported ablate_input={ablate_input!r}. "
            "Choose from ['none', 'hla', 'peptide', 'tcr']."
        )

    return hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs


def _shuffle_each_row(inputs, seed):
    rng = np.random.RandomState(seed)
    shuffled = inputs.clone()
    for row_idx in range(shuffled.size(0)):
        perm = torch.from_numpy(rng.permutation(shuffled.size(1))).long()
        shuffled[row_idx] = shuffled[row_idx, perm]
    return shuffled


def apply_hla_core_ablation(
    hla_alpha_inputs,
    hla_beta_inputs,
    hla_core_ablation='full',
    core_shuffle_seed=20220909,
    split_name='',
):
    """
    HLA-II core ablation for the 15-aa alpha + 19-aa beta pseudosequence.
    The tensor shapes stay unchanged so the fused HLA length remains 34.
    """
    mode = (hla_core_ablation or 'full').lower()
    if mode == 'shuffle':
        mode = 'shuffled'

    pad_id = vocab['-']
    if mode == 'full':
        return hla_alpha_inputs, hla_beta_inputs
    if mode == 'alpha_only':
        return hla_alpha_inputs, torch.full_like(hla_beta_inputs, pad_id)
    if mode == 'beta_only':
        return torch.full_like(hla_alpha_inputs, pad_id), hla_beta_inputs
    if mode == 'shuffled':
        split_offsets = {'train': 0, 'val': 100000, 'test': 200000}
        seed_base = int(core_shuffle_seed) + split_offsets.get(split_name, 300000)
        hla_alpha_inputs = _shuffle_each_row(hla_alpha_inputs, seed_base + 17)
        hla_beta_inputs = _shuffle_each_row(hla_beta_inputs, seed_base + 31)
        return hla_alpha_inputs, hla_beta_inputs

    raise ValueError(
        f"Unsupported hla_core_ablation={hla_core_ablation!r}. "
        "Choose from ['full', 'alpha_only', 'beta_only', 'shuffled']."
    )


HLA_ORIENTATION_TO_SUFFIX = {
    'forward': '(+)',
    'alpha_reverse': '(-)',
    'beta_reverse': '(+,-)',
    'dual_reverse': '(-,-)',
}

LEGACY_ALPHA_DIRECTION_TO_ORIENTATION = {
    'forward': 'forward',
    'reverse': 'alpha_reverse',
}


def normalize_hla_orientation(hla_orientation=None, hla_alpha_direction=None):
    """Resolve the four-condition orientation while preserving the legacy flag."""
    if hla_orientation is not None:
        hla_orientation = hla_orientation.lower()
        if hla_orientation not in HLA_ORIENTATION_TO_SUFFIX:
            raise ValueError(
                f"Unsupported hla_orientation: {hla_orientation}. "
                f"Choose from {sorted(HLA_ORIENTATION_TO_SUFFIX)}."
            )

    if hla_alpha_direction is not None:
        hla_alpha_direction = hla_alpha_direction.lower()
        if hla_alpha_direction not in LEGACY_ALPHA_DIRECTION_TO_ORIENTATION:
            raise ValueError(
                f"Unsupported hla_alpha_direction: {hla_alpha_direction}. "
                "Choose from ['forward', 'reverse']."
            )
        legacy_orientation = LEGACY_ALPHA_DIRECTION_TO_ORIENTATION[hla_alpha_direction]
        if hla_orientation is not None and hla_orientation != legacy_orientation:
            raise ValueError(
                f"Conflicting orientation arguments: --hla_orientation={hla_orientation} "
                f"but --hla_alpha_direction={hla_alpha_direction} maps to {legacy_orientation}."
            )
        hla_orientation = legacy_orientation

    return hla_orientation or 'forward'


def resolve_dataset_filename(type_, hla_alpha_direction=None, hla_orientation=None):
    orientation = normalize_hla_orientation(
        hla_orientation=hla_orientation,
        hla_alpha_direction=hla_alpha_direction,
    )
    suffix = HLA_ORIENTATION_TO_SUFFIX[orientation]
    return f'{type_}_data{suffix}.csv'


def data_with_loader(
    type_='train',
    batch_size=1024,
    pep_max_len=25,
    hla_alpha_max_len=15,
    hla_beta_max_len=19,
    tcr_max_len=30,
    datapath='../data/random',
    oversample_method=None,
    num_workers=4,
    hla_alpha_direction=None,
    hla_orientation=None,
    ablate_input='none',
    hla_core_ablation='full',
    core_shuffle_seed=20220909,
):
    filename = resolve_dataset_filename(
        type_,
        hla_alpha_direction=hla_alpha_direction,
        hla_orientation=hla_orientation,
    )
    file_path = os.path.join(datapath, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Dataset file not found: {file_path}')
    print(f'[Data] {type_:>5} -> {file_path}')
    data = pd.read_csv(file_path, index_col=False)
    if type_ == 'train' and oversample_method:
        data = oversample_sequences(data, method=oversample_method)

    hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels = make_data(
        data, pep_max_len, hla_alpha_max_len, hla_beta_max_len, tcr_max_len
    )
    hla_alpha_inputs, hla_beta_inputs = apply_hla_core_ablation(
        hla_alpha_inputs,
        hla_beta_inputs,
        hla_core_ablation=hla_core_ablation,
        core_shuffle_seed=core_shuffle_seed,
        split_name=type_,
    )
    if hla_core_ablation != 'full':
        print(f'[HLA-core ablation] {type_:>5}: {hla_core_ablation}')

    hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs = apply_input_ablation(
        hla_alpha_inputs,
        hla_beta_inputs,
        pep_inputs,
        tcr_inputs,
        ablate_input=ablate_input,
    )
    if ablate_input != 'none':
        print(f'[Ablation] {type_:>5} input component masked: {ablate_input}')

    use_cuda = torch.cuda.is_available()
    loader = Data.DataLoader(
        Data.TensorDataset(hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels),
        batch_size,
        shuffle=(type_ == 'train'),
        collate_fn=FourSequenceDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )
    return data, hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels, loader


# Evaluation metrics
def calculate_ppvn(y_true, y_scores, n):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    sorted_y_true = [y_true[i] for i in sorted_indices]
    ppvn = sum(sorted_y_true[:n]) / n if n > 0 else 0.0
    return ppvn


def performances(y_true, y_pred, y_prob, print_=True, dataset_name=""):
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        print(f"Warning: y_prob contains NaN/inf in {dataset_name}. Replacing with 0.5.")
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=0.5, neginf=0.5)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    if print_:
        print(f'******* {dataset_name}  *******')
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    n = Counter(y_true)[1]
    ppvn = calculate_ppvn(y_true, y_prob, n)
    if print_:
        print('auc={:.4f}|aupr={:.4f}|ppvn={:.4f}'.format(roc_auc, aupr, ppvn))
    return (roc_auc, aupr, ppvn, tn, fp, fn, tp)


def transfer(y_prob, threshold=0.5):
    return (np.asarray(y_prob) > threshold).astype(np.int64)


# Training / validation
def _run_epoch(model, loader, ae, epoch, epochs, device, threshold, criterion,
               dataset_name, optimizer=None):
    """Share batching and metrics while keeping training-only updates explicit."""
    training = optimizer is not None
    model.train(training)
    losses, y_true, y_prob = [], [], []
    phase = 'Train' if training else 'Eval'
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=f'{ae}|{phase}|Epoch-{epoch}/{epochs}'):
            *inputs, labels = (tensor.to(device) for tensor in batch)
            outputs, _, _ = model(*inputs)
            if training and getattr(model, 'debug', False) and torch.isnan(outputs).any():
                raise ValueError('NaN in model outputs')
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(F.softmax(outputs.detach(), dim=1)[:, 1].cpu().numpy())
            losses.append(loss.item())

    predictions = transfer(y_prob, threshold)
    if training:
        print("^" * 94 + "\n")
        print(f'{ae}****Training: Epoch-{epoch}/{epochs} | Loss = {f_mean(losses):.4f}')
    else:
        print(f'{ae} ****{dataset_name}  Epoch-{epoch}/{epochs}: Loss = {f_mean(losses):.6f}')
    metrics = performances(y_true, predictions, y_prob, True, dataset_name)
    return (y_true, predictions, y_prob), losses, metrics


def train_step(model, train_loader, ae, epoch, epochs, device, threshold, criterion, optimizer):
    start_time = time.time()
    result = _run_epoch(model, train_loader, ae, epoch, epochs, device, threshold,
                        criterion, 'Training', optimizer)
    return (*result, time.time() - start_time)


def eval_step(model, val_loader, ae, epoch, epochs, device, threshold, criterion, dataset_name="Test"):
    return _run_epoch(model, val_loader, ae, epoch, epochs, device, threshold,
                      criterion, dataset_name)


# Utility functions
def set_seed(seed=20220909):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args_box(args):
    """Print all effective options without maintaining a duplicate field list."""
    print('Model Parameter Configuration')
    for key, value in vars(args).items():
        prefix = f'  {key}: '
        print(textwrap.fill(str(value), width=120, initial_indent=prefix,
                            subsequent_indent=' ' * len(prefix), break_on_hyphens=False))


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def safe_remove(path):
    if path and os.path.exists(path):
        os.remove(path)


def format_metric_for_filename(value, is_int=False):
    if is_int:
        return str(int(value))
    return f"{value:.6f}".replace('-', 'neg').replace('.', 'p')


def parse_keyword_string(keyword_string):
    if keyword_string is None or keyword_string == '':
        return []
    return [item.strip() for item in keyword_string.split(',') if item.strip()]


def strip_module_prefix(state_dict):
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in state_dict.items()}


def load_pretrained_weights(model, checkpoint_path, device, ignore_projection=False, strict=False):
    if not checkpoint_path:
        print('[Info] No pretrained_ckpt was provided; training will start from the current initialized parameters.')
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint_path}')

    if hasattr(torch.serialization, 'safe_globals'):
        from numpy.core.multiarray import scalar
        allowed = [scalar, (scalar, 'numpy.core.multiarray.scalar'), np.dtype,
                   type(np.dtype('float64')), type(np.dtype('float32')),
                   type(np.dtype('int64')), type(np.dtype('int32'))]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    state_dict = strip_module_prefix(state_dict)

    if ignore_projection:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('projection.')}

    # Filter keys with mismatched shapes (for example, pos_encoder.pe when pep_max_len differs)
    model_state = model.state_dict()
    mismatched_keys = []
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            mismatched_keys.append(f'{k}: ckpt={tuple(v.shape)} vs model={tuple(model_state[k].shape)}')
        else:
            filtered_state_dict[k] = v
    if mismatched_keys:
        print(f'[Warning] Skipping {len(mismatched_keys)} keys with mismatched shapes (initialized model values will be used):')
        for msg in mismatched_keys:
            print(f'  {msg}')

    load_msg = model.load_state_dict(filtered_state_dict, strict=strict)
    print('=' * 80)
    print(f'Loaded pretrained checkpoint: {checkpoint_path}')
    if hasattr(load_msg, 'missing_keys'):
        print(f'Missing keys   ({len(load_msg.missing_keys)}): {load_msg.missing_keys[:30]}')
        print(f'Unexpected keys({len(load_msg.unexpected_keys)}): {load_msg.unexpected_keys[:30]}')
    print('=' * 80)


def update_best_checkpoint(trackers, key, score, epoch, model, threshold, path_saver, args):
    tracker = trackers[key]
    if score <= tracker['best']:
        return False

    ensure_dir(os.path.dirname(path_saver))
    previous_path = tracker['model_path']
    score_token = format_metric_for_filename(score, is_int=tracker['is_int'])
    model_path = f"{path_saver}_best_{key}_epoch{epoch}_{tracker['file_metric_name']}{score_token}.pkl"

    checkpoint = {
        'epoch': epoch,
        'selection_key': key,
        'selection_name': tracker['display_name'],
        'selection_score': score.item() if isinstance(score, np.generic) else score,
        'threshold': threshold.item() if isinstance(threshold, np.generic) else threshold,
        'model_state_dict': model.state_dict(),
        'use_lora': args.use_lora,
        'lora_config': {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'targets': parse_keyword_string(args.lora_targets),
            'train_projection': args.lora_train_projection,
        },
        'lora_state_dict': extract_lora_state_dict(model, ('projection',) if args.lora_train_projection else tuple()),
        'args': {name: value.item() if isinstance(value, np.generic) else value
                 for name, value in vars(args).items()},
    }
    # Publish a complete checkpoint before removing the previous best model.
    fd, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(model_path) + '.', suffix='.tmp',
        dir=os.path.dirname(model_path) or '.',
    )
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        safe_remove(temporary_path)
    if previous_path and os.path.abspath(previous_path) != os.path.abspath(model_path):
        try:
            safe_remove(previous_path)
        except OSError as exc:
            print(f'[Warning] Saved the new checkpoint but could not remove {previous_path}: {exc}')

    tracker['best'] = score
    tracker['epoch'] = epoch
    tracker['threshold'] = threshold
    tracker['model_path'] = model_path

    print(f"[Best Model Updated] {tracker['display_name']}: {score} @ epoch {epoch}")
    print(f"  saved_model: {model_path}")
    return True


def evaluate_saved_model(
    model,
    checkpoint_path,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    exp,
    epochs,
    criterion,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_eval = model.eval()

    best_epoch = checkpoint.get('epoch', -1)
    eval_threshold = checkpoint.get('threshold', 0.5)
    selection_score = checkpoint.get('selection_score', None)

    print('\n' + '=' * 100)
    print(f'Evaluating the saved best model: {model_name}')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  epoch: {best_epoch}')
    print(f'  selection_score: {selection_score}')
    print(f'  threshold: {eval_threshold}')
    print('=' * 100)

    result = {
        'model_name': model_name,
        'selection_score': selection_score,
        'selection_epoch': best_epoch,
        'threshold': eval_threshold,
    }
    metric_names = ('roc_auc', 'aupr', 'ppvn', 'tn', 'fp', 'fn', 'tp')
    for number, (split, label, loader) in enumerate([
        ('train', 'Training', train_loader),
        ('val', 'Validation', val_loader),
        ('test1', 'Test-1', test_loader),
    ], start=1):
        print(f'\n=== {number}. {label}-set evaluation ===')
        predictions, _, metrics = eval_step(
            model_eval, loader, exp, best_epoch, epochs, device, eval_threshold, criterion,
            dataset_name=f"{model_name} | {label}",
        )
        names = metric_names[:3] if split == 'train' else metric_names
        result.update((f'{split}_{name}', value) for name, value in zip(names, metrics))
        if split != 'test1':
            result[f'{split}_predictions'] = predictions
    return result



def print_full_test1_metrics(metrics_test, epoch=None, title='Test-1 Full evaluation results'):
    prefix = f'Epoch-{epoch} | ' if epoch is not None else ''
    print('\n' + '=' * 100)
    print(f'{prefix}{title}')
    print('=' * 100)
    for name, value in zip(('ROC-AUC', 'AUPR', 'PPVn', 'TN', 'FP', 'FN', 'TP'), metrics_test):
        formatted = f'{value:.6f}' if name in ('ROC-AUC', 'AUPR', 'PPVn') else str(value)
        print(f'{name:<8}: {formatted}')
    print('=' * 100)


def build_model(args, device):
    if args.HLA_fusion == 'avg':
        if args.hla_alpha_max_len != args.hla_beta_max_len:
            raise ValueError(
                f"When HLA_fusion='avg', hla_alpha_max_len ({args.hla_alpha_max_len}) "
                f"must equal hla_beta_max_len ({args.hla_beta_max_len})."
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

    model = Transformer(
        device=device,
        vocab_size=vocab_size,
        d_model=args.d_model,
        e_layers=args.e_layers,
        d=args.dim,
        n_heads=args.n_heads,
        sigma=args.sigma,
        window_threshold=args.window_size,
        d_ff=args.d_ff,
        interact_layers=args.interact_layers,
        hla_alpha_max_len=args.hla_alpha_max_len,
        hla_beta_max_len=args.hla_beta_max_len,
        d_layers=args.d_layers,
        dropout=args.dropout,
        HLA_fusion=args.HLA_fusion,
        pep_max_len=args.pep_max_len,
        debug=args.debug,
    ).to(device)
    return model


def build_arg_parser():
    parser = argparse.ArgumentParser(prog='HLIAIIRevST-MHCII-LoRA', description='Fine-tune MHCI pretrained HLIAIIRevST on MHCII with LoRA')

    parser.add_argument('--pep_max_len', default=25, type=int)
    parser.add_argument('--hla_alpha_max_len', default=15, type=int)
    parser.add_argument('--hla_beta_max_len', default=19, type=int)
    parser.add_argument('--tcr_max_len', default=30, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--gamma', default=3.0, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--output_dir', default='./checkpoints/AE/', type=str)
    parser.add_argument('--d_model', default=64, type=int)
    parser.add_argument('--e_layers', default=3, type=int)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('--window_size', default='default', type=str, choices=['1', '2', '3', '4', 'default'])
    parser.add_argument('--d_ff', default=512, type=int)
    parser.add_argument('--interact_layers', default=1, type=int)
    parser.add_argument('--d_layers', default=1, type=int)
    parser.add_argument('--ae', '--exp', dest='ae', default='1', type=str, help='Ablation Experiment identifier.')
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--data_name', default='hlaii', type=str)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--oversample', default='hybrid', type=str, choices=['simple', 'mutate', 'hybrid'])
    parser.add_argument('--HLA_fusion', default='concat+attention', type=str, choices=['concat', 'concat+attention', 'concat+conv', 'avg'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--hla_orientation', default=None, type=str, choices=sorted(HLA_ORIENTATION_TO_SUFFIX), help='HLA-II alpha/beta-chain pseudosequence orientation: forward -> *_data(+).csv; alpha_reverse -> *_data(-).csv; beta_reverse -> *_data(+,-).csv; dual_reverse -> *_data(-,-).csv.')
    parser.add_argument('--hla_alpha_direction', default=None, type=str, choices=['forward', 'reverse'], help='Legacy compatibility option: forward maps to --hla_orientation forward; reverse maps to --hla_orientation alpha_reverse.')
    parser.add_argument('--ablate_input', default='none', type=str, choices=['none', 'hla', 'peptide', 'tcr'], help='Input component ablation: none=full model; hla=mask HLA_alpha/HLA_beta; peptide=mask MT_pep; tcr=mask CDR3.')
    parser.add_argument('--hla_core_ablation', default='full', type=str, choices=['full', 'alpha_only', 'beta_only', 'shuffled', 'shuffle'], help='HLA-II core ablation: full=alpha15+beta19; alpha_only=mask beta; beta_only=mask alpha; shuffled=shuffle alpha/beta positions within each sample.')
    parser.add_argument('--core_shuffle_seed', default=20220909, type=int, help='Fixed seed for per-sample HLA core order shuffling.')
    parser.add_argument('--seed', default=20220909, type=int)

    # Pretraining / LoRA
    parser.add_argument('--pretrained_ckpt', default='', type=str, help='MHCI pretrained checkpoint path.')
    parser.add_argument('--ignore_pretrained_projection', action='store_true', help='Do not load the old classifier head.')
    parser.add_argument('--strict_pretrained_load', action='store_true', help='Use strict=True when loading pretrained weights.')
    parser.add_argument('--adaptation_mode', default='auto', type=str, choices=['auto', 'lora'], help='Compatibility option: lora is equivalent to the legacy --use_lora flag.')
    parser.add_argument('--use_lora', action='store_true', help='Enable parameter-efficient LoRA fine-tuning.')
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_target_group', default='full', type=str, choices=['none', 'full', 'attention', 'ffn_projection', 'custom'], help='LoRA target family label for logging/checkpoint naming; actual matching uses --lora_targets.')
    parser.add_argument('--lora_targets', default='W_Q,W_K,W_V,W_Q_v,W_K_v,W_V_v,fc,linear1,linear2,out_proj,feed_forward.0,feed_forward.3,ffn.0,ffn.3,projection.0,projection.3,projection.5', type=str, help='Comma-separated module-name keywords used to match nn.Linear modules for LoRA injection.')
    parser.add_argument('--lora_train_projection', action='store_true', help='Keep the classifier head trainable in addition to LoRA weights.')
    parser.add_argument('--lora_train_bias', default='none', choices=['none', 'all', 'lora_only'])

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    args.hla_orientation = normalize_hla_orientation(
        hla_orientation=args.hla_orientation,
        hla_alpha_direction=args.hla_alpha_direction,
    )

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        args.device = 'cpu'
    device = torch.device(args.device)

    if args.window_size == 'default':
        args.window_size = '3'

    if args.adaptation_mode == 'lora':
        args.use_lora = True
    elif args.adaptation_mode == 'auto':
        args.adaptation_mode = 'lora' if args.use_lora else 'full_finetune'

    args.output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    ensure_dir(args.output_dir)
    datapath = os.path.join(args.data_dir, args.data_name)

    args.dataset_suffix = HLA_ORIENTATION_TO_SUFFIX[args.hla_orientation]
    args.train_file = os.path.join(
        datapath,
        resolve_dataset_filename('train', hla_orientation=args.hla_orientation),
    )
    args.val_file = os.path.join(
        datapath,
        resolve_dataset_filename('val', hla_orientation=args.hla_orientation),
    )
    args.test_file = os.path.join(
        datapath,
        resolve_dataset_filename('test', hla_orientation=args.hla_orientation),
    )

    set_seed(args.seed)
    print(datapath)
    print_args_box(args)

    datasets, loaders = {}, {}
    # Preserve loading order and random-number consumption from the trained version.
    for split in ('test', 'train', 'val'):
        loaded = data_with_loader(
            type_=split, batch_size=args.batch_size, pep_max_len=args.pep_max_len,
            hla_alpha_max_len=args.hla_alpha_max_len, hla_beta_max_len=args.hla_beta_max_len,
            tcr_max_len=args.tcr_max_len, datapath=datapath, num_workers=args.num_workers,
            oversample_method=args.oversample if split == 'train' else None,
            hla_orientation=args.hla_orientation, ablate_input=args.ablate_input,
            hla_core_ablation=args.hla_core_ablation, core_shuffle_seed=args.core_shuffle_seed,
        )
        datasets[split], loaders[split] = loaded[0], loaded[-1]
    train_data, val_data, test_data = (datasets[split] for split in ('train', 'val', 'test'))
    train_loader, val_loader, test_loader = (loaders[split] for split in ('train', 'val', 'test'))

    print('{} Label info: Train = {} | Val = {}'.format(args.ae, Counter(train_data.label), Counter(val_data.label)))

    print('-----Compile model-----')
    model = build_model(args, device)
    load_pretrained_weights(
        model,
        checkpoint_path=args.pretrained_ckpt,
        device=device,
        ignore_projection=args.ignore_pretrained_projection,
        strict=args.strict_pretrained_load,
    )

    if args.use_lora:
        lora_targets = parse_keyword_string(args.lora_targets)
        apply_lora_to_model(
            model,
            target_keywords=lora_targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=True,
        )
        extra_trainable = ('projection',) if args.lora_train_projection else tuple()
        mark_only_lora_and_selected_trainable(
            model,
            extra_trainable_keywords=extra_trainable,
            train_bias=args.lora_train_bias,
        )
    else:
        print('[Info] use_lora=False; standard full-parameter fine-tuning will be performed.')

    print_trainable_parameters(model)

    criterion = WeightedFocalLoss(device=device, alpha=args.alpha, gamma=args.gamma)
    optimizer = optim.Adam(get_trainable_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)

    print('-----Train-----')
    rank_tag = f'r{args.lora_r}' if args.adaptation_mode == 'lora' else 'r0'
    path_saver = os.path.join(
        args.output_dir,
        f'{args.ae}_{args.adaptation_mode}_{args.lora_target_group}_{rank_tag}_'
        f'frac-100_{args.ablate_input}_core-{args.hla_core_ablation}_{args.hla_orientation}'
    )
    print('output_dir:', args.output_dir)
    print('path_saver:', path_saver)


    best_trackers = {
        'test1_ppvn': {
            'best': -float('inf'), 'epoch': -1, 'threshold': args.threshold, 'model_path': None,
            'display_name': 'Test-PPVn exploratory', 'file_metric_name': 'PPVN', 'is_int': False,
        },
        'val_aupr': {
            'best': -float('inf'), 'epoch': -1, 'threshold': args.threshold, 'model_path': None,
            'display_name': 'Val-AUPR primary', 'file_metric_name': 'AUPR', 'is_int': False,
        },
    }

    time_train = 0.0
    for epoch in range(1, args.epochs + 1):
        _, _, _, time_train_ep = train_step(
            model, train_loader, args.ae, epoch, args.epochs, device, args.threshold,
            criterion, optimizer
        )
        time_train += time_train_ep
        print()

        split_metrics = {}
        for split, label in (('val', 'Validation'), ('test', 'Test-1')):
            _, _, split_metrics[split] = eval_step(
                model, loaders[split], args.ae, epoch, args.epochs, device, args.threshold,
                criterion, dataset_name=label,
            )
            print()
        metrics_val, metrics_test = split_metrics['val'], split_metrics['test']
        print('\n' + '^' * 94)

        update_best_checkpoint(best_trackers, 'test1_ppvn', metrics_test[2], epoch, model, args.threshold, path_saver, args)
        update_best_checkpoint(best_trackers, 'val_aupr', metrics_val[1], epoch, model, args.threshold, path_saver, args)

    print('\n' + '=' * 100)
    print('Training completed; starting final evaluation of the two saved best models')
    print(f'Final reference threshold: {args.threshold}')
    print('=' * 100)

    evaluation_summaries = []
    last_predictions = None
    for key in ['test1_ppvn', 'val_aupr']:
        tracker = best_trackers[key]
        if tracker['model_path'] is None:
            print(f"[Warning] {tracker['display_name']} was not generated; skipping.")
            continue
        result = evaluate_saved_model(
            model=model,
            checkpoint_path=tracker['model_path'],
            model_name=tracker['display_name'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            exp=args.ae,
            epochs=args.epochs,
            criterion=criterion,
        )
        evaluation_summaries.append({k: v for k, v in result.items() if k not in ['train_predictions', 'val_predictions']})
        print_full_test1_metrics(
            (
                result['test1_roc_auc'], result['test1_aupr'], result['test1_ppvn'],
                result['test1_tn'], result['test1_fp'], result['test1_fn'], result['test1_tp']
            ),
            epoch=result['selection_epoch'],
            title=f"{result['model_name']} | Test-1 Full evaluation results"
        )
        last_predictions = result

    if last_predictions is not None:
        ys_train_fold_dict = {args.ae: last_predictions['train_predictions']}
        ys_val_fold_dict = {args.ae: last_predictions['val_predictions']}
        with open(os.path.join(args.output_dir, 'train.pkl'), 'wb') as f:
            pickle.dump(ys_train_fold_dict, f)
        with open(os.path.join(args.output_dir, 'val.pkl'), 'wb') as f:
            pickle.dump(ys_val_fold_dict, f)

    summary_df = pd.DataFrame(evaluation_summaries)
    print('\n' + '=' * 100)
    print('Final evaluation summary of the two best models (printed only; not saved as CSV):')
    print('=' * 100)
    if summary_df.empty:
        print('No evaluation results are available.')
    else:
        print(summary_df.to_string(index=False))
    print('Total training time: {:6.2f} sec'.format(time_train))


if __name__ == '__main__':
    main()
