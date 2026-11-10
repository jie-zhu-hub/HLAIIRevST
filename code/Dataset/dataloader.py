"""HLAIIRevST tensors in model order: HLA-alpha, HLA-beta, peptide, TCR, label."""

import pandas as pd
import torch
from torch.utils.data import TensorDataset


class MyDataSet(TensorDataset):
    """A four-input dataset; each sample has four token vectors and a scalar label."""

    def __init__(self, hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs, labels):
        inputs = (hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs)
        if any(t.ndim != 2 or t.dtype != torch.long for t in inputs):
            raise ValueError('Sequence tensors must be two-dimensional torch.long tensors.')
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError('Labels must be a one-dimensional torch.long tensor.')
        if any(len(t) != len(labels) for t in inputs):
            raise ValueError('All sequence tensors and labels must have the same sample count.')
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError('Labels must contain only 0 or 1.')
        super().__init__(*inputs, labels)

    @classmethod
    def from_dataframe(cls, data, vocab, peptide_col='MT_pep', pep_max_len=25,
                       hla_alpha_max_len=15, hla_beta_max_len=19, tcr_max_len=30):
        """Encode labelled CSV rows, rejecting missing, invalid or over-length sequences."""
        label_col = 'label' if 'label' in data.columns else 'Label'
        columns = ('HLA_alpha', 'HLA_beta', peptide_col, 'CDR3')
        missing = [c for c in (*columns, label_col) if c not in data.columns]
        if missing:
            raise ValueError(f'Missing required CSV columns: {missing}')
        if data.empty:
            raise ValueError('Evaluation CSV contains no samples.')
        labels = pd.to_numeric(data[label_col], errors='raise')
        if not labels.isin([0, 1]).all():
            raise ValueError('Evaluation labels must contain only 0 or 1, without missing values.')
        if 'label' in data and 'Label' in data:
            if not labels.eq(pd.to_numeric(data['Label'], errors='raise')).all():
                raise ValueError('The label and Label columns disagree.')
        lengths = (hla_alpha_max_len, hla_beta_max_len, pep_max_len, tcr_max_len)
        encoded = []
        # Training encodes the ambiguous HLA residue X as token zero.
        encoding_vocab = dict(vocab, X=vocab['-'])
        for column, max_len in zip(columns, lengths):
            if max_len <= 0:
                raise ValueError(f'Maximum length for {column} must be positive.')
            tensor = torch.full((len(data), max_len), vocab['-'], dtype=torch.long)
            for row, sequence in enumerate(data[column]):
                if not isinstance(sequence, str) or not sequence:
                    raise ValueError(f'{column}, CSV row {row + 2}: missing or empty sequence.')
                invalid = set(sequence) - set(encoding_vocab)
                if invalid:
                    raise ValueError(f'{column}, CSV row {row + 2}: invalid residues {sorted(invalid)}.')
                if len(sequence) > max_len:
                    raise ValueError(f'{column}, CSV row {row + 2}: length {len(sequence)} exceeds {max_len}.')
                tensor[row, :len(sequence)] = torch.tensor([encoding_vocab[aa] for aa in sequence], dtype=torch.long)
            encoded.append(tensor)
        return cls(*encoded, torch.tensor(labels.to_numpy(), dtype=torch.long))
