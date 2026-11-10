import torch
from torch.utils.data import Dataset

class FourSequenceDataset(Dataset):
    """Dataset for four sequences: peptide 1, peptide 2, HLA, and TCR."""
    
    def __init__(self, data_df, vocab, pep_max_len=11, hla_max_len=34, tcr_max_len=30):
        """
        Args:
            data_df: DataFrame containing the following columns:
                - MT_pep1: Peptide 1 sequence
                - MT_pep2: Peptide 2 sequence  
                - HLA_sequence: HLA sequence
                - CDR3: TCR sequence
                - Label: Label
            vocab: Dictionary mapping amino acids to token indices
            pep_max_len: Maximum length of each peptide
            hla_max_len: Maximum HLA sequence length
            tcr_max_len: Maximum TCR sequence length
        """
        self.data = data_df
        self.vocab = vocab
        self.pep_max_len = pep_max_len
        self.hla_max_len = hla_max_len
        self.tcr_max_len = tcr_max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Pad sequences
        pep1 = row['MT_pep1'].ljust(self.pep_max_len, '-')
        pep2 = row['MT_pep2'].ljust(self.pep_max_len, '-')
        hla = row['HLA_sequence'].ljust(self.hla_max_len, '-')
        tcr = row['CDR3'].ljust(self.tcr_max_len, '-')
        
        # Convert residues to token indices
        pep1_idx = [self.vocab.get(aa, 0) for aa in pep1]
        pep2_idx = [self.vocab.get(aa, 0) for aa in pep2]
        hla_idx = [self.vocab.get(aa, 0) for aa in hla]
        tcr_idx = [self.vocab.get(aa, 0) for aa in tcr]
        
        label = int(row['Label'])
        
        return (
            torch.LongTensor(pep1_idx),
            torch.LongTensor(pep2_idx),
            torch.LongTensor(hla_idx),
            torch.LongTensor(tcr_idx),
            torch.LongTensor([label])
        )
    
    @staticmethod
    def collate_fn(batch):
        """Custom batch collation function."""
        pep1_batch, pep2_batch, hla_batch, tcr_batch, label_batch = zip(*batch)
        
        return (
            torch.stack(pep1_batch),
            torch.stack(pep2_batch),
            torch.stack(hla_batch),
            torch.stack(tcr_batch),
            torch.stack(label_batch).reshape(-1)
        )
