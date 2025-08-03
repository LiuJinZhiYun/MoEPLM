import torch.utils.data as Data

class MyDataSet(Data.Dataset):
    def __init__(self, wild_codes, mut_codes, labels, wild_seqs, mut_seqs):
        self.wild_codes = wild_codes
        self.mut_codes = mut_codes
        self.labels = labels
        self.wild_seqs = wild_seqs
        self.mut_seqs = mut_seqs


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wild_code = self.wild_codes[idx]
        mut_code = self.mut_codes[idx]
        label = self.labels[idx]
        wild_seq = self.wild_seqs[idx]
        mut_seq = self.mut_seqs[idx]
        return wild_code, mut_code, label, wild_seq, mut_seq