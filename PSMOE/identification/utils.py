import torch
from Bio import SeqIO
import re
from torch.nn.utils.rnn import pad_sequence

def get_sequence_vectors(z, seq2vec, device):
    vec_list = []
    for i in range(len(z)):

        vec_list.append(torch.tensor(seq2vec[z[i]]).unsqueeze(0).to(device))
    vec = torch.cat(vec_list, dim=0)
    return vec


AMINO_ACIDS = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
               'U': 21, 'X': 22, 'B': 23, 'J': 24, 'Z': 25}

valid_amino_acids = ''.join(AMINO_ACIDS.keys())

def genData(file, max_len):
    long_pep_counter = 0
    pep_codes = []
    labels = []
    pep_seq = []

    for record in SeqIO.parse(file, 'fasta'):
        label_str = record.description.split('-')[-1]
        label = int(label_str)

        seq = str(record.seq)
        seq_clean = re.sub(f'[^{valid_amino_acids}]', '', seq.upper())
        pep_seq.append(seq)

        if len(seq_clean) <= max_len:
            current_pep = [AMINO_ACIDS.get(aa, AMINO_ACIDS['X']) for aa in seq_clean]
        else:
            long_pep_counter += 1
            seq_clean = seq_clean[:max_len]
            current_pep = [AMINO_ACIDS.get(aa, AMINO_ACIDS['X']) for aa in seq_clean]

        pep_codes.append(torch.tensor(current_pep))
        labels.append(label)

    print(f"length > {max_len}: {long_pep_counter}")
    data = pad_sequence(pep_codes, batch_first=True)
    print(f"Data length: {len(data)}, Label length: {len(labels)}")
    return data, torch.tensor(labels), pep_seq
