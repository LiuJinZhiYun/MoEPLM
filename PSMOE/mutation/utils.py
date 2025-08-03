import re
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch


def process_xlsx(file_path, max_len=600):
    AMINO_ACIDS = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                   'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
                   'U': 21, 'X': 22, 'B': 23, 'J': 24, 'Z': 25}
    valid_amino_acids = set(AMINO_ACIDS.keys())
    MUTATION_PATTERN = re.compile(r'([A-Za-z])(\d+)([A-Za-z])')
    CLEAN_SEQ_PATTERN = re.compile(f'[^{re.escape("".join(valid_amino_acids))}]')

    df = pd.read_excel(file_path)
    mutation_codes = []
    labels = []
    wild_type_codes = []
    original_mutation_seqs = []
    original_wild_type_seqs = []
    for index, row in df.iterrows():
        label = row['label']
        labels.append(float(label))
        valid_mutations = []
        for mut in row['mutations'].split(','):
            mut = mut.strip().upper()
            match = MUTATION_PATTERN.fullmatch(mut)
            if match:
                orig_aa, pos_str, new_aa = match.groups()
                if orig_aa in valid_amino_acids and new_aa in valid_amino_acids:
                    position = int(pos_str)
                    valid_mutations.append((orig_aa, position, new_aa))

        wild_type_seq = row['sequence'].upper()
        wild_type_clean = CLEAN_SEQ_PATTERN.sub('', wild_type_seq)
        original_wild_type_seqs.append(wild_type_clean)
        mutant_seq = list(wild_type_clean)

        for orig_aa, pos, new_aa in sorted(valid_mutations, key=lambda x: x[1]):
            idx = pos - 1
            if 0 <= idx < len(mutant_seq) and mutant_seq[idx] == orig_aa:
                mutant_seq[idx] = new_aa

        mutant_seq_str = ''.join(mutant_seq).upper()
        mutant_clean = CLEAN_SEQ_PATTERN.sub('', mutant_seq_str)
        original_mutation_seqs.append(mutant_clean)

        mutant_padded = mutant_seq + ['X'] * max_len
        mutant_padded = mutant_padded[:max_len]
        mutant_code = [AMINO_ACIDS.get(aa, 22) for aa in mutant_padded]
        mutation_codes.append(torch.tensor(mutant_code))

        wild_type_list = list(wild_type_clean)
        wild_type_padded = wild_type_list + ['X'] * max_len
        wild_type_padded = wild_type_padded[:max_len]
        wild_type_code = [AMINO_ACIDS.get(aa, 22) for aa in wild_type_padded]
        wild_type_codes.append(torch.tensor(wild_type_code))

    # 张量填充
    mutation_data = pad_sequence(mutation_codes, batch_first=True, padding_value=0)
    wild_type_data = pad_sequence(wild_type_codes, batch_first=True, padding_value=0)
    return mutation_data, torch.tensor(labels), wild_type_data, original_mutation_seqs, original_wild_type_seqs


def get_sequence_vectors(z, seq2vec, device):
    vec_list = []
    for i in range(len(z)):
        vec_list.append(torch.tensor(seq2vec[z[i]]).unsqueeze(0).to(device))
    vec = torch.cat(vec_list, dim=0)
    return vec


def map_labels(labels):
    label_mapping = {-1: 0, 0: 1, 1: 2}
    return torch.tensor([label_mapping[label.item()] for label in labels])


