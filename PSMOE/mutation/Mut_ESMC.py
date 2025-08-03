import json
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_xlsx(file_path):
    AMINO_ACIDS = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                   'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
                   'U': 21, 'X': 22, 'B': 23, 'J': 24, 'Z': 25}
    valid_amino_acids = set(AMINO_ACIDS.keys())
    MUTATION_PATTERN = re.compile(r'([A-Za-z])(\d+)([A-Za-z])')
    CLEAN_SEQ_PATTERN = re.compile(f'[^{re.escape("".join(valid_amino_acids))}]')

    df = pd.read_excel(file_path)
    labels = []
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

    return original_mutation_seqs, original_wild_type_seqs


if __name__ == '__main__':
    train_file = r"PSMOE/mutation/data/train.xlsx"
    val_file = r'PSMOE/mutation/data/val.xlsx'
    test_file = r'PSMOE/mutation/data/test.xlsx'
    train_mut_seq, train_w_seq = process_xlsx(train_file)
    val_mut_seq, val_w_seq = process_xlsx(val_file)
    test_mut_seq, test_w_seq = process_xlsx(test_file)
    mut_seq = train_mut_seq + val_mut_seq + test_mut_seq
    w_seq = train_w_seq + val_w_seq + test_w_seq
    w_pooled_vec = dict()
    mut_pooled_vec = dict()
    client = ESMC.from_pretrained("esmc_600m", device)
   
    i = 0

    for w_pep, mut_pep in zip(w_seq, mut_seq):
        i += 1
        print(f'processing {i}/{len(w_seq)}')
        w_pep_str = "".join(w_pep)
        mut_pep_str = "".join(mut_pep)
        w_protein = ESMProtein(sequence=w_pep_str)
        mut_protein = ESMProtein(sequence=mut_pep_str)

        w_protein_tensor = client.encode(w_protein)
        mut_protein_tensor = client.encode(mut_protein)

        w_logits_output = client.logits(
            w_protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        mut_logits_output = client.logits(
            mut_protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        w_residue_embedding = w_logits_output.embeddings.squeeze(0)
        mut_residue_embedding = mut_logits_output.embeddings.squeeze(0)

        diff = torch.abs(w_residue_embedding - mut_residue_embedding)
        diff_sum = torch.sum(diff, dim=1)
        _, indices = torch.topk(diff_sum, k=60)


        w_selected = w_residue_embedding[indices]
        mut_selected = mut_residue_embedding[indices]
        w_pooled = torch.mean(w_selected, dim=0).squeeze()
        mut_pooled = torch.mean(mut_selected, dim=0).squeeze()

        w_pooled_vec[w_pep_str] = w_pooled.cpu().numpy().tolist()
        mut_pooled_vec[mut_pep_str] = mut_pooled.cpu().numpy().tolist()

        # 分别保存池化后的特征到两个文件
    with open('wild_type_60.emb', 'w') as f:
        json.dump(w_pooled_vec, f)

    with open('mutation_type_60.emb', 'w') as f:
        json.dump(mut_pooled_vec, f)
