import re
import json
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
AMINO_ACIDS = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
               'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
               'U': 20, 'X': 21, 'B': 22, 'J': 23, 'Z': 24}

valid_amino_acids = ''.join(AMINO_ACIDS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def extract_seq(file):
    pep_seq = []
    for record in SeqIO.parse(file, 'fasta'):
        seq = str(record.seq)
        seq_clean = re.sub(f'[^{valid_amino_acids}]', '', seq.upper())
        pep_seq.append(seq_clean)
    return pep_seq

if __name__ == '__main__':
    train_file = r"../identification/data/Train_dataset.fasta"
    test_file = r"../identification/data/test_dataset.fasta"
    val_file= r"../identification/data/validation_dataset.fasta"
    train_seq = extract_seq(train_file)
    test_seq = extract_seq(test_file)
    val_seq = extract_seq(val_file)
    seq = train_seq + test_seq + val_seq

    seq2vec = dict()
    client = ESMC.from_pretrained("esmc_600m", device)
    i = 0

    for pep in seq:
        i+=1
        print(f'{i}/{len(seq)}')
        pep_str = "".join(pep)
        protein = ESMProtein(sequence=pep_str)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        print(logits_output.embeddings.shape)
        residue_embedding = logits_output.embeddings.squeeze(0)
        protein_embedding = torch.mean(residue_embedding, dim=0)

        out_ten = protein_embedding.cpu().numpy().tolist()
        seq2vec[pep] = out_ten

    with open('../identification/data/esmc600m_all_data.emb', 'w') as g:
        g.write(json.dumps(seq2vec))
