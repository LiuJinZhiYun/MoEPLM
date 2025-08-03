import json
import numpy as np
import torch
from utils import genData, get_sequence_vectors
from model import My_Model
from metrics import caculate_metric
from dataset import MyDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq2vec = json.load(open(r"D:\Pycharm\pythonProjectDDD\MOE_NewData\esmc600m_all_data_zky.emb"))
batch_size = 128
test_data, test_label, test_seq = genData(r"autodl-tmp/solubility/test_dataset.fasta", 600)
test_dataset = MyDataSet(test_data, test_label, test_seq)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model_path = r"./output/m454.pth"
net = My_Model()
state_dict = torch.load(model_path)
net.load_state_dict(state_dict).to(device)

net.eval()
with torch.no_grad():
    prelabel, relabel = [], []
    pred_prob = []
    net.eval()
    with torch.no_grad():
        for x, y, z in test_iter:
            x, y = x.to(device), y.to(device)
            vec = get_sequence_vectors(z, seq2vec, device)
            logits, _ = net.forward(x, vec)
            prelabel.append(logits.argmax(dim=1).cpu().numpy())
            relabel.append(y.cpu().numpy())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred_prob.extend(probs)
    pred_prob.extend(probs)
    prelabel = np.concatenate(prelabel)
    relabel = np.concatenate(relabel)
    metric, roc_data, prc_data = caculate_metric(prelabel, relabel, pred_prob)

    print(f"TestAccuracy: {metric[0].item():.4f}")
    print(f"Sensitivity: {metric[2].item():.4f}")
    print(f"Specificity: {metric[3].item():.4f}")
    print(f"F1-score: {metric[4].item():.4f}")
    print(f"ROCAUC: {metric[5].item():.4f}")
    print(f"MCC: {metric[6].item():.4f}")

   


