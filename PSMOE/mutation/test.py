import json
import torch
from utils import process_xlsx, map_labels, get_sequence_vectors
from model import My_Model
from metrics import calculate_metric
from dataset import MyDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w_seq2vec = json.load(open(r"PSMOE/mutation/data/wild_type_60.emb"))
m_seq2vec = json.load(open(r"PSMOE/mutation/data/wild_type_60.emb"))


test_mutation_data, test_label, test_wild_type_data, test_original_mutation_seqs, test_original_wild_type_seqs = process_xlsx(r"autodl-tmp/test.xlsx", 600)
test_dataset = MyDataSet(test_wild_type_data, test_mutation_data, test_label, test_original_wild_type_seqs, test_original_mutation_seqs)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 加载模型
model_path = r"./output/m454.pth"
net = My_Model()
state_dict = torch.load(model_path)
net.load_state_dict(state_dict).to(device)

net.eval()
with torch.no_grad():
    all_outputs = []
    all_labels = []
    for wild_data, mut_data, val_label, val_w_seq, val_m_seq in test_iter:
        wild_data = wild_data.to(device)
        mut_data = mut_data.to(device)
        val_label = map_labels(val_label)
        val_label = val_label.to(device)
        w_vec = get_sequence_vectors(val_w_seq, w_seq2vec, device)
        m_vec = get_sequence_vectors(val_m_seq, m_seq2vec, device)
        outputs,_ = net.forward(wild_data, mut_data, w_vec, m_vec)
        all_outputs.append(outputs)
        all_labels.append(val_label)
    output = torch.cat(all_outputs)
    label = torch.cat(all_labels)
    metrics,test_acc,normalized_acc,gcc = calculate_metric(output, label)
   
    for cls in [0, 1, 2]:
        print(f"class {cls} :")
        print(f"  sensitivity: {metrics[f'{cls}_SE']:.4f}")
        print(f"  Specificity: {metrics[f'{cls}_SP']:.4f}")
        print(f"  PPV: {metrics[f'{cls}_PPV']:.4f}")
        print(f"  NPV: {metrics[f'{cls}_NPV']:.4f}")
    print(f"test acc {test_acc}, GCC {gcc}")

   


