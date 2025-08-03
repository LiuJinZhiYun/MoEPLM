import json
import time
from termcolor import colored
import datetime
import numpy as np
import torch
import torch.nn as nn
from utils import process_xlsx, map_labels, get_sequence_vectors
from model import My_Model
from metrics import calculate_metric
from dataset import MyDataSet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
w_seq2vec = json.load(open(r"PSMOE/mutation/data/wild_type_60.emb"))
m_seq2vec = json.load(open(r"PSMOE/mutation/data/mutation_type_60_mut.emb"))
train_mutation_data, train_label, train_wild_type_data, train_original_mutation_seqs, train_original_wild_type_seqs = process_xlsx(r"autodl-tmp/train_dataset_with_sequence.xlsx", 600)
val_mutation_data, val_label, val_wild_type_data, val_original_mutation_seqs, val_original_wild_type_seqs = process_xlsx(r"autodl-tmp/val.xlsx", 600)
train_dataset = MyDataSet(train_wild_type_data, train_mutation_data, train_label,train_original_wild_type_seqs,train_original_mutation_seqs)
val_dataset = MyDataSet(val_wild_type_data, val_mutation_data, val_label,val_original_wild_type_seqs,val_original_mutation_seqs)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def spearman_rank_loss(pred, true):

    pred_rank = torch.argsort(torch.argsort(pred))
    true_rank = torch.argsort(torch.argsort(true))
    n = len(pred)
    d = pred_rank - true_rank
    corr = 1.0 - (6.0 * torch.sum(d ** 2)) / (n * (n ** 2 - 1))
    loss = 1.0 - corr
    return loss

def train(model, train_iter, optimizer, criterion):
    model.train()

    for train_wild_data, train_mut_data, train_label, w_seq, m_seq in train_iter:
        optimizer.zero_grad()
        wild_data = train_wild_data.to(device)
        mut_data = train_mut_data.to(device)
        train_label = map_labels(train_label)
        train_label = train_label.to(device)
        w_vec = get_sequence_vectors(w_seq, w_seq2vec, device)
        m_vec = get_sequence_vectors(m_seq, m_seq2vec, device)
        outputs, aux_loss = model(wild_data, mut_data, w_vec, m_vec)
        pred_classes = torch.argmax(outputs, dim=1)
        rank_loss = spearman_rank_loss(pred_classes, train_label)
        main_loss = criterion(outputs, train_label)
        total_loss = main_loss + aux_loss + 0.3 * rank_loss
        total_loss.backward()
        optimizer.step()

        return total_loss

def val(model, val_iter):
    model.eval()
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for val_wild_data, val_mut_data, val_label, val_w_seq, val_m_seq in val_iter:
            wild_data = val_wild_data.to(device)
            mut_data = val_mut_data.to(device)
            val_label = map_labels(val_label)
            val_label = val_label.to(device)
            w_vec = get_sequence_vectors(val_w_seq, w_seq2vec, device)
            m_vec = get_sequence_vectors(val_m_seq, m_seq2vec, device)
            outputs, _ = model(wild_data, mut_data, w_vec, m_vec)
            all_outputs.append(outputs)
            all_labels.append(val_label)
    output = torch.cat(all_outputs)
    label = torch.cat(all_labels)
    metrics, val_acc, gcc = calculate_metric(output, label)
    return metrics, val_acc, gcc



if __name__ == "__main__":
    model = My_Model().to(device)
    lr = 2e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stop_counter = 0
    patience = 15
    best_acc = 0
    best_gcc = 0
    EPOCH = 50
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    results_list = []
    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        total_loss = train(model, train_iter, criterion, optimizer)
        loss_ls.append(total_loss.item())
        metrics, val_acc, gcc = val(model, val_iter)
        epoch_time = time.time() - t0
        avg_loss = np.mean(loss_ls)
        print(f"Epoch: {epoch + 1}, train_loss: {avg_loss}, val_acc: {colored(val_acc, 'red')}, time: {epoch_time:.2f}")
        for cls in [0, 1, 2]:
            print(f"class {cls} :")
            print(f"  sensitivity: {metrics[f'{cls}_SE']:.4f}")
            print(f"  Specificity: {metrics[f'{cls}_SP']:.4f}")
            print(f"  PPV: {metrics[f'{cls}_PPV']:.4f}")
            print(f"  NPV: {metrics[f'{cls}_NPV']:.4f}")
        print(f"val acc {val_acc}, GCC {gcc}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_gcc = gcc
            save_path = f'./output/PSMOE_Mutation{timestamp}.pth'
            torch.save(model.state_dict(), save_path)
            early_stop_counter=0
            print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_acc:.4f}")
            print(f"best_balanced_acc: {best_acc}, metric: {metrics}, gcc {best_gcc}")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(colored(f"Early stopping triggered at epoch {epoch + 1}", "red"))
            print(f"Best val_acc: {best_acc:.4f}")
            break




