import json
import numpy as np
import torch
import time
from termcolor import colored
import datetime
import torch.optim as optim
from utils import genData, get_sequence_vectors
from dataset import MyDataSet
from metrics import evaluate_accuracy
from model import My_Model
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
seq2vec = json.load(open(r"D:\Pycharm\pythonProjectDDD\MOE_NewData\esmc600m_all_data_zky.emb"))
train_data, train_label, train_seq= genData(r"D:\Pycharm\pythonProjectDDD\MOE_NewData\Train_dataset.fasta", 600)
val_data,val_label,val_seq=genData(r"D:\Pycharm\pythonProjectDDD\MOE_NewData\validation_dataset.fasta", 600)
train_dataset = MyDataSet(train_data, train_label, train_seq)
val_dataset = MyDataSet(val_data, val_label, val_seq)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):

        pred_positive = pred[:, 1]
        ce_loss = F.binary_cross_entropy_with_logits(pred_positive, target.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
        focal_loss = focal_loss.mean()
        return focal_loss

def train(model, train_iter, optimizer, criterion):
    model.train()
    for x, y, z in train_iter:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        vec = get_sequence_vectors(z, seq2vec, device)
        outputs, aux_loss = model.trainModel(x, vec)
        main_loss = criterion(outputs, y)
        total_loss = main_loss + aux_loss
        total_loss.backward()
        optimizer.step()
        return total_loss


def val(model, val_iter):
    prelabel_list = []
    relabel_list = []
    pre_pro_list = []
    model.eval()
    with torch.no_grad():
        for x, y, z in val_iter:
            x, y = x.to(device), y.to(device)
            vec = get_sequence_vectors(z, seq2vec, device)
            outputs, _ = model.trainModel(x, vec)
            prelabel_list.append(outputs.argmax(dim=1).cpu().numpy())
            relabel_list.append(y.cpu().numpy())
            output = torch.softmax(outputs, dim=1)
            pre_pro_list.append(output[:, 1].cpu().detach().numpy())
            val_acc = evaluate_accuracy(output, y, seq2vec, device)
        return val_acc



if __name__ == "__main__":
    model = My_Model().to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=1, min_lr=1e-6)
    alpha = torch.tensor([0.6062, 0.3938])
    criterion = FocalLoss(alpha=alpha).to(device)
    early_stop_counter = 0
    patience = 5
    best_acc = 0
    EPOCH = 30
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    results_list = []
    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        total_loss = train(model, train_iter, criterion, optimizer)
        loss_ls.append(total_loss.item())
        val_acc = val(model, val_iter)
        epoch_time = time.time() - t0
        avg_loss = np.mean(loss_ls)
        print(f"Epoch: {epoch + 1}, train_loss: {avg_loss}, val_acc: {colored(val_acc, 'red')}, time: {epoch_time:.2f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'./output/PSMOE_Solubility{timestamp}.pth'
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
            print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_acc:.4f}")
            print(f"best_balanced_acc: {best_acc}")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(colored(f"Early stopping triggered at epoch {epoch + 1}", "red"))
            print(f"Best val_acc: {best_acc:.4f}")
            break





