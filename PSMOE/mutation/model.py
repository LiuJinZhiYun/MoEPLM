from mamba_ssm.models.mixer_seq_simple import Mamba2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEConfig:
    def __init__(self):
        self.num_experts_per_tok = 2
        self.n_routed_experts = 4
        self.num_shared_experts = 1
        self.scoring_func = 'softmax'
        self.aux_loss_alpha = 0.1
        self.norm_topk_prob = True
        self.input_size = 256
        self.output_dim = 128
        self.hidden_dim = 128
config = MoEConfig()

class AddAuxiliaryLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        loss, = ctx.saved_tensors
        grad_loss = torch.ones_like(loss) if loss.numel() == 1 else torch.ones_like(loss).mean()
        return grad_output, grad_loss

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.register_buffer('expert_counts', torch.zeros(config.n_routed_experts))
        self.register_buffer('total_steps', torch.tensor(0))
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.input_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = None
        if self.training and self.alpha > 0.0:
            Pi = scores.mean(dim=0)
            expert_counts = F.one_hot(topk_idx, num_classes=self.num_experts).sum(dim=[0, 1]).float()
            fi = expert_counts * self.num_experts / (bsz * self.top_k)
            load_balance_loss = torch.sum(Pi * fi)
            aux_loss = self.alpha * load_balance_loss
            if aux_loss.numel() > 1:
                aux_loss = aux_loss.mean()
        return topk_idx, topk_weight, aux_loss

class MLPExpert(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128, dropout: float = 0.4):
        super().__init__()
        self.n_hidden_layers = 1
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.hidden.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ))
        self.output = nn.Sequential(*[
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        ])

    def forward(self, x):
        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        o = self.output(o)
        return o

class DeepseekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.expert_dim = config.output_dim
        self.num_shared_experts = config.num_shared_experts
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([
            MLPExpert(input_dim=config.input_size,hidden_dim=config.hidden_dim,output_dim=self.expert_dim)
            for _ in range(self.num_experts)
        ])
        self.num_experts_per_tok=config.num_experts_per_tok
        self.shared_experts = MLPExpert(input_dim=config.input_size,hidden_dim=config.hidden_dim,output_dim=self.expert_dim)

    def forward(self, combined):
        topk_indices, topk_weights, aux_loss = self.gate(combined)
        identity = combined
        if self.training:
            combined = combined.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.zeros(combined.size(0), self.expert_dim, device=combined.device)
            flat_topk_idx = topk_indices.view(-1)

            for i, expert in enumerate(self.experts):
                expert_input = combined[flat_topk_idx == i]
                if expert_input.size(0) < 2:
                    print(f"Warning: Expert {i} skipped due to insufficient samples ({expert_input.size(0)})")
                    continue
                expert_output = expert(expert_input)
                y[flat_topk_idx == i] = expert_output
            y = y.view(*topk_weights.shape, -1)
            y = (y * topk_weights.unsqueeze(-1)).sum(dim=1)
            y = AddAuxiliaryLoss.apply(y, aux_loss)

        else:
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            weights_sum = (expert_mask * topk_weights.unsqueeze(-1)).sum(dim=1)
            expert_outputs = torch.stack([expert(combined) for expert in self.experts], dim=1)
            moe_output = (expert_outputs * weights_sum.unsqueeze(-1)).sum(dim=1)
            y = moe_output
        shared_feature = self.shared_experts(identity)
        final_output = y + shared_feature
        return final_output, aux_loss


class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size,padding_idx=0)
        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs):
        embeds = self.embed(inputs).transpose(-1,-2)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2)
        embeds = self.projection(res_embed)
        return embeds


class My_Model(nn.Module):
    def __init__(self,vocab_size=24, embedding_dim=128):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.block3 = nn.Sequential(
            nn.Linear(1152, 128),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mamba=Mamba2(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=32,
            device="cuda",
        )
        self.moe = DeepseekMoE(
            config=config,
        )
        conv_filters = [[1, 32], [7, 128]]
        self.emb = ConvEmbedding(vocab_size=vocab_size, embedding_size=embedding_dim,
                                 conv_filters=conv_filters, output_dim=embedding_dim)

    def forward(self, wild_x, mut_x, w_vec, m_vec):
        wild_embeds = self.emb(wild_x)
        wild_output = self.mamba(wild_embeds)

        mut_embeds = self.emb(mut_x)
        mut_output = self.mamba(mut_embeds)

        diff = torch.abs(wild_output - mut_output)
        diff_sum = torch.sum(diff, dim=2)
        topk_indices = torch.topk(diff_sum, k=30, dim=1)[1]

        wild_selected = torch.gather(wild_output, 1, topk_indices.unsqueeze(-1).expand(-1, -1, wild_output.size(2)))
        w_pooled_output = torch.mean(wild_selected, dim=1)
        w_output = self.block1(w_pooled_output)
        w_vec = self.block3(w_vec)
        w_output = w_output + w_vec

        mut_selected = torch.gather(mut_output, 1, topk_indices.unsqueeze(-1).expand(-1, -1, mut_output.size(2)))
        m_pooled_output = torch.mean(mut_selected, dim=1)
        m_output = self.block1(m_pooled_output)
        m_vec = self.block3(m_vec)
        m_output = m_output + m_vec

        seq_features = torch.cat((w_output, m_output), dim=1)
        moe_output, aux_loss = self.moe(seq_features)
        logits = self.block2(moe_output)
        logits = torch.softmax(logits, dim=1)

        return logits, aux_loss


