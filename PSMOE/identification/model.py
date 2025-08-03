from mamba_ssm.models.mixer_seq_simple import Mamba2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MoEConfig:
    def __init__(self):
        self.num_experts_per_tok = 8
        self.n_routed_experts = 16
        self.scoring_func = 'softmax'
        self.aux_loss_alpha = 0.2
        self.seq_aux = False
        self.norm_topk_prob = True
        self.input_size = 256
        self.output_dim=256
        self.hidden_dim=128
config = MoEConfig()

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = False  # 强制关闭seq_aux功能（适配二维输入）
        self.register_buffer('expert_counts', torch.zeros(config.n_routed_experts))
        self.register_buffer('total_steps', torch.tensor(0))
        self.diagnostic_interval = 5000  #

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
            expert_counts = torch.bincount(topk_idx.view(-1), minlength=config.n_routed_experts)
            self.expert_counts += expert_counts
            self.total_steps += 1

            # 定期输出诊断信息
            if self.total_steps % self.diagnostic_interval == 0:
                self._log_expert_distribution()
        return topk_idx, topk_weight, aux_loss




class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout: float = 0.25):
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
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        return self.output(o)


class DeepseekMoE(nn.Module):
    def __init__(self, config, num_shared_experts=1):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.expert_dim = config.output_dim
        self.num_shared_experts = num_shared_experts
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([
            MLPExpert(input_dim=config.input_size, hidden_dim=config.hidden_dim, output_dim=self.expert_dim)
            for _ in range(self.num_experts)
        ])
        self.num_experts_per_tok = config.num_experts_per_tok

        self.shared_experts = MLPExpert(input_dim=config.input_size, hidden_dim=config.hidden_dim,
                                        output_dim=self.expert_dim)

    def forward(self, combined):
        topk_indices, topk_weights, aux_loss = self.gate(combined)
        identity = combined
        if self.training:
            combined = combined.repeat_interleave(self.num_experts_per_tok, dim=0)  # 256 512
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

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    修正后的辅助损失函数，确保领域专家参数通过 aux_loss 更新。
    """

    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x  # 正向传播不修改输出

    @staticmethod
    def backward(ctx, grad_output):
        loss, = ctx.saved_tensors
        # 主任务梯度：grad_output 直接传递给前层
        # 辅助损失梯度：loss 的梯度为 1（因为总损失是 main_loss + loss）
        grad_loss = torch.ones_like(loss) if loss.numel() == 1 else torch.ones_like(loss).mean()
        return grad_output, grad_loss  # 返回主梯度和辅助梯度


class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
            self.convolutions.append(conv)

        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs):

        embeds = self.embed(inputs).transpose(-1, -2)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim=1).transpose(-1, -2)
        embeds = self.projection(res_embed)
        return embeds


conv_filters = [[1, 32], [7, 128]]
d_model = 512
d_state = 32
d_conv = 4
expand = 2
headdim = 32
# 加深模型深度
model = Mamba2(
    d_model=d_model,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand,
    headdim=headdim,
    device="cuda",
)


class My_Model(nn.Module):
    def __init__(self, vocab_size=24, embedding_dim=512):
        super().__init__()
        # 原始特征提取部分
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        self.emb = ConvEmbedding(vocab_size=vocab_size, embedding_size=embedding_dim,
                                 conv_filters=conv_filters, output_dim=embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)  # 512 8
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mamba = model
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim, 256),  # 输入维度对齐Transformer输出
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        self.block3 = nn.Sequential(
            nn.Linear(1152, 256),  # 输入维度调整为 embedding_dim
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.output_dim, 64),  # 输入维度对齐MoE输出
            nn.ReLU(),
            nn.Linear(64, 2)  # 直接输出
        )

        self.moe = DeepseekMoE(
            config=config,
        )

    def forward(self, x):
        """特征提取部分"""
        x = x.to(device)
        x = self.emb(x)

        output1 = self.mamba(x)
        output3 = torch.mean(output1, dim=1)
        output = self.block1(output3)
        return output

    def trainModel(self, x, vec):
        seq_features = self.forward(x)
        vec = self.block3(vec)
        combined = seq_features + vec
        moe_output, aux_loss = self.moe(combined)
        logits = self.block2(moe_output)
        return logits, aux_loss


