import torch
import torch.nn as nn
import math
import random
import warnings
import numpy as np
from .SubLayer import PoswiseFeedForwardNet

random.seed(1234)
warnings.filterwarnings("ignore")


class DualInterAttention(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device, total_pep_len=22, hla_len=34):
        super(DualInterAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = d
        self.sigma = sigma
        self.d_model = d_model
        self.total_pep_len = total_pep_len
        self.hla_len = hla_len

        if window_threshold == "3":
            self.window_T = 0.01
        elif window_threshold == "2":
            self.window_T = math.exp(-4 / 2)
        elif window_threshold == '4':
            self.window_T = math.exp(-16 / 2)
        else:
            self.window_T = math.exp(-1 / 2)
        self.device = device
        self.W_Q = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d, d_model, bias=False)

        self.W_Q_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_K_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc_v = nn.Linear(n_heads * d, d_model, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)

    def initial_position(self, name, device, len_q, len_k):
        """
        len_q: [batch, 1]  Actual query sequence length (excluding padding)
        len_k: [batch, 1]  Actual key sequence length
        """
        if name == "HLA_pep":
            B = len_q.shape[0]
            k_pos = torch.arange(self.hla_len, device=device).float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

            # FIX: reshape (B,1) → (B,1,1) so that dividing a (1,1,L) arange tensor
            # yields (B,1,L) instead of the incorrect (1,B,L) caused by PyTorch
            # prepending a leading 1 to the 2-D len_q before broadcasting.
            len_q_val = len_q.view(B, 1, 1).clamp(min=1.0)
            q_pos_1 = 13. * (torch.arange(self.total_pep_len, device=device).float().unsqueeze(0).unsqueeze(0) / len_q_val)
            q_pos_1[q_pos_1 > 13] = 1e4
            q_pos_2 = 9. * (torch.arange(self.total_pep_len, device=device).float().unsqueeze(0).unsqueeze(0) / len_q_val) + 14.
            q_pos_2[q_pos_2 > 23] = 1e4
            q_pos_3 = 9 - 9. * (torch.arange(self.total_pep_len, device=device).float().unsqueeze(0).unsqueeze(0) / len_q_val) + 24.
            q_pos_3[q_pos_3 < 24] = 1e4
            q_pos = [q_pos_1, q_pos_2, q_pos_3]

        elif name == "HLApep_TCR":
            B = len_q.shape[0]
            # FIX: reshape (B,1) → (B,1,1) so that dividing a (1,1,30) arange tensor
            # yields (B,1,30) instead of the incorrect (1,B,30).  Without this fix,
            # q_pos_1 is (1,B,30), which then broadcasts with key_pos (B,1,1,L) to
            # produce a (B,B,30,L) tensor — an accidental batch-outer-product that
            # causes the observed 3.98 GiB OOM allocation.
            len_q_val = len_q.view(B, 1, 1).clamp(min=1.0)
            k_pos_pep = torch.arange(self.total_pep_len, device=device).float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            k_pos_hla = torch.arange(self.hla_len, device=device).float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            k_pos = torch.cat([k_pos_pep, k_pos_hla], dim=-1)

            pep_len_safe = max(float(self.total_pep_len), 1.0)
            q_pos_1 = (torch.arange(30, device=device).float().unsqueeze(0).unsqueeze(0) + 1.) / ((len_q_val + 1.) * (pep_len_safe / 30.))
            q_pos_1[q_pos_1 > 1.] = -1
            q_pos_1[q_pos_1 < 8. / pep_len_safe] = -1
            q_pos_1[q_pos_1 > 0] = 7. / 10
            q_pos_1 = q_pos_1 * (pep_len_safe - 1)
            q_pos_1[q_pos_1 < 0] = -1e4
            q_pos = [q_pos_1]
        else:
            raise ValueError(f"Invalid interaction name: {name}. "
                             f"L_Q={len_q.shape}, L_K={len_k.shape}, "
                             f"total_pep_len={self.total_pep_len}, hla_len={self.hla_len}")
        return q_pos, k_pos

    def forward(self, query, key, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k):
        residual_q = query
        residual_k = key
        B, L_Q, _ = query.shape
        B, L_K, _ = key.shape

        if L_Q == self.total_pep_len and L_K == self.hla_len:
            name = "HLA_pep"
        elif L_Q == 30 and L_K == self.hla_len + self.total_pep_len:
            name = "HLApep_TCR"
        else:
            raise ValueError(
                f"[DualInterAttention] Sequence lengths do not match the model parameters!\n"
                f"  L_Q={L_Q}, L_K={L_K}\n"
                f"  Expected HLA_pep:    L_Q=total_pep_len({self.total_pep_len}), L_K=hla_len({self.hla_len})\n"
                f"  Expected HLApep_TCR: L_Q=30, L_K=hla_len+total_pep_len({self.hla_len + self.total_pep_len})\n"
                f"  Check that --pep_max_len / --hla_alpha_max_len / --hla_beta_max_len match the actual data lengths."
            )

        Q = self.W_Q(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        V = self.W_V(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)

        Q_v = self.W_Q_v(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        K_v = self.W_K_v(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        V_v = self.W_V_v(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)

        context = Q
        context_v = Q_v

        query_positions, key_pos = self.initial_position(name, Q.device, len_q, len_k)

        # ★ Optimization 1: call generate_space_matrix once initially, then recompute after update_pos inside the loop
        # Previously: 1 calculation before the loop + 2 inside = 3; now: 1 initial calculation + 1 after update_pos = 2
        # Equivalent to the original logic: the pre-loop calculation supplies the first update_pos call
        amino_attn, full_amino_attn = self.generate_amino_matrix(context, K, pep_attn_mask)
        amino_attn_v, _ = self.generate_amino_matrix(context_v, K_v, hla_attn_mask)

        full_amino_attn.masked_fill_(amino_mask.unsqueeze(1) < 1.0, -1e4)
        amino_attn.masked_fill_(amino_mask.unsqueeze(1) < 1.0, -1e4)

        # Calculate once before the loop for the first update_pos call
        space_attn = self.generate_space_matrix(name, Q.device, key_pos, query_positions)
        for i in range(2):
            query_positions = self.update_pos(amino_attn, space_attn, query_positions, key_pos, name)
            space_attn = self.generate_space_matrix(name, Q.device, key_pos, query_positions)

        # ★ Optimization 2: replace torch.where(isnan, full_like, ...) with nan_to_num
        # torch.where(isnan, ...) requires a full_like temporary tensor and a GPU synchronization;
        # nan_to_num is an in-place fused kernel without extra allocation or synchronization points
        attn = torch.softmax((amino_attn + space_attn).masked_fill_(amino_mask.unsqueeze(1) < 1.0, -1e4), dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_sum = attn.sum(-1).unsqueeze(-1).clamp(0.0, 1.0)
        context = torch.matmul(attn, V) + (1 - attn_sum) * context

        attn_v = torch.softmax(
            (amino_attn_v + space_attn.transpose(-1, -2)).masked_fill_(
                amino_mask.unsqueeze(1).transpose(-1, -2) < 1.0, -1e4), dim=-1)
        attn_v = torch.nan_to_num(attn_v, nan=0.0)
        attn_v_sum = attn_v.sum(-1).unsqueeze(-1).clamp(0.0, 1.0)
        context_v = torch.matmul(attn_v, V_v) + (1 - attn_v_sum) * context_v

        context = context * attn_sum.clamp(min=0.0)
        context_v = context_v * attn_v_sum.clamp(min=0.0)

        context = context.transpose(1, 2).reshape(B, -1, self.n_heads * self.dim)
        context_v = context_v.transpose(1, 2).reshape(B, -1, self.n_heads * self.dim)

        output = self.fc(context)
        output_v = self.fc_v(context_v)

        return (self.layer_norm_q(output + residual_q),
                self.layer_norm_k(output_v + residual_k),
                attn, attn_v,
                (full_amino_attn + space_attn).masked_fill_(amino_mask.unsqueeze(1) < 1.0, -1e4))

    def generate_amino_matrix(self, Q, K, attn_mask):
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim)
        full_scores = scores.clone()
        scores.masked_fill_(attn_mask, -1e4)
        return scores, full_scores

    def generate_space_matrix(self, name, device, key_pos, query_positions, transpose=False):
        # Guard against division by zero in sigma
        sigma2 = max(self.sigma * self.sigma, 1e-8)
        spaceMask = None
        if name == "HLA_pep":
            key = key_pos.unsqueeze(-2)
            q_1 = query_positions[0].unsqueeze(-1)
            q_2 = query_positions[1].unsqueeze(-1)
            q_3 = query_positions[2].unsqueeze(-1)
            k_q_1 = torch.exp(-torch.square(q_1 - key[:, :, :, :14]) / (2 * sigma2))
            k_q_2 = torch.exp(-torch.square(q_2 - key[:, :, :, 14:24]) / (2 * sigma2))
            k_q_3 = torch.exp(-torch.square(q_3 - key[:, :, :, 24:34]) / (2 * sigma2))
            spaceMask = torch.cat([k_q_2[:, :, :, :4], k_q_1, k_q_2[:, :, :, 4:10], k_q_3], dim=-1)
        elif name == "HLApep_TCR":
            key = key_pos.unsqueeze(-2)
            q_1 = query_positions[0].unsqueeze(-1)
            k_q_pep = torch.exp(-torch.square(q_1 - key[:, :, :, :self.total_pep_len]) / (2 * sigma2))
            B, H, _, _ = k_q_pep.shape
            hla_space = torch.zeros(B, H, 30, self.hla_len, dtype=torch.float32, device=device)
            spaceMask = torch.cat([k_q_pep, hla_space], dim=-1)
        # Replace exp overflow/NaN from extreme position differences (1e4) with zero
        spaceMask = torch.nan_to_num(spaceMask, nan=0.0, posinf=1.0, neginf=0.0)
        return spaceMask.transpose(-1, -2) if transpose else spaceMask

    def update_pos(self, amino_att, space_att, query_positions, key_pos, name):
        # ★ Optimization 3: replace .type('torch.FloatTensor') with .float()
        # .type(str) uses the slower string-based lookup; .float() is a direct call without string-parsing overhead
        mask = torch.ones_like(space_att, dtype=torch.float32, device=amino_att.device)
        mask[space_att >= self.window_T] = 0.
        mask = mask.bool()

        if mask.shape[1] != amino_att.shape[1]:
            if mask.shape[1] > amino_att.shape[1]:
                mask = mask[:, :amino_att.shape[1], :, :]
            else:
                mask = mask.expand(-1, amino_att.shape[1], -1, -1)

        target_head = amino_att.shape[1]
        for i in range(len(query_positions)):
            if query_positions[i].shape[1] != target_head:
                if query_positions[i].shape[1] == 1:
                    query_positions[i] = query_positions[i].expand(-1, target_head, -1)
                else:
                    query_positions[i] = query_positions[i][:, :target_head, :]

        if key_pos.shape[1] != 1:
            key_pos = key_pos[:, :1, :]

        if name == "HLA_pep":
            full_att_1 = torch.softmax(amino_att[:, :, :, 4:18].clone().masked_fill_(mask[:, :, :, 4:18], -1e4), dim=-1)
            # ★ Optimization 4: replace torch.where(isnan, full_like, ...) with nan_to_num to reduce temporary tensor allocation
            full_att_1 = torch.nan_to_num(full_att_1, nan=0.0)

            region2_indices = torch.cat([amino_att[:, :, :, :4].clone(), amino_att[:, :, :, 18:24].clone()], dim=-1)
            region2_mask = torch.cat([mask[:, :, :, :4], mask[:, :, :, 18:24]], dim=-1)
            full_att_2 = torch.softmax(region2_indices.masked_fill_(region2_mask, -1e4), dim=-1)
            full_att_2 = torch.nan_to_num(full_att_2, nan=0.0)

            full_att_3 = torch.softmax(amino_att[:, :, :, 24:34].clone().masked_fill_(mask[:, :, :, 24:34], -1e4), dim=-1)
            full_att_3 = torch.nan_to_num(full_att_3, nan=0.0)

            new_pos_0 = torch.matmul(full_att_1, key_pos[:, :, :14].unsqueeze(-1)).squeeze(-1)
            new_pos_1 = torch.matmul(full_att_2, key_pos[:, :, 14:24].unsqueeze(-1)).squeeze(-1)
            new_pos_2 = torch.matmul(full_att_3, key_pos[:, :, 24:34].unsqueeze(-1)).squeeze(-1)
            query_positions[0] = torch.nan_to_num(new_pos_0, 0.0) + (1 - full_att_1.sum(-1)) * query_positions[0]
            query_positions[1] = torch.nan_to_num(new_pos_1, 0.0) + (1 - full_att_2.sum(-1)) * query_positions[1]
            query_positions[2] = torch.nan_to_num(new_pos_2, 0.0) + (1 - full_att_3.sum(-1)) * query_positions[2]

        elif name == "HLApep_TCR":
            s_idx, e_idx = 0, self.total_pep_len
            full_att_1 = torch.softmax(amino_att[:, :, :, s_idx:e_idx].clone().masked_fill_(mask[:, :, :, s_idx:e_idx], -1e4), dim=-1)
            full_att_1 = torch.nan_to_num(full_att_1, nan=0.0)
            new_pos_0 = torch.matmul(full_att_1, key_pos[:, :, s_idx:e_idx].unsqueeze(-1)).squeeze(-1)
            query_positions[0] = torch.nan_to_num(new_pos_0, 0.0) + (1 - full_att_1.sum(-1)) * query_positions[0]
        return query_positions


class IntercoderLayer(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device, d_ff, total_pep_len=22, hla_len=34):
        super(IntercoderLayer, self).__init__()
        self.dual_att = DualInterAttention(d_model, d, n_heads, sigma, window_threshold, device, total_pep_len, hla_len)
        self.pep_ffn = PoswiseFeedForwardNet(device=device, d_model=d_model, d_ff=d_ff)
        self.hla_ffn = PoswiseFeedForwardNet(device=device, d_model=d_model, d_ff=d_ff)

    def forward(self, pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k):
        pep_outputs, hla_outputs, pep_attn, hla_attn, attn = self.dual_att(
            pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k)
        pep_outputs = self.pep_ffn(pep_outputs)
        hla_outputs = self.hla_ffn(hla_outputs)
        return pep_outputs, hla_outputs, pep_attn, hla_attn, attn


class Intercoder(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device, d_ff, interact_layers, total_pep_len=22, hla_len=34):
        super(Intercoder, self).__init__()
        self.layers = nn.ModuleList([
            IntercoderLayer(d_model, d, n_heads, sigma, window_threshold, device, d_ff, total_pep_len, hla_len)
            for _ in range(interact_layers)
        ])

    def forward(self, pep, hla, amino_mask, pep_attn_mask, hla_attn_mask, len_q, len_k):
        pep_attns, hla_attns = [], []
        for layer in self.layers:
            pep, hla, pep_attn, hla_attn, attn = layer(
                pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k)
            pep_attns.append(pep_attn)
            hla_attns.append(hla_attn)
        return pep, hla, pep_attns, hla_attns, attn
