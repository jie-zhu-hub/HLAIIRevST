import math
import random
import warnings

import torch
import torch.nn as nn

from .DeepSlidingAttention import Intercoder

random.seed(1234)
warnings.filterwarnings("ignore")


# Retained for compatibility with external callers.









def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape[0], seq_q.shape[1]
    _, len_k = seq_k.shape[0], seq_k.shape[1]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=34):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, device, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.cov1 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.cov2 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gn1 = torch.nn.GroupNorm(num_groups=1, num_channels=d_model, eps=1e-5)
        self.gn2 = torch.nn.GroupNorm(num_groups=1, num_channels=d_model, eps=1e-5)

    def forward(self, enc_inputs):
        x = self.cov1(enc_inputs)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.cov2(x)
        x = self.gn2(x)
        x = x + enc_inputs
        enc_outputs = self.relu(x)
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, e_layers):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model) for _ in range(e_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_outputs.transpose(1, 2)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        enc_outputs = enc_outputs.transpose(2, 1)
        return enc_outputs






class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d = d
        self.n_heads = n_heads
        self.device = device
        self.W_Q = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_V, attn):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        all_neginf = torch.isneginf(attn).all(dim=-1, keepdim=True)
        attn = attn.masked_fill(all_neginf.expand_as(attn), 0.0)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(all_neginf.expand_as(attn), 0.0)
        context = torch.matmul(attn, V) + (1 - attn.sum(-1).unsqueeze(-1)) * Q
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d, n_heads, device, d_ff, peptide_total_len=22):
        super(DecoderLayer, self).__init__()
        self.peptide_total_len = peptide_total_len
        self.pep_self_attn = MultiHeadAttention(d_model, d, n_heads, device)
        self.tcr2pep_attn = MultiHeadAttention(d_model, d, n_heads, device)
        self.tcr2pep2hla_attn = MultiHeadAttention(d_model, d, n_heads, device)
        self.pep_pos_ffn = PoswiseFeedForwardNet(device, d_model, d_ff)
        self.tcr2hla_pos_ffn = PoswiseFeedForwardNet(device, d_model, d_ff)

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn_mask, tcr_attn_mask):
        pep_outputs, pep_attn = self.pep_self_attn(hla_inputs, pep_inputs, pep_attn_mask)
        tcr2pep_outputs, _ = self.tcr2pep_attn(pep_inputs, tcr_inputs, tcr_attn_mask[:, :, :self.peptide_total_len, :])
        tcr2pep2hla_outputs, _ = self.tcr2pep2hla_attn(hla_inputs, tcr2pep_outputs, pep_attn_mask)
        pep_outputs = self.pep_pos_ffn(pep_outputs)
        tcr2pep2hla_outputs = self.tcr2hla_pos_ffn(tcr2pep2hla_outputs)
        return pep_outputs, tcr2pep2hla_outputs, pep_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d, n_heads, device, d_ff, d_layers, peptide_total_len=22):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d, n_heads, device, d_ff, peptide_total_len) for _ in range(d_layers)])

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn, tcr_attn):
        hla_outputs = hla_inputs
        pep_outputs = pep_inputs
        tcr_outputs = tcr_inputs
        pep_attn = pep_attn.transpose(-1, -2)
        tcr_attn = tcr_attn.transpose(-1, -2)
        dec_self_attns = []
        for layer in self.layers:
            pep_outputs, tcr2hla, dec_self_attn = layer(hla_outputs, pep_outputs, tcr_outputs, pep_attn, tcr_attn)
            dec_self_attns.append(dec_self_attn)
        return pep_outputs, tcr2hla, dec_self_attns


class PepTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x


class PepTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, dropout=0.1, max_len=25):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            PepTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_inputs):
        src_mask = (enc_inputs == 0)
        x = self.src_emb(enc_inputs)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        d_model,
        e_layers,
        d,
        n_heads,
        sigma,
        window_threshold,
        d_ff,
        interact_layers,
        hla_alpha_max_len,
        hla_beta_max_len,
        pep_max_len,
        d_layers,
        dropout=0.1,
        HLA_fusion='concat',
        debug=False,
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.HLA_fusion = HLA_fusion
        self.hla_alpha_max_len = hla_alpha_max_len
        self.hla_beta_max_len = hla_beta_max_len
        self.pep_max_len = pep_max_len
        self.debug = debug
        self.d_model = d_model
        self.is_hlai_style_pair = (self.hla_alpha_max_len == self.hla_beta_max_len)

        # ------------------------------------------------------------------
        # The fused HLA length must be exactly 34 to match the spatial-prior splitting logic
        # for 34-position HLA sequences in DeepSlidingAttention.
        #   - Legacy MHCI path: 34 / 34 -> fused length remains 34
        #   - MHCII path: 15 / 19 -> sequential concatenation produces length 34
        # ------------------------------------------------------------------
        if self.HLA_fusion == 'avg':
            if self.hla_alpha_max_len != self.hla_beta_max_len:
                raise ValueError(
                    f"When HLA_fusion='avg', hla_alpha_max_len ({self.hla_alpha_max_len}) "
                    f"must equal hla_beta_max_len ({self.hla_beta_max_len})."
                )
            self.fused_hla_len = self.hla_alpha_max_len
        elif self.HLA_fusion in ['concat', 'concat+conv', 'concat+attention']:
            if self.is_hlai_style_pair:
                self.fused_hla_len = self.hla_alpha_max_len
            else:
                self.fused_hla_len = self.hla_alpha_max_len + self.hla_beta_max_len
        else:
            raise ValueError(f"Unknown fusion method: {self.HLA_fusion}")

        if self.fused_hla_len != 34:
            raise ValueError(
                f"The fused HLA length must be 34 for DeepSlidingAttention-compatible ternary modeling, "
                f"but got {self.fused_hla_len}. For MHCII please use alpha=15 and beta=19."
            )

        # ------------------------------------------------------------------
        # Retain the position-wise concatenation projection for compatibility with legacy MHCI checkpoints.
        # Used when alpha and beta have equal lengths (the legacy MHCI path);
        # when their lengths differ (MHCII: 15+19), use sequential concatenation instead of this layer.
        # ------------------------------------------------------------------
        self.hla_concat_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

        if self.HLA_fusion == 'concat+conv':
            self.conv_fusion = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        elif self.HLA_fusion == 'concat+attention':
            self.attn_fusion = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )

        # Encoders
        if self.HLA_fusion == 'avg':
            self.hla_encoder = Encoder(vocab_size, d_model, e_layers).to(device)
        else:
            self.hla_alpha_encoder = Encoder(vocab_size, d_model, e_layers).to(device)
            self.hla_beta_encoder = Encoder(vocab_size, d_model, e_layers).to(device)

        self.tcr_encoder = Encoder(vocab_size, d_model, e_layers).to(device)
        self.pep_encoder = PepTransformerEncoder(
            vocab_size, d_model, n_heads, d_ff, e_layers, dropout, max_len=pep_max_len
        ).to(device)

        self.binaryInteraction = Intercoder(
            d_model, d, n_heads, sigma, window_threshold, device,
            d_ff, interact_layers, total_pep_len=pep_max_len, hla_len=self.fused_hla_len
        ).to(device)
        self.ternaryInteraction = Intercoder(
            d_model, d, n_heads, sigma, window_threshold, device,
            d_ff, interact_layers, total_pep_len=pep_max_len, hla_len=self.fused_hla_len
        ).to(device)

        self.decoder = Decoder(
            d_model, d, n_heads, device, d_ff, d_layers,
            peptide_total_len=pep_max_len
        ).to(device)

        flatten_dim = 3 * d_model
        self.projection = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(True),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        ).to(device)
        self.register_buffer('_aa_acid', torch.tensor([14, 7, 0]))
        self.register_buffer('_aa_alkali', torch.tensor([5, 8, 17, 0]))
        self.register_buffer('_aa_hydrophobic', torch.tensor([19, 4, 12, 20, 11, 2, 10, 3, 0]))
        self.register_buffer('_aa_hydrogen', torch.tensor([17, 8, 14, 7, 13, 9, 6, 16, 5, 18, 2, 0]))

    def inactive_classification_parameters(self):
        # Exclude only from the optimizer; keep original flags and autograd paths.
        modules = []
        if self.HLA_fusion == 'avg' or not self.is_hlai_style_pair:
            modules.append(self.hla_concat_proj)
        if self.ternaryInteraction.layers:
            final = self.ternaryInteraction.layers[-1]
            modules.extend((final.dual_att.W_Q_v, final.dual_att.W_K_v,
                            final.dual_att.W_V_v, final.dual_att.fc_v,
                            final.dual_att.layer_norm_k, final.hla_ffn))
        return (parameter for module in modules for parameter in module.parameters())

    def _encode_physicochemical(self, seq):
        code = torch.zeros(*seq.shape, 4, dtype=torch.float32, device=seq.device)
        code[:, :, 0] = torch.isin(seq, self._aa_acid).float()
        code[:, :, 1] = torch.isin(seq, self._aa_alkali).float()
        code[:, :, 2] = torch.isin(seq, self._aa_hydrophobic).float()
        code[:, :, 3] = torch.isin(seq, self._aa_hydrogen).float()
        return code

    def _check_nan(self, tensor, name):
        if not self.debug:
            return
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            raise RuntimeError(
                f"[NaN/Inf check] {name} contains invalid values! "
                f"shape={list(tensor.shape)}, NaN={torch.isnan(tensor).sum().item()}, "
                f"inf={torch.isinf(tensor).sum().item()}"
            )

    def _apply_post_hla_fusion(self, fused_hla, fused_hla_seq):
        if self.HLA_fusion == 'concat':
            return fused_hla
        if self.HLA_fusion == 'concat+conv':
            return self.conv_fusion(fused_hla.transpose(1, 2)).transpose(1, 2)
        if self.HLA_fusion == 'concat+attention':
            key_padding_mask = (fused_hla_seq == 0)
            return self.attn_fusion(fused_hla, src_key_padding_mask=key_padding_mask)
        raise ValueError(f"Unknown fusion method: {self.HLA_fusion}")

    def _fuse_hla_positionwise(
        self,
        hla_alpha_enc,
        hla_beta_enc,
        hla_alpha_inputs,
        hla_beta_inputs,
        hla_alpha_code,
        hla_beta_code,
    ):
        concat_hla_enc = torch.cat([hla_alpha_enc, hla_beta_enc], dim=-1)
        fused_hla = self.hla_concat_proj(concat_hla_enc)
        fused_hla_seq = torch.where(hla_alpha_inputs > 0, hla_alpha_inputs, hla_beta_inputs)
        hla_code = torch.maximum(hla_alpha_code, hla_beta_code)
        fused_hla = self._apply_post_hla_fusion(fused_hla, fused_hla_seq)
        return fused_hla, fused_hla_seq, hla_code

    def _fuse_hla_sequential(
        self,
        hla_alpha_enc,
        hla_beta_enc,
        hla_alpha_inputs,
        hla_beta_inputs,
        hla_alpha_code,
        hla_beta_code,
    ):
        # MHCII path: concatenate alpha(15) + beta(19) along the sequence dimension to obtain 34 positions before attention/conv.
        fused_hla = torch.cat([hla_alpha_enc, hla_beta_enc], dim=1)
        fused_hla_seq = torch.cat([hla_alpha_inputs, hla_beta_inputs], dim=1)
        hla_code = torch.cat([hla_alpha_code, hla_beta_code], dim=1)
        fused_hla = self._apply_post_hla_fusion(fused_hla, fused_hla_seq)
        return fused_hla, fused_hla_seq, hla_code

    def forward(self, hla_alpha_inputs, hla_beta_inputs, pep_inputs, tcr_inputs):
        if self.HLA_fusion == 'avg':
            hla_inputs = torch.where(hla_alpha_inputs > 0, hla_alpha_inputs, hla_beta_inputs)
            fused_hla = self.hla_encoder(hla_inputs)
            fused_hla_seq = hla_inputs
            pep_code = self._encode_physicochemical(pep_inputs)
            hla_code = self._encode_physicochemical(hla_inputs)
        else:
            hla_alpha_enc = self.hla_alpha_encoder(hla_alpha_inputs)
            self._check_nan(hla_alpha_enc, "hla_alpha_enc")
            hla_beta_enc = self.hla_beta_encoder(hla_beta_inputs)
            self._check_nan(hla_beta_enc, "hla_beta_enc")

            pep_code = self._encode_physicochemical(pep_inputs)
            hla_alpha_code = self._encode_physicochemical(hla_alpha_inputs)
            hla_beta_code = self._encode_physicochemical(hla_beta_inputs)

            if self.is_hlai_style_pair:
                fused_hla, fused_hla_seq, hla_code = self._fuse_hla_positionwise(
                    hla_alpha_enc, hla_beta_enc,
                    hla_alpha_inputs, hla_beta_inputs,
                    hla_alpha_code, hla_beta_code,
                )
            else:
                fused_hla, fused_hla_seq, hla_code = self._fuse_hla_sequential(
                    hla_alpha_enc, hla_beta_enc,
                    hla_alpha_inputs, hla_beta_inputs,
                    hla_alpha_code, hla_beta_code,
                )

        pep_enc = self.pep_encoder(pep_inputs)
        self._check_nan(pep_enc, "pep_enc")
        tcr_enc = self.tcr_encoder(tcr_inputs)
        self._check_nan(tcr_enc, "tcr_enc")
        self._check_nan(fused_hla, "fused_hla")

        binary_mask = torch.matmul(pep_code, hla_code.transpose(-1, -2))
        binary_mask[binary_mask > 0] = 1

        pepHLA_mask = get_attn_pad_mask(pep_inputs, fused_hla_seq)
        HLApep_mask = get_attn_pad_mask(fused_hla_seq, pep_inputs)

        len_pep = (pep_inputs > 0).sum(dim=-1).unsqueeze(-1).float()
        len_hla = (fused_hla_seq > 0).sum(dim=-1).unsqueeze(-1).float()

        pep_inter, hla_inter, _, _, binary_pep_attn = self.binaryInteraction(
            pep_enc, fused_hla, binary_mask, pepHLA_mask, HLApep_mask, len_pep, len_hla
        )
        self._check_nan(pep_inter, "pep_inter (binaryInteraction)")
        self._check_nan(hla_inter, "hla_inter (binaryInteraction)")
        self._check_nan(binary_pep_attn, "binary_pep_attn")

        binary_out = torch.cat((pep_enc + pep_inter, fused_hla + hla_inter), dim=1)
        pHLA_seq = torch.cat([pep_inputs, fused_hla_seq], dim=1)
        pHLA_code = torch.cat([pep_code, hla_code], dim=1)

        tcr_code = self._encode_physicochemical(tcr_inputs)
        ternary_mask = torch.matmul(tcr_code, pHLA_code.transpose(-1, -2))
        ternary_mask[ternary_mask > 0] = 1

        TCRpHLA_mask = get_attn_pad_mask(tcr_inputs, pHLA_seq)
        pHLATCR_mask = get_attn_pad_mask(pHLA_seq, tcr_inputs)

        len_tcr = (tcr_inputs > 0).sum(dim=-1).unsqueeze(-1).float()
        len_phla = (pHLA_seq > 0).sum(dim=-1).unsqueeze(-1).float()

        tcr_inter, binary_inter, _, _, tcr_attn = self.ternaryInteraction(
            tcr_enc, binary_out, ternary_mask, TCRpHLA_mask, pHLATCR_mask, len_tcr, len_phla
        )
        self._check_nan(tcr_inter, "tcr_inter (ternaryInteraction)")
        self._check_nan(binary_inter, "binary_inter (ternaryInteraction)")
        self._check_nan(tcr_attn, "tcr_attn")

        pep_dec, tcr2hla_out, dec_self_attns = self.decoder(
            fused_hla + hla_inter,
            pep_enc + pep_inter,
            tcr_enc + tcr_inter,
            binary_pep_attn,
            tcr_attn,
        )
        self._check_nan(pep_dec, "pep_dec")
        self._check_nan(tcr2hla_out, "tcr2hla_out")

        pep_out_pooled = torch.mean(pep_dec, dim=1)
        tcr2hla_out_pooled = torch.mean(tcr2hla_out, dim=1)
        tcr_dec_pooled = torch.mean(tcr_enc + tcr_inter, dim=1)

        combined = torch.cat([pep_out_pooled, tcr2hla_out_pooled, tcr_dec_pooled], dim=1)
        output = self.projection(combined)
        self._check_nan(output, "output")

        return output, dec_self_attns, binary_pep_attn
