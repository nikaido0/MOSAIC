import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """随机深度丢弃模块，用于增强模型正则化"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PositionalEmbedding(nn.Module):
    """位置嵌入模块，为序列数据添加位置信息"""

    def __init__(self, embed_dim, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器模块，处理全局序列特征"""

    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, max_len=128, dropout=0.1):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(embed_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        src = self.pos_embedding(src)
        return self.transformer(src)


class MultiScaleLocalEnhancer(nn.Module):

    def __init__(self, dim, kernel_sizes=[3, 5, 7], groups=8, dropout=0.1):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        # 多尺度卷积块列表
        self.conv_blocks = nn.ModuleList([
            self._build_conv_block(dim, k, groups, dropout)
            for k in kernel_sizes
        ])

        # 融合权重，可学习参数
        self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)))

        # 输出投影卷积层
        self.output_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _build_conv_block(self, dim, kernel_size, groups, dropout):
        """构建卷积块，包含卷积 + GELU + SE 注意力 + Dropout"""
        return nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.GELU(),
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(dim, dim // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(dim // 8, dim, kernel_size=1),
                nn.Sigmoid()
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        输入:  x: [B, L, dim]
        输出: out: [B, L, dim]
        """
        x = x.permute(0, 2, 1)  # [B, dim, L]

        features = []
        for block in self.conv_blocks:
            out = block[0](x)  # Conv1d
            out = block[1](out)  # GELU
            se_weight = block[2](out)  # SE注意力
            out = out * se_weight  # SE加权
            out = block[3](out)  # Dropout
            features.append(out)  # 每个卷积尺度的输出 [B, dim, L]

        # 权重归一化融合
        stacked = torch.stack(features, dim=0)  # [K, B, dim, L]
        weights = F.softmax(self.fusion_weights.view(-1, 1, 1, 1), dim=0)  # [K, 1, 1, 1]
        fused = (stacked * weights).sum(dim=0)  # [B, dim, L]

        out = self.output_proj(fused)  # [B, dim, L]
        return out.permute(0, 2, 1)  # [B, L, dim]


class GateFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled, global_feat):
        x = torch.cat([pooled, global_feat], dim=-1)
        gate = self.gate(x)
        fused = gate * pooled + (1 - gate) * global_feat
        return self.norm(self.dropout(fused))


class AttentionPool(nn.Module):
    """注意力池化模块，将序列特征聚合成单一向量"""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return torch.sum(x * attn_weights, dim=1)


class MOSAIC(nn.Module):
    """多模态融合模型"""

    def __init__(self, pre_dim, seq_dim, global_dim, hidden_dim=256, fusion_dim=256,
                 num_transformer_layers=3, dropout=0.3, stochastic_depth=0.1):
        super().__init__()

        # 序列特征归一化和多尺度局部卷积（投影至 seq_proj_dim）
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.multiscale_conv = MultiScaleLocalEnhancer(seq_dim, dropout=dropout)

        self.transformer = TransformerEncoder(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            num_layers=num_transformer_layers,
            ff_dim=hidden_dim * 4,
            max_len=128,
            dropout=dropout
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

        # 投影 Transformer 输出到统一融合维度
        self.transformer_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pre_projection = nn.Sequential(
            nn.Linear(pre_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.seq_projection = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # global_features 投影到统一融合维度
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pool = AttentionPool(hidden_dim * 2)
        self.fusion = GateFusion(fusion_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(fusion_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pre_features, seq_features, global_features):
        if global_features.dim() == 3 and global_features.size(1) == 1:
            global_features = global_features.squeeze(1)

        # 局部特征提取 + 投影到 hidden_dim
        pre_features = self.pre_projection(pre_features)  # [B, L, hidden_dim]
        seq_features = self.multiscale_conv(seq_features)  # [B, L, seq_dim]
        seq_features = self.seq_projection(seq_features)  # [B, L, hidden_dim]

        # 拼接 pre_features 和 seq_features（特征维度拼接）
        combined = torch.cat([pre_features, seq_features], dim=-1)  # [B, L, pre_dim + seq_dim]

        # Transformer编码
        transformer_out = self.transformer(combined)  # [B, L, pre_dim + seq_dim]
        transformer_out = self.stochastic_depth(transformer_out)

        # 注意力池化序列到向量
        pooled = self.pool(transformer_out)

        # 投影到融合维度
        pooled_proj = self.transformer_projection(pooled)  # [B, fusion_dim]
        global_proj = self.global_projection(global_features)  # [B, fusion_dim]

        # 融合
        fused = self.fusion(pooled_proj, global_proj)  # [B, fusion_dim]

        # 分类
        return self.classifier(fused)


class MOSAIC_wo_global(nn.Module):
    def __init__(self, pre_dim, seq_dim, hidden_dim=256, fusion_dim=256,
                 num_transformer_layers=3, dropout=0.3, stochastic_depth=0.1):
        super().__init__()

        self.seq_norm = nn.LayerNorm(seq_dim)
        self.multiscale_conv = MultiScaleLocalEnhancer(seq_dim, dropout=dropout)

        self.pre_projection = nn.Sequential(
            nn.Linear(pre_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.seq_projection = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transformer = TransformerEncoder(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            num_layers=num_transformer_layers,
            ff_dim=hidden_dim * 4,
            max_len=128,
            dropout=dropout
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

        self.transformer_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(fusion_dim // 2, 1)
        )

        self.pool = AttentionPool(hidden_dim * 2)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pre_features, seq_features):
        pre_features = self.pre_projection(pre_features)
        seq_features = self.multiscale_conv(seq_features)
        seq_features = self.seq_projection(seq_features)

        combined = torch.cat([pre_features, seq_features], dim=-1)
        transformer_out = self.transformer(combined)
        transformer_out = self.stochastic_depth(transformer_out)

        pooled = self.pool(transformer_out)
        pooled_proj = self.transformer_projection(pooled)

        return self.classifier(pooled_proj)
