from __future__ import annotations

import timm
import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        cos = torch.repeat_interleave(frequencies.cos(), 2, dim=-1).to(dtype=dtype)
        sin = torch.repeat_interleave(frequencies.sin(), 2, dim=-1).to(dtype=dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Transformer dimension must be divisible by the number of heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        cos, sin = self.rope(seq_len, x.device, query.dtype)
        query = apply_rotary_embedding(query, cos, sin)
        key = apply_rotary_embedding(key, cos, sin)

        attention_scores = (query * self.scale) @ key.transpose(-2, -1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended = attention_weights @ value
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.projection_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RotaryTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = RotarySelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.feedforward_norm = nn.LayerNorm(dim)
        self.feedforward = FeedForward(dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class FramewiseAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        framewise_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attention(tokens), dim=1)
        pooled = (framewise_logits * weights).sum(dim=1)
        return pooled, weights


class SEDAttentionHead(nn.Module):
    """Sound Event Detection head with parallel attention + max pooling.

    Pools features first, then classifies — avoids noisy frame-level logit aggregation.
    Uses the dual-pooling strategy from PANNs / top BirdCLEF solutions.
    """

    def __init__(self, dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.att_fc = nn.Linear(dim, num_classes)
        self.cls_fc = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D)
        x = self.dropout(torch.relu(self.fc(tokens)))
        # Per-class attention weights across frames
        att_weights = torch.softmax(self.att_fc(x), dim=1)  # (B, T, C)
        # Frame-level class logits
        frame_logits = self.cls_fc(x)  # (B, T, C)
        # Clip-level: attention-weighted sum + max-pool, averaged
        clip_att = (frame_logits * att_weights).sum(dim=1)  # (B, C)
        clip_max = frame_logits.max(dim=1).values  # (B, C)
        return (clip_att + clip_max) / 2.0


def combine_primary_secondary_probabilities(
    primary_logits: torch.Tensor,
    secondary_logits: torch.Tensor,
) -> torch.Tensor:
    # Cast to float32 for numerical stability (inputs may be float16 from AMP)
    primary_logits = primary_logits.float()
    secondary_logits = secondary_logits.float()
    primary_probs = torch.softmax(primary_logits, dim=-1)
    secondary_probs = torch.sigmoid(secondary_logits)
    combined = 1.0 - (1.0 - primary_probs) * (1.0 - secondary_probs)
    return torch.clamp(combined, 0.0, 1.0)


class GeMPooling(nn.Module):
    """Generalized Mean Pooling (GeM).

    Learnable pooling between average (p=1) and max (p→∞).
    Used by top BirdCLEF solutions (2024 2nd place, 2025 top-2%).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return (
            x.clamp(min=self.eps).pow(self.p).mean(dim=dim).pow(1.0 / self.p)
        )


class MultiScaleFeatureFusion(nn.Module):
    """Fuse features from multiple backbone stages.

    Extracts from last two stages, upsamples the deeper one if needed,
    and concatenates along the channel dimension before projecting down.
    This captures both fine-grained spectral texture (earlier stage) and
    high-level semantic features (later stage).
    """

    def __init__(self, channels_list: list[int], output_dim: int) -> None:
        super().__init__()
        total_channels = sum(channels_list)
        self.projection = nn.Sequential(
            nn.Linear(total_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        # Each feature map: (B, C_i, H_i, W_i) — make channel-first
        processed = []
        target_h = feature_maps[-1].shape[2]
        target_w = feature_maps[-1].shape[3]
        for fm in feature_maps:
            fm = to_channel_first(fm)
            if fm.shape[2] != target_h or fm.shape[3] != target_w:
                fm = nn.functional.adaptive_avg_pool2d(fm, (target_h, target_w))
            processed.append(fm)
        # Concatenate along channel dim: (B, sum(C_i), H, W)
        fused = torch.cat(processed, dim=1)
        # Collapse frequency, keep time: (B, sum(C_i), W)
        fused = fused.mean(dim=2).transpose(1, 2)  # (B, W, sum(C_i))
        return self.projection(fused)  # (B, W, output_dim)


def to_channel_first(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, got shape {tuple(feature_map.shape)}")
    # timm returns NHWC tensors for Swin features_only models and NCHW for ConvNeXt.
    if feature_map.shape[1] > feature_map.shape[-1]:
        return feature_map
    return feature_map.permute(0, 3, 1, 2).contiguous()


class TokenSemanticLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = FeedForward(dim=dim, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        class_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention_weights = self.cross_attention(
            query=self.query_norm(class_queries),
            key=self.token_norm(tokens),
            value=tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        class_queries = class_queries + attended
        class_queries = class_queries + self.output_mlp(self.output_norm(class_queries))
        return class_queries, attention_weights


class BirdClefSingleHeadClassifier(nn.Module):
    def __init__(
        self, num_classes: int, backbone: str, pretrained: bool, drop_path_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BirdClefDualHeadClassifier(nn.Module):
    def __init__(
        self, num_classes: int, backbone: str, pretrained: bool, drop_path_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        self.primary_head = nn.Linear(self.backbone.num_features, num_classes)
        self.secondary_head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "primary_logits": self.primary_head(features),
            "secondary_logits": self.secondary_head(features),
        }


class BirdClefTransformerSEDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        pretrained: bool,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        dropout: float,
        transformer_pooling: str,
        drop_path_rate: float = 0.0,
        multi_scale: bool = False,
        gem_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.multi_scale = multi_scale
        self.gem_pooling = gem_pooling

        if multi_scale:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=1,
                features_only=True,
                out_indices=(-2, -1),
                drop_path_rate=drop_path_rate,
            )
            channels_list = self.backbone.feature_info.channels()[-2:]
            self.feature_fusion = MultiScaleFeatureFusion(
                channels_list=channels_list,
                output_dim=transformer_dim,
            )
        else:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=1,
                features_only=True,
                out_indices=(-1,),
                drop_path_rate=drop_path_rate,
            )
            feature_channels = self.backbone.feature_info.channels()[-1]
            self.token_projection = nn.Linear(feature_channels, transformer_dim)

        if gem_pooling:
            self.gem = GeMPooling(p=3.0)

        self.sequence_encoder = nn.ModuleList(
            RotaryTransformerBlock(
                dim=transformer_dim,
                num_heads=transformer_heads,
                dropout=dropout,
            )
            for _ in range(transformer_layers)
        )
        self.transformer_pooling = transformer_pooling
        self.output_norm = nn.LayerNorm(transformer_dim)
        if transformer_pooling == "sed_attention":
            self.sed_head = SEDAttentionHead(
                dim=transformer_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif transformer_pooling == "clip_attention":
            self.attention_pool = AttentionPooling(transformer_dim, dropout=dropout)
            self.classifier = nn.Linear(transformer_dim, num_classes)
        else:
            self.frame_classifier = nn.Linear(transformer_dim, num_classes)
            if transformer_pooling == "attention":
                self.frame_attention_pool = FramewiseAttentionPooling(
                    dim=transformer_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                )
            elif transformer_pooling not in {"mean", "max"}:
                raise ValueError(f"Unknown transformer pooling mode: {transformer_pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_scale:
            feature_maps = self.backbone(x)[-2:]
            tokens = self.feature_fusion(feature_maps)
        else:
            feature_map = to_channel_first(self.backbone(x)[-1])
            # Collapse the frequency axis and keep the time axis as tokens.
            if self.gem_pooling:
                # GeM pool over frequency (dim=2): (B, C, H, W) -> (B, C, W)
                tokens = self.gem(feature_map, dim=2).transpose(1, 2)
            else:
                tokens = feature_map.mean(dim=2).transpose(1, 2)
            tokens = self.token_projection(tokens)

        for block in self.sequence_encoder:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        if self.transformer_pooling == "sed_attention":
            return self.sed_head(tokens)
        if self.transformer_pooling == "clip_attention":
            pooled, _ = self.attention_pool(tokens)
            return self.classifier(pooled)
        framewise_logits = self.frame_classifier(tokens)
        if self.transformer_pooling == "mean":
            return framewise_logits.mean(dim=1)
        if self.transformer_pooling == "max":
            return framewise_logits.max(dim=1).values
        pooled, _ = self.frame_attention_pool(tokens, framewise_logits)
        return pooled


class BirdClefHTSATModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        pretrained: bool,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        dropout: float,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if transformer_layers < 1:
            raise ValueError("HTS-AT requires at least one token-semantic layer.")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            out_indices=(-1,),
            drop_path_rate=drop_path_rate,
        )
        feature_channels = self.backbone.feature_info.channels()[-1]
        self.token_projection = nn.Linear(feature_channels, transformer_dim)
        self.semantic_head = nn.ModuleList(
            TokenSemanticLayer(
                dim=transformer_dim,
                num_heads=transformer_heads,
                dropout=dropout,
            )
            for _ in range(transformer_layers)
        )
        self.class_queries = nn.Parameter(torch.randn(num_classes, transformer_dim) * 0.02)
        self.output_norm = nn.LayerNorm(transformer_dim)
        self.classifier_weight = nn.Parameter(torch.randn(num_classes, transformer_dim) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = to_channel_first(self.backbone(x)[-1])
        tokens = feature_map.mean(dim=2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        class_queries = self.class_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.semantic_head:
            class_queries, _ = layer(tokens, class_queries)
        class_queries = self.output_norm(class_queries)
        logits = (class_queries * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        return logits + self.classifier_bias


def build_model(
    num_classes: int,
    architecture: str,
    backbone: str,
    pretrained: bool,
    classifier_head_mode: str = "single",
    transformer_dim: int = 256,
    transformer_heads: int = 8,
    transformer_layers: int = 2,
    dropout: float = 0.1,
    transformer_pooling: str = "clip_attention",
    drop_path_rate: float = 0.0,
    multi_scale: bool = False,
    gem_pooling: bool = False,
) -> nn.Module:
    if architecture == "efficientnet_classifier":
        if classifier_head_mode == "dual":
            return BirdClefDualHeadClassifier(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
            )
        if classifier_head_mode == "single":
            return BirdClefSingleHeadClassifier(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
            )
        raise ValueError(f"Unknown classifier head mode: {classifier_head_mode}")
    if architecture == "efficientnet_transformer_sed":
        return BirdClefTransformerSEDModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            transformer_pooling=transformer_pooling,
            drop_path_rate=drop_path_rate,
            multi_scale=multi_scale,
            gem_pooling=gem_pooling,
        )
    if architecture == "htsat_token_semantic":
        return BirdClefHTSATModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
    raise ValueError(f"Unknown architecture: {architecture}")
