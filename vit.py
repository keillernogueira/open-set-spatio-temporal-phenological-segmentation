import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        feature_dim,
        seq_len,
        in_channels,
        embed_dim,
        arrgmnt="rgbrgb",
        region_size=5,
        dimension_order="TR",
        normalized_rgb=False,
        has_pos_encoding=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dimension_order = dimension_order
        self.in_channels = in_channels
        self.region_size = region_size
        self.arrgmnt = arrgmnt
        self.normalized_rgb = normalized_rgb
        self.dimension_order = dimension_order
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.has_pos_encoding = has_pos_encoding
        dim_1 = self.seq_len if self.dimension_order == "TR" else self.region_size

        if self.has_pos_encoding:
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + dim_1, embed_dim))
        else:
            self.register_buffer("pos_embed", torch.zeros(1, 1 + dim_1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.filter_region(x)
        x = self.normalize_rgb(x) if self.normalized_rgb else x
        x = self.define_dimension_order(x)
        x = self.get_feature_arrangement(x)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

    def get_feature_arrangement(self, x):
        B = x.size(0)
        dim_1 = self.seq_len if self.dimension_order == "TR" else self.region_size
        dim_2 = self.region_size if self.dimension_order == "TR" else self.seq_len

        if self.arrgmnt == "rgbrgb":
            x = x.reshape(B, dim_1, dim_2 * self.in_channels)
        else:

            x = x.view(
                B, dim_1, dim_2, self.in_channels
            )  # [B, seq_len, region_size, channels]
            x = x.permute(0, 1, 3, 2)  # [B, seq_len, channels, region_size]
            x = x.reshape(B, dim_1, self.in_channels * dim_2)
        return x

    def filter_region(self, x):
        region_5_filter = torch.tensor([1, 3, 4, 5, 7])
        region_1_filter = torch.tensor([4])
        if self.region_size == 5:
            x = x[:, :, region_5_filter, :]
        if self.region_size == 1:
            x = x[:, :, region_1_filter, :]
        return x

    def define_dimension_order(self, x):
        B = x.size(0)
        if self.dimension_order == "RT":
            x = x.permute(0, 2, 1, 3)
        return x

    def normalize_rgb(self, x):
        """
        x: Tensor of shape [B, T, R, 3]
        """
        eps = 1e-8
        # Sum over RGB channels
        rgb_sum = x.sum(dim=-1, keepdim=True)
        # Normalize
        x_norm = x / (rgb_sum + eps)

        return x_norm


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=drop_rate, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class OpenSetHead(nn.Module):
    def __init__(self, embed_dim, num_known_classes, threshold):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.cset_fc = nn.Linear(embed_dim, self.num_known_classes)
        self.oset_fc = nn.Linear(embed_dim, self.num_known_classes * 2)
        self.threshold = threshold

    def forward(self, feats):
        batch_size = feats.size(0)
        cset_logit = self.cset_fc(feats)
        oset_logit = self.oset_fc(feats).view(batch_size, 2, self.num_known_classes)
        if not self.training:
            feats_flatten = torch.flatten(feats, 1)
            oset_prob = F.softmax(oset_logit, dim=1)

            cset_pred = torch.max(cset_logit, dim=1)[1]
            is_unknown_prob = oset_prob[torch.arange(feats.size(0)), 1, cset_pred]
            oset_pred = (
                is_unknown_prob > self.threshold
            )  # Returns true or false for unknown depending on threshold
            return cset_pred, oset_pred, feats_flatten, is_unknown_prob
        else:
            oset_prob = F.softmax(oset_logit, dim=1)
            cset_pred = torch.max(cset_logit, dim=1)[1]
            # import pdb; pdb.set_trace()
            is_unknown_prob = oset_prob[torch.arange(feats.size(0)), 1, cset_pred]

            return cset_logit, oset_logit, is_unknown_prob


class VisionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        seq_len,
        in_channels,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_dim,
        drop_rate,
        region_size,
        feat_arrgmnt,
        data_structure,
        normalized_rgb,
        has_pos_encoding,
        pool_type="cls",
        open_set=False,
        threshold=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pool_type = (
            pool_type  # 'cls' for class token or 'gap' for global average pooling
        )
        self.patch_embed = PatchEmbedding(
            feature_dim,
            seq_len,
            in_channels,
            embed_dim,
            feat_arrgmnt,
            region_size,
            data_structure,
            normalized_rgb,
            has_pos_encoding,
        )
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if not open_set:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = OpenSetHead(embed_dim, num_classes, threshold)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.pool_type == "cls":
            # Use class token (index 0)
            x = x[:, 0]
        elif self.pool_type == "gap":
            # Global average pooling over all tokens
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

        return self.head(x)

    def get_num_classes(self):
        return self.num_classes


class OpenSetLoss(nn.Module):

    def __init__(self, loss_weights={"source_cset": 1.0, "source_oset": 0.5}):
        super().__init__()
        self.loss_weights = loss_weights

    def forward(self, logits, y):
        cset_logit, oset_logit, _ = logits

        batch_size = cset_logit.size(0)

        oset_prob = F.softmax(oset_logit, dim=1)

        oset_pos_target = torch.zeros_like(cset_logit)
        oset_pos_target[torch.arange(batch_size), y] = 1
        oset_neg_target = 1 - oset_pos_target

        cset_loss = F.cross_entropy(cset_logit, y)
        oset_pos_loss = torch.mean(
            torch.sum(-oset_pos_target * torch.log(oset_prob[:, 0, :] + 1e-8), dim=1)
        )
        oset_neg_loss = torch.mean(
            torch.max(-oset_neg_target * torch.log(oset_prob[:, 1, :] + 1e-8), dim=1)[0]
        )
        oset_loss = oset_pos_loss + oset_neg_loss

        loss = (
            cset_loss * self.loss_weights["source_cset"]
            + oset_loss * self.loss_weights["source_oset"]
        )
        return loss
