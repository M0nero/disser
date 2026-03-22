from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjacency import _normalize_adjacency, _hand_adjacency_42
from .attention import AttentionPoolVT, ChannelLayerNorm, NodeAttention, StreamAttention
from .blocks import GCNBlock, StreamStem
from .encoder import StreamFeatureEncoder
from .heads import CosineClassifier


class MultiStreamAGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        V: int = 42,
        A: Optional[torch.Tensor] = None,
        in_ch: int = 3,
        streams: Sequence[str] = ("joints", "bones", "velocity"),
        drop: float = 0.25,
        droppath: float = 0.1,
        depths: Sequence[int] = (64, 128, 256, 256),
        temp_ks: Sequence[int] = (9, 7, 5, 5),
        use_groupnorm_stem: bool = True,
        stream_drop_p: float = 0.15,
        use_cosine_head: bool = False,
        cosine_margin: float = 0.20,
        cosine_scale: float = 30.0,
        use_ctr_hand_refine: bool = False,
        ctr_groups: int = 4,
        ctr_hand_nodes: int = 42,
        ctr_rel_channels: Optional[int] = None,
        ctr_alpha_init: float = 0.0,
    ):
        super().__init__()
        assert len(depths) == len(temp_ks) == 4
        self.streams = list(streams)
        self.stream_drop_p = float(stream_drop_p)
        self.V = int(V)
        ctr_rel_channels = None if (ctr_rel_channels is not None and ctr_rel_channels <= 0) else ctr_rel_channels

        # adjacency
        A = _hand_adjacency_42() if A is None else _normalize_adjacency(A)
        self.register_buffer("A", A)  # (K,V,V) or (1,V,V)

        # stems per stream
        stem_out = depths[0] // 2
        self.stems = nn.ModuleDict(
            {s: StreamStem(in_ch, stem_out, use_groupnorm_stem, temp_k=5) for s in self.streams}
        )

        # fusion + attention
        self.fuse = StreamAttention(stem_out, self.streams)
        self.node_attn = NodeAttention(stem_out)
        self.stream_encoder = (
            StreamFeatureEncoder(
                self.streams,
                stem_out,
                A,
                drop=drop * 0.5,
                droppath=droppath * 0.5,
            )
            if len(self.streams) > 1
            else None
        )
        self.pre_fuse_norm = ChannelLayerNorm(stem_out)

        # backbone blocks
        chs = [stem_out, *depths]
        blocks = []
        for i in range(4):
            stride_t = 2 if i == 0 else 1
            blocks.append(
                GCNBlock(
                    chs[i],
                    chs[i + 1],
                    A,
                    temp_k=temp_ks[i],
                    stride_t=stride_t,
                    drop=drop,
                    droppath=droppath * (i + 1) / 4.0,
                    use_mstcn=(i > 0),
                    use_se=True,
                    use_ctr_hand_refine=use_ctr_hand_refine,
                    ctr_groups=ctr_groups,
                    ctr_hand_nodes=ctr_hand_nodes,
                    ctr_rel_channels=ctr_rel_channels,
                    ctr_alpha_init=ctr_alpha_init,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # pooling
        self.pool = AttentionPoolVT(depths[-1])
        self.embed_norm = nn.LayerNorm(depths[-1])

        # head
        if use_cosine_head:
            self.head = CosineClassifier(depths[-1], num_classes, s=cosine_scale, m=cosine_margin)
            self._use_cos = True
        else:
            self.head = nn.Sequential(
                nn.Linear(depths[-1], depths[-1]),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(depths[-1], num_classes),
            )
            self._use_cos = False

    def _apply_stream_dropout(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Zero-out entire streams during training, keep keys; ensure >=1 stays."""
        if not self.training or self.stream_drop_p <= 0.0 or len(feats) <= 1:
            return feats
        streams = list(feats.keys())
        dev = next(iter(feats.values())).device
        keep_mask = torch.rand(len(streams), device=dev) > self.stream_drop_p
        if not keep_mask.any():
            idx = torch.randint(0, len(streams), (1,), device=dev)
            keep_mask[idx] = True
        for i, s in enumerate(streams):
            if not bool(keep_mask[i]):
                feats[s] = feats[s] * 0.0
        return feats

    def forward(
        self,
        X: Dict[str, torch.Tensor],  # streams -> (B,C,V,T)
        mask: Optional[torch.Tensor] = None,  # (B,1,V,T)
        A: Optional[torch.Tensor] = None,  # (V,V) or (K,V,V)
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert isinstance(X, dict) and len(X) > 0, "X must be dict of streams"
        X = {k: v for k, v in X.items() if k in self.streams}
        if len(X) == 0:
            raise ValueError("No known streams provided")

        feats = {s: self.stems[s](x) for s, x in X.items()}  # (B,c0,V,T)
        feats = self._apply_stream_dropout(feats)
        ref_feat = next(iter(feats.values()))
        A_eff = (
            _normalize_adjacency(A).to(ref_feat.device, ref_feat.dtype) if A is not None else self.A.to(ref_feat.device, ref_feat.dtype)
        )
        if self.stream_encoder is not None and len(feats) > 1:
            feats = self.stream_encoder(feats, mask, A_eff)

        cur_mask = mask
        if len(feats) == 1:
            y_feat = next(iter(feats.values()))
        else:
            y_feat = self.fuse(feats, cur_mask)
        y_feat = self.node_attn(y_feat, cur_mask)
        y_feat = self.pre_fuse_norm(y_feat)

        for blk in self.blocks:
            y_feat = blk(y_feat, cur_mask, A_override=A_eff)
            if cur_mask is not None and cur_mask.shape[-1] != y_feat.shape[-1]:
                cur_mask = F.interpolate(cur_mask.float(), size=(cur_mask.shape[2], y_feat.shape[-1]), mode="nearest")

        g = self.pool(y_feat, cur_mask)  # (B,C)
        g = self.embed_norm(g)
        out = self.head(g, y) if self._use_cos else self.head(g)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
