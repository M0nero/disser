from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .blocks import GCNBlock


class StreamFeatureEncoder(nn.Module):
    """Light per-stream GCN encoder before fusion (keeps modalities separate longer)."""

    def __init__(
        self,
        streams: Sequence[str],
        ch: int,
        A: torch.Tensor,
        drop: float,
        droppath: float,
        *,
        use_ctr_hand_refine: bool = False,
        ctr_groups: int = 4,
        ctr_hand_nodes: int = 42,
        ctr_rel_channels: Optional[int] = None,
        ctr_alpha_init: float = 0.0,
    ):
        super().__init__()
        self.streams = list(streams)
        self.blocks = nn.ModuleDict(
            {
                s: GCNBlock(
                    ch,
                    ch,
                    A,
                    temp_k=5,
                    stride_t=1,
                    drop=drop,
                    droppath=droppath,
                    use_mstcn=False,
                    use_se=True,
                    use_ctr_hand_refine=use_ctr_hand_refine,
                    ctr_groups=ctr_groups,
                    ctr_hand_nodes=ctr_hand_nodes,
                    ctr_rel_channels=ctr_rel_channels,
                    ctr_alpha_init=ctr_alpha_init,
                )
                for s in self.streams
            }
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        A_override: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        for name, block in self.blocks.items():
            if name not in feats:
                continue
            encoded[name] = block(feats[name], mask, A_override=A_override)
        # keep pass-through streams (e.g., if some were missing during init)
        for name, feat in feats.items():
            if name not in encoded:
                encoded[name] = feat
        return encoded
