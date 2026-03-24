"""Back-compat module: re-export dataset utilities from msagcn.data."""

from msagcn.data import ClassBalancedBatchSampler, DSConfig, MultiStreamGestureDataset, build_sample_weights

__all__ = ["DSConfig", "MultiStreamGestureDataset", "build_sample_weights", "ClassBalancedBatchSampler"]
