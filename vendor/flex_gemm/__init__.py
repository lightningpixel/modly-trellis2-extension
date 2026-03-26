"""
flex_gemm shim for Modly's TRELLIS.2 extension.

flex_gemm is a proprietary CUDA extension from Microsoft that is not
available on PyPI. This shim provides a PyTorch-based fallback for the
only operation used at inference time: ops.grid_sample.grid_sample_3d.

Sparse-convolution ops (ops.spconv.*) are NOT implemented here because
they are only called when SPARSE_CONV_BACKEND=flex_gemm, which Modly
never sets — it always uses spconv instead.
"""

from . import ops
