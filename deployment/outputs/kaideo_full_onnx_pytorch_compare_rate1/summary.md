# ONNX vs PyTorch Full Video Compare

input: `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/data/kaideo_2560x1440_yuv420p_0.yuv`

| model | frames | max abs Y diff | mean abs Y diff | diff Y ratio | ONNX output | PyTorch output |
|---|---:|---:|---:|---:|---|---|
| edge_w10_b13 | 100 | 0 | 0 | 0 | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/edge_w10_b13/kaideo_2560x1440_yuv420p_0_onnx_rate1.yuv` | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/edge_w10_b13/kaideo_2560x1440_yuv420p_0_pytorch_rate1.yuv` |
| teacher_qat_w10_b12 | 100 | 0 | 0 | 0 | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/teacher_qat_w10_b12/kaideo_2560x1440_yuv420p_0_onnx_rate1.yuv` | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/teacher_qat_w10_b12/kaideo_2560x1440_yuv420p_0_pytorch_rate1.yuv` |
| fudan_fp16 | 100 | 1 | 1.6682943e-06 | 1.6682943e-06 | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/fudan_fp16/kaideo_2560x1440_yuv420p_0_onnx_rate1.yuv` | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/kaideo_full_onnx_pytorch_compare_rate1/fudan_fp16/kaideo_2560x1440_yuv420p_0_pytorch_rate1.yuv` |
