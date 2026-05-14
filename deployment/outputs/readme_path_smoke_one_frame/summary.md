# ONNX vs PyTorch Full Video Compare

input: `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/data/kaideo_2560x1440_yuv420p_0.yuv`

| model | frames | max abs Y diff | mean abs Y diff | diff Y ratio | ONNX output | PyTorch output |
|---|---:|---:|---:|---:|---|---|
| teacher_qat_w10_b12 | 1 | 0 | 0 | 0 | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/readme_path_smoke_one_frame/teacher_qat_w10_b12/kaideo_2560x1440_yuv420p_0_onnx_rate1.yuv` | `/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/outputs/readme_path_smoke_one_frame/teacher_qat_w10_b12/kaideo_2560x1440_yuv420p_0_pytorch_rate1.yuv` |
