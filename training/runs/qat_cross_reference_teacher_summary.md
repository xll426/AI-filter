# QAT cross reference/teacher summary

Full test set, Y channel only. SSIM omitted here for speed; previous task_qat_noedge_test_summary.md contains task_qat vs reference SSIM.

## Baseline
| item | PSNR | L1 | L2 |
|---|---:|---:|---:|
| input_vs_reference | 34.0913 | 0.01847592 | 0.00086775712 |
| fp32_teacher_vs_reference | 36.7181 | 0.01242504 | 0.00038421729 |

## Models
| model | W/B | ckpt iter | vs reference PSNR | vs reference L1 | vs teacher PSNR | vs teacher L1 | q_w range/use | q_b range/use | shift |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| teacher_qat_w12_b14 | W12/B14 | 1000 | 36.5618 | 0.01247776 | 56.0403 | 0.00130108 | -1862..306 / 0.909 | -7961..-470 / 0.972 | 12.0..12.0 |
| teacher_qat_w10_b12 | W10/B12 | 1000 | 36.3627 | 0.01272368 | 51.3205 | 0.00237744 | -466..76 / 0.910 | -1990..-117 / 0.972 | 10.0..10.0 |
| task_qat_w12_b14_noedge | W12/B14 | 1000 | 36.7419 | 0.01230410 | 54.3051 | 0.00157892 | -1864..305 / 0.910 | -7956..-474 / 0.971 | 12.0..12.0 |
| task_qat_w10_b12_noedge | W10/B12 | 1000 | 36.7303 | 0.01230828 | 54.5418 | 0.00153511 | -466..76 / 0.910 | -1989..-119 / 0.971 | 10.0..10.0 |
