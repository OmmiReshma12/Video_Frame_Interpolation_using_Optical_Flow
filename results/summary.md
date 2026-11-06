Evaluated 600 triplets from 200 videos (gap=1)

| Method | PSNR (dB) mean ± std | SSIM mean ± std |
|---|---|---|
| average | 32.59 ± 7.81 | 0.9161 ± 0.1026 |
| flow_oneway_dis | 32.76 ± 7.50 | 0.9304 ± 0.0844 |
| dis | 33.66 ± 7.24 | 0.9379 ± 0.0809 |
| farneback | 33.44 ± 7.26 | 0.9346 ± 0.0826 |

DIS flow beats the cross-fade baseline (PSNR) on 370/600 triplets (61.7%).

### Results by motion amount (mean flow magnitude between A and B; thresholds 0.60 px and 2.18 px, equal-sized groups)

| Motion group | n | average PSNR | flow_oneway_dis PSNR | dis PSNR | farneback PSNR | DIS gain over average (dB) | DIS win rate |
|---|---|---|---|---|---|---|---|
| low motion | 200 | 39.97 | 39.24 | 39.92 | 39.49 | -0.05 | 46.5% |
| medium motion | 200 | 31.54 | 31.80 | 32.89 | 32.65 | +1.36 | 67.5% |
| high motion | 200 | 26.27 | 27.24 | 28.18 | 28.19 | +1.90 | 71.0% |

### Paired significance tests on per-triplet PSNR (two-sided)

| Comparison | mean diff (dB) | paired t-test p | Wilcoxon p |
|---|---|---|---|
| dis vs average | +1.069 | 2.96e-19 | 2.29e-16 |
| farneback vs average | +0.845 | 4.30e-11 | 1.75e-12 |
| dis vs flow_oneway_dis | +0.902 | 7.38e-35 | 6.74e-50 |
| dis vs farneback | +0.224 | 1.32e-03 | 1.51e-17 |

Note: triplets from the same video are not fully independent, so p-values are optimistic.
