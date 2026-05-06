# figW batch analysis

This file summarizes the batch experiment for the `3x3` full-flow figures.

## Scan setup

- Output folder: `exp/analysitic/figW`
- Number of cases: `15`
- parameter pairs: `[(20.0, 380.0), (20.0, 200.0), (20.0, 0.3), (20.0, 2.0), (5.0, 50.0), (0.5, 5.0), (0.5, 0.1), (0.1, 0.0001), (0.6, 0.45), (0.1, 0.05), (0.1, 0.9), (1.0, 0.3), (1.0, 2.6), (3.0, 1.0), (0.3, 1.0)]`
- The listed parameter pairs are evaluated directly.
- For large parameter values, the script uses fewer time steps, so the plots stay readable.
- When a curve spans many orders of magnitude, the plot uses a symlog axis.

## Main rough findings

1. When `lam` and `mu` are both small, the phase portrait and lifted trajectories stay close to the origin, and the singular values are also small.
2. When `lam` becomes much larger than `mu`, the off-diagonal term `lam^2 - mu` becomes very large, and the lifted observable panel changes much more strongly.
3. When `mu` becomes much larger than `lam`, the `y` direction becomes dominant, and the phase portrait changes shape in a different way from the `lam`-dominant cases.
4. When `lam` and `mu` are on the same scale, the figures usually look more balanced, and the left/right singular-vector matrices tend to change more smoothly across neighboring coarse scales.
5. Very large values such as `1e4` do not just rescale the plots. They also change the relative structure of `A_o`, so the singular-value spectrum and the singular-vector heatmaps can look qualitatively different.

## Cases with the largest difference between U and V

| lam | mu | fro_gap_U_V | sigma_1 | sigma_2 | sigma_3 | file_name |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.3 | 1.08141 | 565.473 | 20 | 0.212212 | lam_20__mu_0.3.png |
| 20 | 2 | 1.07586 | 564.275 | 20 | 1.41775 | lam_20__mu_2.png |
| 0.1 | 0.0001 | 1.06931 | 0.1 | 0.0140718 | 7.10642e-05 | lam_0.1__mu_0.0001.png |
| 0.1 | 0.9 | 1.06786 | 1.26576 | 0.1 | 0.00711035 | lam_0.1__mu_0.9.png |
| 0.5 | 5 | 1.01679 | 6.89871 | 0.5 | 0.181193 | lam_0.5__mu_5.png |
| 0.3 | 1 | 0.964067 | 1.35343 | 0.3 | 0.0664976 | lam_0.3__mu_1.png |
| 3 | 1 | 0.936229 | 12.06 | 3 | 0.74627 | lam_3__mu_1.png |
| 0.1 | 0.05 | 0.819633 | 0.1 | 0.0643398 | 0.00777124 | lam_0.1__mu_0.05.png |

## Cases with the largest first singular value

| lam | mu | sigma_1 | sigma_2 | sigma_3 | file_name |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.3 | 565.473 | 20 | 0.212212 | lam_20__mu_0.3.png |
| 20 | 2 | 564.275 | 20 | 1.41775 | lam_20__mu_2.png |
| 20 | 200 | 457.649 | 174.806 | 20 | lam_20__mu_200.png |
| 20 | 380 | 404.27 | 375.986 | 20 | lam_20__mu_380.png |
| 5 | 50 | 57.2061 | 21.8508 | 5 | lam_5__mu_50.png |
| 3 | 1 | 12.06 | 3 | 0.74627 | lam_3__mu_1.png |
| 0.5 | 5 | 6.89871 | 0.5 | 0.181193 | lam_0.5__mu_5.png |
| 1 | 2.6 | 3.10114 | 1 | 0.838401 | lam_1__mu_2.6.png |

## Cases with the largest eigenvector non-orthogonality

| lam | mu | fro_norm_QTQ_minus_I | eig_1 | eig_2 | eig_3 | file_name |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.0001 | 1 | 0.1 | 0.01 | 0.0001 | lam_0.1__mu_0.0001.png |
| 0.1 | 0.05 | 1 | 0.1 | 0.05 | 0.01 | lam_0.1__mu_0.05.png |
| 0.1 | 0.9 | 1 | 0.9 | 0.1 | 0.01 | lam_0.1__mu_0.9.png |
| 0.3 | 1 | 1 | 1 | 0.3 | 0.09 | lam_0.3__mu_1.png |
| 0.5 | 0.1 | 1 | 0.5 | 0.25 | 0.1 | lam_0.5__mu_0.1.png |
| 0.5 | 5 | 1 | 5 | 0.5 | 0.25 | lam_0.5__mu_5.png |
| 0.6 | 0.45 | 1 | 0.6 | 0.45 | 0.36 | lam_0.6__mu_0.45.png |
| 1 | 0.3 | 1 | 1 | 1 | 0.3 | lam_1__mu_0.3.png |

## Cases with the smallest observable amplitude

| lam | mu | obs_max_abs | phase_max_abs | file_name |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0001 | 0.8 | 0.941451 | lam_0.1__mu_0.0001.png |
| 0.1 | 0.05 | 0.8 | 0.941451 | lam_0.1__mu_0.05.png |
| 0.5 | 0.1 | 0.8 | 0.941451 | lam_0.5__mu_0.1.png |
| 0.6 | 0.45 | 0.8 | 0.941451 | lam_0.6__mu_0.45.png |
| 1 | 0.3 | 0.8 | 0.941451 | lam_1__mu_0.3.png |
| 0.1 | 0.9 | 0.8846 | 1.58038 | lam_0.1__mu_0.9.png |
| 0.3 | 1 | 0.99 | 1.7658 | lam_0.3__mu_1.png |
| 1 | 2.6 | 2.91847e+07 | 24926.4 | lam_1__mu_2.6.png |

## Same-scale cases: lam = mu

| lam | mu | sigma_1 | sigma_2 | sigma_3 | fro_gap_U_V | file_name |
| --- | --- | --- | --- | --- | --- | --- |

## lam-dominant cases with large U-V gap

| lam | mu | fro_gap_U_V | sigma_1 | sigma_2 | sigma_3 | file_name |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.3 | 1.08141 | 565.473 | 20 | 0.212212 | lam_20__mu_0.3.png |
| 20 | 2 | 1.07586 | 564.275 | 20 | 1.41775 | lam_20__mu_2.png |
| 0.1 | 0.0001 | 1.06931 | 0.1 | 0.0140718 | 7.10642e-05 | lam_0.1__mu_0.0001.png |
| 3 | 1 | 0.936229 | 12.06 | 3 | 0.74627 | lam_3__mu_1.png |
| 0.1 | 0.05 | 0.819633 | 0.1 | 0.0643398 | 0.00777124 | lam_0.1__mu_0.05.png |
| 1 | 0.3 | 0.691459 | 1.23322 | 1 | 0.243266 | lam_1__mu_0.3.png |
| 0.5 | 0.1 | 0.5687 | 0.5 | 0.29646 | 0.0843283 | lam_0.5__mu_0.1.png |
| 0.6 | 0.45 | 0.156413 | 0.6 | 0.471132 | 0.343853 | lam_0.6__mu_0.45.png |

## mu-dominant cases with large U-V gap

| lam | mu | fro_gap_U_V | sigma_1 | sigma_2 | sigma_3 | file_name |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.9 | 1.06786 | 1.26576 | 0.1 | 0.00711035 | lam_0.1__mu_0.9.png |
| 0.5 | 5 | 1.01679 | 6.89871 | 0.5 | 0.181193 | lam_0.5__mu_5.png |
| 0.3 | 1 | 0.964067 | 1.35343 | 0.3 | 0.0664976 | lam_0.3__mu_1.png |
| 1 | 2.6 | 0.587157 | 3.10114 | 1 | 0.838401 | lam_1__mu_2.6.png |
| 5 | 50 | 0.453064 | 57.2061 | 21.8508 | 5 | lam_5__mu_50.png |
| 20 | 200 | 0.453064 | 457.649 | 174.806 | 20 | lam_20__mu_200.png |
| 20 | 380 | 0.0362529 | 404.27 | 375.986 | 20 | lam_20__mu_380.png |
