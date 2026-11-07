# Recording-level re-analysis of the paired comparisons

Unit: recording (duplicates merged); n = 28 recordings (31 ids before merging duplicates {'10': 4, '11': 3, '13': 6}). Arm field: `model_mc`.
Multiplicity: BH and Holm over all 88 tests jointly, per level. Bootstrap: recordings within tissue, then clips within recording; equal weight per recording, 10000 draws.

Clip-to-recording mapping verified on 24 dumped clips (model and oracle fields recomputed from the dumps matched one stored clip carrying the predicted recording; 24 decisive matches, 0 mismatches).

`first` = ours (A) or the site map (B) is better; `arm` = the other arm is better; `ns` = q >= 0.05. Δ = first − arm, mean of recording means, with its 95% bootstrap CI.

## A: ours vs arm

| metric | task | arm | first | arm | Δ (95% CI) | rec. won | clip q | clip verdict | rec. q | rec. verdict | changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel AP | recon | MaskGIT-flat | 0.0127 | 0.0130 | +0.0007 (-0.0039, +0.0043) | 71% | 3.5e-04 | first | 5.9e-02 | ns | **yes** |
| voxel AP | recon | 3D U-Net (det.) | 0.0127 | 0.0269 | -0.0117 (-0.0181, -0.0064) | 18% | 1.1e-32 | arm | 8.4e-06 | arm |  |
| voxel AP | recon | 3D U-Net seed 2 | 0.0127 | 0.0322 | -0.0176 (-0.0244, -0.0117) | 7% | 8.7e-42 | arm | 8.0e-08 | arm |  |
| voxel AP | recon | 3D CVAE | 0.0127 | 0.0250 | -0.0114 (-0.0204, -0.0047) | 29% | 6.2e-20 | arm | 7.7e-04 | arm |  |
| voxel AP | causal | MaskGIT-flat | 0.0137 | 0.0151 | -0.0004 (-0.0047, +0.0041) | 46% | 3.1e-01 | ns | 8.8e-01 | ns |  |
| voxel AP | causal | 3D U-Net (det.) | 0.0137 | 0.0359 | -0.0194 (-0.0287, -0.0113) | 11% | 2.9e-34 | arm | 1.2e-06 | arm |  |
| voxel AP | causal | 3D U-Net seed 2 | 0.0137 | 0.0339 | -0.0183 (-0.0259, -0.0119) | 11% | 1.1e-37 | arm | 2.8e-07 | arm |  |
| voxel AP | causal | 3D CVAE | 0.0137 | 0.0252 | -0.0102 (-0.0199, -0.0024) | 29% | 9.9e-17 | arm | 8.1e-03 | arm |  |
| voxel AP | noncausal | MaskGIT-flat | 0.0154 | 0.0162 | -0.0001 (-0.0043, +0.0037) | 50% | 6.0e-01 | ns | 7.5e-01 | ns |  |
| voxel AP | noncausal | 3D U-Net (det.) | 0.0154 | 0.0367 | -0.0187 (-0.0273, -0.0112) | 11% | 1.2e-29 | arm | 1.2e-05 | arm |  |
| voxel AP | noncausal | 3D U-Net seed 2 | 0.0154 | 0.0355 | -0.0181 (-0.0253, -0.0119) | 7% | 1.8e-32 | arm | 4.6e-07 | arm |  |
| voxel AP | noncausal | 3D CVAE | 0.0154 | 0.0257 | -0.0090 (-0.0179, -0.0018) | 39% | 4.8e-11 | arm | 4.5e-02 | arm |  |
| voxel AP | spatial | MaskGIT-flat | 0.0200 | 0.0119 | +0.0089 (+0.0031, +0.0166) | 81% | 6.8e-13 | first | 2.8e-04 | first |  |
| voxel AP | spatial | 3D U-Net (det.) | 0.0200 | 0.0255 | -0.0034 (-0.0114, +0.0057) | 33% | 3.2e-12 | arm | 1.1e-01 | ns | **yes** |
| voxel AP | spatial | 3D U-Net seed 2 | 0.0200 | 0.0253 | -0.0049 (-0.0109, +0.0035) | 19% | 1.3e-15 | arm | 9.9e-03 | arm |  |
| voxel AP | spatial | 3D CVAE | 0.0200 | 0.0209 | -0.0009 (-0.0085, +0.0082) | 59% | 2.4e-03 | arm | 9.2e-01 | ns | **yes** |
| site AP | recon | MaskGIT-flat | 0.2244 | 0.1481 | +0.0758 (+0.0340, +0.1185) | 75% | 3.7e-17 | first | 6.2e-03 | first |  |
| site AP | recon | 3D U-Net (det.) | 0.2244 | 0.3412 | -0.1019 (-0.1458, -0.0571) | 18% | 1.8e-27 | arm | 2.7e-04 | arm |  |
| site AP | recon | 3D U-Net seed 2 | 0.2244 | 0.4045 | -0.1690 (-0.2085, -0.1276) | 4% | 1.1e-42 | arm | 2.8e-07 | arm |  |
| site AP | recon | 3D CVAE | 0.2244 | 0.2709 | -0.0427 (-0.0952, +0.0106) | 39% | 1.5e-06 | arm | 1.4e-01 | ns | **yes** |
| site AP | causal | MaskGIT-flat | 0.2170 | 0.1472 | +0.0684 (+0.0325, +0.1059) | 75% | 8.8e-15 | first | 6.6e-03 | first |  |
| site AP | causal | 3D U-Net (det.) | 0.2170 | 0.3297 | -0.1014 (-0.1424, -0.0590) | 14% | 4.5e-26 | arm | 2.9e-04 | arm |  |
| site AP | causal | 3D U-Net seed 2 | 0.2170 | 0.3414 | -0.1184 (-0.1546, -0.0834) | 4% | 1.7e-33 | arm | 5.9e-07 | arm |  |
| site AP | causal | 3D CVAE | 0.2170 | 0.2368 | -0.0159 (-0.0638, +0.0314) | 46% | 2.4e-02 | arm | 6.0e-01 | ns | **yes** |
| site AP | noncausal | MaskGIT-flat | 0.1889 | 0.1387 | +0.0504 (+0.0205, +0.0806) | 71% | 5.6e-10 | first | 1.3e-02 | first |  |
| site AP | noncausal | 3D U-Net (det.) | 0.1889 | 0.3057 | -0.1052 (-0.1422, -0.0682) | 11% | 4.1e-30 | arm | 2.9e-05 | arm |  |
| site AP | noncausal | 3D U-Net seed 2 | 0.1889 | 0.3079 | -0.1109 (-0.1417, -0.0794) | 7% | 9.4e-35 | arm | 4.6e-07 | arm |  |
| site AP | noncausal | 3D CVAE | 0.1889 | 0.2127 | -0.0191 (-0.0619, +0.0229) | 50% | 1.7e-03 | arm | 5.4e-01 | ns | **yes** |
| site AP | spatial | MaskGIT-flat | 0.3092 | 0.1174 | +0.1994 (+0.1495, +0.2562) | 96% | 3.1e-35 | first | 6.6e-08 | first |  |
| site AP | spatial | 3D U-Net (det.) | 0.3092 | 0.3342 | -0.0074 (-0.0602, +0.0508) | 52% | 3.2e-05 | arm | 7.1e-01 | ns | **yes** |
| site AP | spatial | 3D U-Net seed 2 | 0.3092 | 0.3475 | -0.0301 (-0.0748, +0.0199) | 30% | 1.0e-08 | arm | 8.9e-02 | ns | **yes** |
| site AP | spatial | 3D CVAE | 0.3092 | 0.2417 | +0.0770 (+0.0150, +0.1420) | 74% | 5.1e-03 | first | 1.9e-02 | first |  |

## B: site map vs arm

| metric | task | arm | first | arm | Δ (95% CI) | rec. won | clip q | clip verdict | rec. q | rec. verdict | changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel AP | recon | Ours (4C+soft) | 0.0670 | 0.0127 | +0.0514 (+0.0403, +0.0641) | 100% | 4.3e-46 | first | 2.2e-08 | first |  |
| voxel AP | recon | MaskGIT-flat | 0.0670 | 0.0130 | +0.0521 (+0.0429, +0.0619) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| voxel AP | recon | 3D U-Net (det.) | 0.0670 | 0.0269 | +0.0397 (+0.0312, +0.0498) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| voxel AP | recon | 3D U-Net seed 2 | 0.0670 | 0.0322 | +0.0338 (+0.0268, +0.0417) | 100% | 8.1e-44 | first | 2.2e-08 | first |  |
| voxel AP | recon | 3D CVAE | 0.0670 | 0.0250 | +0.0400 (+0.0331, +0.0471) | 100% | 1.4e-45 | first | 2.2e-08 | first |  |
| voxel AP | recon | Dich. Gaussian (ref) | 0.0670 | 0.0545 | +0.0120 (+0.0091, +0.0148) | 100% | 2.6e-36 | first | 2.2e-08 | first |  |
| voxel AP | recon | Coupled GLM (ref) | 0.0670 | 0.0688 | -0.0017 (-0.0035, +0.0001) | 36% | 5.0e-04 | arm | 4.5e-02 | arm |  |
| voxel AP | causal | Ours (4C+soft) | 0.0695 | 0.0137 | +0.0523 (+0.0391, +0.0674) | 100% | 1.3e-43 | first | 2.2e-08 | first |  |
| voxel AP | causal | MaskGIT-flat | 0.0695 | 0.0151 | +0.0519 (+0.0409, +0.0642) | 100% | 1.5e-45 | first | 2.2e-08 | first |  |
| voxel AP | causal | 3D U-Net (det.) | 0.0695 | 0.0359 | +0.0329 (+0.0257, +0.0405) | 100% | 1.3e-44 | first | 2.2e-08 | first |  |
| voxel AP | causal | 3D U-Net seed 2 | 0.0695 | 0.0339 | +0.0339 (+0.0259, +0.0427) | 100% | 5.7e-43 | first | 2.2e-08 | first |  |
| voxel AP | causal | 3D CVAE | 0.0695 | 0.0252 | +0.0421 (+0.0339, +0.0508) | 100% | 1.9e-45 | first | 2.2e-08 | first |  |
| voxel AP | causal | Dich. Gaussian (ref) | 0.0695 | 0.0574 | +0.0111 (+0.0070, +0.0152) | 86% | 8.7e-26 | first | 1.3e-05 | first |  |
| voxel AP | causal | Coupled GLM (ref) | 0.0695 | 0.0737 | -0.0039 (-0.0067, -0.0011) | 32% | 4.1e-05 | arm | 3.8e-03 | arm |  |
| voxel AP | noncausal | Ours (4C+soft) | 0.0719 | 0.0154 | +0.0532 (+0.0417, +0.0661) | 100% | 3.6e-44 | first | 2.2e-08 | first |  |
| voxel AP | noncausal | MaskGIT-flat | 0.0719 | 0.0162 | +0.0532 (+0.0433, +0.0639) | 100% | 1.1e-44 | first | 2.2e-08 | first |  |
| voxel AP | noncausal | 3D U-Net (det.) | 0.0719 | 0.0367 | +0.0345 (+0.0272, +0.0419) | 100% | 2.2e-40 | first | 2.2e-08 | first |  |
| voxel AP | noncausal | 3D U-Net seed 2 | 0.0719 | 0.0355 | +0.0351 (+0.0272, +0.0433) | 96% | 5.1e-40 | first | 5.3e-08 | first |  |
| voxel AP | noncausal | 3D CVAE | 0.0719 | 0.0257 | +0.0443 (+0.0368, +0.0522) | 100% | 9.2e-45 | first | 2.2e-08 | first |  |
| voxel AP | noncausal | Dich. Gaussian (ref) | 0.0719 | 0.0611 | +0.0098 (+0.0050, +0.0140) | 86% | 6.7e-22 | first | 1.2e-04 | first |  |
| voxel AP | noncausal | Coupled GLM (ref) | 0.0719 | 0.0777 | -0.0052 (-0.0081, -0.0025) | 21% | 7.2e-08 | arm | 1.7e-04 | arm |  |
| voxel AP | spatial | Ours (4C+soft) | 0.0835 | 0.0200 | +0.0575 (+0.0410, +0.0764) | 96% | 2.4e-37 | first | 9.1e-08 | first |  |
| voxel AP | spatial | MaskGIT-flat | 0.0835 | 0.0119 | +0.0665 (+0.0520, +0.0849) | 100% | 6.6e-40 | first | 3.6e-08 | first |  |
| voxel AP | spatial | 3D U-Net (det.) | 0.0835 | 0.0255 | +0.0541 (+0.0409, +0.0717) | 96% | 2.6e-38 | first | 6.6e-08 | first |  |
| voxel AP | spatial | 3D U-Net seed 2 | 0.0835 | 0.0253 | +0.0527 (+0.0391, +0.0706) | 100% | 1.4e-39 | first | 3.6e-08 | first |  |
| voxel AP | spatial | 3D CVAE | 0.0835 | 0.0209 | +0.0566 (+0.0436, +0.0741) | 100% | 1.2e-40 | first | 3.6e-08 | first |  |
| voxel AP | spatial | Dich. Gaussian (ref) | 0.0835 | 0.0682 | +0.0142 (+0.0015, +0.0297) | 85% | 5.4e-20 | first | 6.4e-04 | first |  |
| voxel AP | spatial | Coupled GLM (ref) | 0.0835 | 0.0806 | +0.0034 (-0.0046, +0.0166) | 52% | 4.3e-01 | ns | 8.2e-01 | ns |  |
| site AP | recon | Ours (4C+soft) | 0.6796 | 0.2244 | +0.4513 (+0.4019, +0.4969) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | recon | MaskGIT-flat | 0.6796 | 0.1481 | +0.5271 (+0.4920, +0.5599) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | recon | 3D U-Net (det.) | 0.6796 | 0.3412 | +0.3494 (+0.2990, +0.3987) | 100% | 3.6e-46 | first | 2.2e-08 | first |  |
| site AP | recon | 3D U-Net seed 2 | 0.6796 | 0.4045 | +0.2823 (+0.2336, +0.3298) | 100% | 1.5e-45 | first | 2.2e-08 | first |  |
| site AP | recon | 3D CVAE | 0.6796 | 0.2709 | +0.4087 (+0.3734, +0.4441) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | recon | Dich. Gaussian (ref) | 0.6796 | 0.6997 | -0.0189 (-0.0341, -0.0042) | 32% | 2.8e-09 | arm | 1.2e-02 | arm |  |
| site AP | recon | Coupled GLM (ref) | 0.6796 | 0.6984 | -0.0176 (-0.0324, -0.0031) | 18% | 2.0e-08 | arm | 8.1e-03 | arm |  |
| site AP | causal | Ours (4C+soft) | 0.5714 | 0.2170 | +0.3517 (+0.3020, +0.3996) | 100% | 3.2e-45 | first | 2.2e-08 | first |  |
| site AP | causal | MaskGIT-flat | 0.5714 | 0.1472 | +0.4201 (+0.3822, +0.4558) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | causal | 3D U-Net (det.) | 0.5714 | 0.3297 | +0.2503 (+0.2102, +0.2915) | 100% | 2.0e-45 | first | 2.2e-08 | first |  |
| site AP | causal | 3D U-Net seed 2 | 0.5714 | 0.3414 | +0.2333 (+0.1922, +0.2728) | 100% | 1.2e-44 | first | 2.2e-08 | first |  |
| site AP | causal | 3D CVAE | 0.5714 | 0.2368 | +0.3358 (+0.2997, +0.3735) | 100% | 6.3e-46 | first | 2.2e-08 | first |  |
| site AP | causal | Dich. Gaussian (ref) | 0.5714 | 0.5875 | -0.0148 (-0.0290, -0.0011) | 29% | 1.5e-06 | arm | 3.9e-02 | arm |  |
| site AP | causal | Coupled GLM (ref) | 0.5714 | 0.5938 | -0.0208 (-0.0367, -0.0052) | 21% | 1.9e-08 | arm | 4.1e-03 | arm |  |
| site AP | noncausal | Ours (4C+soft) | 0.5184 | 0.1889 | +0.3263 (+0.2863, +0.3640) | 100% | 2.9e-45 | first | 2.2e-08 | first |  |
| site AP | noncausal | MaskGIT-flat | 0.5184 | 0.1387 | +0.3766 (+0.3405, +0.4110) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | noncausal | 3D U-Net (det.) | 0.5184 | 0.3057 | +0.2210 (+0.1845, +0.2575) | 100% | 7.3e-45 | first | 2.2e-08 | first |  |
| site AP | noncausal | 3D U-Net seed 2 | 0.5184 | 0.3079 | +0.2153 (+0.1781, +0.2509) | 100% | 1.7e-43 | first | 2.2e-08 | first |  |
| site AP | noncausal | 3D CVAE | 0.5184 | 0.2127 | +0.3072 (+0.2735, +0.3408) | 100% | 3.0e-46 | first | 2.2e-08 | first |  |
| site AP | noncausal | Dich. Gaussian (ref) | 0.5184 | 0.5302 | -0.0104 (-0.0233, +0.0016) | 43% | 1.0e-03 | arm | 1.5e-01 | ns | **yes** |
| site AP | noncausal | Coupled GLM (ref) | 0.5184 | 0.5375 | -0.0169 (-0.0314, -0.0032) | 21% | 3.7e-07 | arm | 9.9e-03 | arm |  |
| site AP | spatial | Ours (4C+soft) | 0.7224 | 0.3092 | +0.3929 (+0.3283, +0.4490) | 96% | 1.7e-35 | first | 9.1e-08 | first |  |
| site AP | spatial | MaskGIT-flat | 0.7224 | 0.1174 | +0.5924 (+0.5446, +0.6393) | 100% | 4.2e-41 | first | 3.6e-08 | first |  |
| site AP | spatial | 3D U-Net (det.) | 0.7224 | 0.3342 | +0.3856 (+0.3248, +0.4451) | 96% | 2.4e-39 | first | 6.6e-08 | first |  |
| site AP | spatial | 3D U-Net seed 2 | 0.7224 | 0.3475 | +0.3629 (+0.3137, +0.4104) | 100% | 1.6e-39 | first | 3.6e-08 | first |  |
| site AP | spatial | 3D CVAE | 0.7224 | 0.2417 | +0.4700 (+0.4225, +0.5199) | 100% | 4.9e-41 | first | 3.6e-08 | first |  |
| site AP | spatial | Dich. Gaussian (ref) | 0.7224 | 0.7392 | -0.0106 (-0.0291, +0.0074) | 33% | 8.4e-05 | arm | 1.5e-01 | ns | **yes** |
| site AP | spatial | Coupled GLM (ref) | 0.7224 | 0.7402 | -0.0103 (-0.0409, +0.0183) | 26% | 3.7e-04 | arm | 1.8e-01 | ns | **yes** |

## Tally

| comparison | level | first | arm | ns |
|---|---|---|---|---|
| A: ours vs arm | clip | 7 | 23 | 2 |
| A: ours vs arm | recording | 6 | 16 | 10 |
| B: site map vs arm | clip | 44 | 11 | 1 |
| B: site map vs arm | recording | 44 | 8 | 4 |
