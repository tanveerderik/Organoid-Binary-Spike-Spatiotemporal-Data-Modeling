# External baseline comparison

Regime `global_full_local`, paired against **4C+soft (ship)**. Medians across clips; q-values are BH-FDR within family.

## Conditioning gain (per-clip error, random -> full context)

| set | random | full | gain | clips better | q |
|---|---|---|---|---|---|
| 4C+soft (ship) | 0.6814 | 0.4175 | +0.1435 | 82/128 | 4.71e-02 |
| 4A+soft | 0.7008 | 0.4098 | +0.1687 | 86/128 | 3.66e-02 |
| 4B | 0.6283 | 0.3959 | +0.1576 | 83/128 | 3.66e-02 |
| 4C+hard | 0.6875 | 0.4048 | +0.1555 | 83/128 | 3.66e-02 |
| DG (Macke'09) | 0.8751 | 0.3907 | +0.3135 | 98/128 | 8.49e-11 |
| GLM (Pillow'08) | 0.6261 | 0.6287 | -0.0329 | 55/128 | 4.71e-02 |
| MaskGIT-flat | 0.7233 | 0.3831 | +0.3817 | 106/128 | 5.42e-13 |

## Per-clip vs_truth

| metric | 4C+soft (ship) | 4A+soft | 4B | 4C+hard | DG (Macke'09) | GLM (Pillow'08) | MaskGIT-flat |
|---|---|---|---|---|---|---|---|
| stat_error_clean | 0.4175 | 0.4098 | 0.3959 | 0.4048 | 0.3907 | 0.6287 | 0.3831 |
| stat_error | 0.8420 | 0.9518 | 1.0549 | 0.8746 | 0.8587 | 0.9245 | 53922.4126 |
| ks_avalanche | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.7500 | 0.7500 | 0.5000 |
| ks_isi | 0.2799 | 0.2805 | 0.2785 | 0.2756 | 0.2020 | 0.1984 | 0.4231 |
| rel_rate | 0.2120 | 0.1861 | 0.2118 | 0.2188 | 0.0568 | 0.2844 | 0.0430 |
| rel_persist4 | 0.6884 | 0.6873 | 0.6097 | 0.7392 | 0.4375 | 0.4220 | 0.4718 |
| rel_isi_mean | 0.2258 | 0.1867 | 0.1982 | 0.2356 | 0.1733 | 0.2185 | 0.4721 |
| rel_avalanche_mean | 0.4919 | 0.3800 | 0.3529 | 0.4441 | 0.7398 | 1.1168 | 0.3140 |
| rel_burst_rate | 0.3333 | 0.3333 | 0.3333 | 0.2857 | 0.3333 | 0.4286 | 0.4286 |

## Pooled vs_real (reported for completeness; blind to conditioning)

| set | stat_error | ks_avalanche | ks_isi |
|---|---|---|---|
| 4C+soft (ship) | 0.7734 | 0.0452 | 0.1790 |
| 4A+soft | 0.7599 | 0.0659 | 0.1848 |
| 4B | 0.7551 | 0.0865 | 0.1768 |
| 4C+hard | 0.7925 | 0.0634 | 0.1846 |
| DG (Macke'09) | 0.2056 | 0.0980 | 0.0807 |
| GLM (Pillow'08) | 0.2692 | 0.3881 | 0.1290 |
| MaskGIT-flat | 0.8098 | 0.1081 | 0.3213 |
| *ceiling (real vs real)* | 0.1060 | 0.0930 | 0.0379 |
| *floor (mean-field)* | 0.9659 | 0.7229 | 0.5254 |
