# External baseline comparison

Regime `global_full_local`, paired against **4C+soft (ship)**. Medians across clips; q-values are BH-FDR within family.

## Conditioning gain (per-clip error, random -> full context)

| set | random | full | gain | clips better | q |
|---|---|---|---|---|---|
| 4C+soft (ship) | 1.2257 | 0.8420 | +0.2590 | 87/128 | 1.84e-02 |
| 4A+soft | 1.3897 | 0.9518 | +0.2513 | 82/128 | 3.33e-02 |
| 4B | 1.2922 | 1.0549 | +0.2679 | 82/128 | 1.84e-02 |
| 4C+hard | 1.4256 | 0.8746 | +0.2569 | 80/128 | 5.41e-02 |
| DG (Macke'09) | 0.9493 | 0.8636 | +0.2042 | 82/128 | 3.33e-02 |

## Per-clip vs_truth

| metric | 4C+soft (ship) | 4A+soft | 4B | 4C+hard | DG (Macke'09) |
|---|---|---|---|---|---|
| stat_error | 0.8420 | 0.9518 | 1.0549 | 0.8746 | 0.8636 |
| ks_avalanche | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.8333 |
| ks_isi | 0.2799 | 0.2805 | 0.2785 | 0.2756 | 0.2033 |
| rel_rate | 0.2120 | 0.1861 | 0.2118 | 0.2188 | 0.0075 |
| rel_persist4 | 0.6884 | 0.6873 | 0.6097 | 0.7392 | 0.4325 |
| rel_isi_mean | 0.2258 | 0.1867 | 0.1982 | 0.2356 | 0.1596 |
| rel_avalanche_mean | 0.4919 | 0.3800 | 0.3529 | 0.4441 | 0.9799 |
| rel_burst_rate | 0.3333 | 0.3333 | 0.3333 | 0.2857 | 0.3333 |

## Pooled vs_real (reported for completeness; blind to conditioning)

| set | stat_error | ks_avalanche | ks_isi |
|---|---|---|---|
| 4C+soft (ship) | 0.7734 | 0.0452 | 0.1790 |
| 4A+soft | 0.7599 | 0.0659 | 0.1848 |
| 4B | 0.7551 | 0.0865 | 0.1768 |
| 4C+hard | 0.7925 | 0.0634 | 0.1846 |
| DG (Macke'09) | 0.2108 | 0.1022 | 0.0788 |
| *ceiling (real vs real)* | 0.1060 | 0.0930 | 0.0379 |
| *floor (mean-field)* | 0.9659 | 0.7229 | 0.5254 |
