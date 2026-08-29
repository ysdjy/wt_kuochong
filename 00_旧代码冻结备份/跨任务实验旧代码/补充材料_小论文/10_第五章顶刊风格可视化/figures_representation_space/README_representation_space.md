# Representation space visualization

## Purpose

This folder contains low-dimensional representation-space figures for Chapter 5.

## Main figure

- `Fig5_repr_main_umap`: Raw relative feature representation vs shared latent representation `h_ct` of multi-task TCN-GRU.
- `Fig5_repr_main_pca`: PCA fallback / backup version.
- `Fig5_repr_main_misclassified`: Same as the main PCA layout, but misclassified samples are highlighted by black outlines.

Rows:

1. Raw relative feature representation.
2. Shared latent representation `h_ct` of multi-task TCN-GRU.

Columns:

1. True stage.
2. Relative degradation position `q`.
3. Predictive entropy.

Marker shapes encode condition:

- circle: C1
- triangle: C4
- square: C6

## Supplementary DC-PSR state figure

- `Fig5_repr_dcpsr_final_umap`
- `Fig5_repr_dcpsr_final_pca`

These figures visualize the final probabilistic state of DC-PSR, i.e.,
`[p_E*, p_M*, p_L*, q_hat]`.

## Data status

Proxy mode used: False

If `Proxy mode used` is True, run `extract_hidden_representation.py` first to export real `h_ct`.

## Recommended conclusion

Compared with raw online relative features, the shared latent representation `h_ct`
forms clearer stage separation and a more continuous degradation trajectory.
High-uncertainty or misclassified samples tend to appear near stage transition
regions. The final probabilistic state of DC-PSR remains consistent with the
continuous degradation position, supporting its interpretation as a
degradation-consistent state representation.
