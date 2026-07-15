# Anorexia Nervosa — Dynamic Functional Connectivity Analysis

fMRI resting-state analysis pipeline investigating dynamic functional connectivity (DFC) differences between anorexia nervosa (AN) patients and healthy controls (HC).

## Goal

Use Multi-Layer Community Detection (MLCD) on sliding-window Pearson correlation matrices to characterise how brain network organisation changes over time, and whether those changes differ between AN and HC.

## Atlas

- **Cortical**: Schaefer-200 (Yeo-7 networks, MNI152 1mm)
- **Subcortical**: Tian Scale I (16 bilateral ROIs, MNI152 2009cAsym)
- **Combined**: 216 regions (200 cortical + 16 subcortical)

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 0 | `00_atlas_overview.py` | Atlas visualisation figure |
| 0b | `00b_edge_density.py` | Edge density threshold comparison (5% vs 30%) |
| 1 | `01_cortical_fc.py` | Cortical sliding-window FC → .mat for MLCD |
| 1b | `01b_subcortical_fc.py` | Subcortical sliding-window FC |
| 1c | `01c_combined_fc.py` | Combined 216-region FC |
| 1d | `01d_static_fc.py` | Static FC and static vs DFC figures |
| 2 | `02_outcome_measures.py` | Allegiance, RC, IC, flexibility, promiscuity (cortical) |
| 2b | `02b_outcome_subcortical.py` | Same measures for subcortical |
| 2c | `02c_outcome_combined.py` | Combined 216-region outcome measures |
| 3 | `03_statistical_analysis.py` | Permutation tests, FDR correction |
| 3b | `03b_statistical_subcortical.py` | Subcortical statistical analysis |
| 4 | `04_visualization.py` | Publication figures |
| 4b | `04b_visualization_subcortical.py` | Subcortical figures |

MATLAB scripts for MLCD are in `code/matlab/`.

## Parameters

- TR: 0.8 s
- Window: 30 s (38 TRs), step 1 TR → ~663 windows per subject
- Edge density: 5% (combined), 30% (standalone subcortical)
- Pilot cohort: 5 AN + 5 HC (full cohort: 22 AN + 22 HC)

## References

- Schaefer et al. (2018). *Cerebral Cortex*
- Tian et al. (2020). *Nature Neuroscience*
- Yeo et al. (2011). *Journal of Neurophysiology*
