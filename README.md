# Anorexia Nervosa (AN) Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-0076A8?logo=mathworks&logoColor=white&style=flat-square)
![Conda](https://img.shields.io/badge/conda-fmri-44A833?logo=anaconda&logoColor=white&style=flat-square)
![nilearn](https://img.shields.io/badge/nilearn-0.10%2B-4B8BBE?style=flat-square)
![Modality](https://img.shields.io/badge/modality-resting--state%20fMRI-7B68EE?style=flat-square)
![Method](https://img.shields.io/badge/method-MLCD%20·%20GenLouvain-5C4DBE?style=flat-square)
![Atlas](https://img.shields.io/badge/atlas-Schaefer--200%20·%20Tian%20S2-4682B4?style=flat-square)
![Cohort](https://img.shields.io/badge/cohort-22%20AN%20·%2022%20HC%20(N%3D44)-E8622A?style=flat-square)
![Windows](https://img.shields.io/badge/windows-~663%20per%20subject-557A95?style=flat-square)
![Status](https://img.shields.io/badge/status-in%20progress-F5A623?style=flat-square)

Resting-state fMRI pipeline comparing functional connectivity between anorexia nervosa (AN) patients and healthy controls (HC) across both static and dynamic analyses.

## Goal

Quantify functional connectivity differences between AN and HC at ROIs of interest using two complementary approaches:

- **Static FC**: a single full-scan Pearson correlation matrix per subject, providing a mean connectivity baseline across the session.
- **Dynamic FC (DFC)**: sliding-window correlation matrices submitted to Multi-Layer Community Detection (MLCD), capturing how network community structure reorganises over time.

Comparing static and dynamic results at the same ROIs distinguishes group differences that reflect stable mean connectivity from those that emerge in temporal network dynamics, or both.

## Atlas

- **Cortical**: Schaefer-200 (Yeo-7 networks, MNI152 1mm)
- **Subcortical**: Tian Scale I (16 bilateral ROIs, MNI152 2009cAsym)
- **Combined**: 216 regions (200 cortical + 16 subcortical)

## Pipeline

<details>
<summary>Show pipeline steps</summary>

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

</details>

## Parameters

- TR: 0.8 s
- Window: 30 s (38 TRs), step 1 TR → ~663 windows per subject
- Edge density: 5% (combined), 30% (standalone subcortical)
- Pilot cohort: 5 AN + 5 HC (full cohort: 22 AN + 22 HC)

## References

- Schaefer et al. (2018). *Cerebral Cortex*
- Tian et al. (2020). *Nature Neuroscience*
- Yeo et al. (2011). *Journal of Neurophysiology*
