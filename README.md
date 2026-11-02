# Dynamic multilayer network analysis of anorexia nervosa

Analysis code for a resting-state fMRI study comparing individuals with anorexia
nervosa (AN) and healthy controls (HC). The brain is divided into 216 regions:
200 cortical regions from the Schaefer atlas, grouped into seven functional
networks, and 16 subcortical regions from the Tian atlas (for example thalamus,
amygdala, hippocampus and striatum), each assigned to one of those networks.
Sliding-window functional connectivity between these regions is analysed with
multilayer community detection (MLCD), and groups are compared on recruitment,
integration, flexibility and promiscuity.

No data are included in this repository.

## Methods in brief

- Groups: cortical and subcortical regions of the same network are kept apart,
  giving 13 functional groups (7 cortical, 6 subcortical).
- Windows: 38 TR (30 s, TR = 0.8 s), 1-TR step, 663 windows per participant;
  each window keeps the strongest 5% of positive edges.
- MLCD: GenLouvain with ordinal coupling, gamma = omega = 1, 100 runs with
  iterative consensus per window; labels matched across windows by largest overlap.
- Statistics: subject-label permutation (20,000), Cohen's d, Benjamini-Hochberg
  FDR, Freedman-Lane adjustment for mean framewise displacement, PERMANOVA and
  PERMDISP.

## Pipeline

Run in this order. Python scripts are in `code/python/`, MATLAB scripts in `code/matlab/`.

| Step | Script | Purpose |
|---|---|---|
| 1 | `01c_combined_fc.py` | Regional time series and sliding-window FC for the 216-region atlas |
| 2 | `01e_subcortical_network_assignment.py` | Assign each Tian S1 region to a Yeo network |
| 3 | `00b_edge_density.py` | Compare 5% and 30% edge density |
| 4 | `gridsearch_multilayer_modularity.m` | Grid search over gamma and omega |
| 5 | `combined_mlcd_full44_d05_par.m` | MLCD for every participant (calls `multilayer_community_detection_individual.m`) |
| 6 | `05b_dfc_group_statistics.py` | Outcome measures and group statistics |
| 7 | `17_apriori_region_test.py` | Tests on 38 AN-related regions, with motion adjustment (uses `14_individual_vs_group.py`) |
| 8 | `18_censoring_sensitivity.py` | Sensitivity of the results to censored volumes |
| 9 | `19_label_matching_rasters.py` | Check of community-label matching |
| 10 | `09b_fig1_panels.py`, `09_abstract_figures.py` | Figures |

Shared modules: `paths.py` (project root), `atlas_groups.py` (functional groups),
`mlcd_measures.py` (outcome measures).

## Figures

`figures/` holds the three main figures:

- `fig1_pipeline.png`: analysis pipeline, from windowed connectivity to the outcome measures.
- `fig2_dynamic_fc.png`: whole-brain effect sizes for the four measures and the network-block allegiance difference (AN minus HC).
- `fig3_significant_regions.png`: group differences in the 38 AN-related regions.

Figures 2 and 3 are produced by `09_abstract_figures.py`; the data panels of Figure 1 by `09b_fig1_panels.py`.

## Requirements

- Python 3.9 or later with numpy, pandas, scipy, statsmodels, nibabel, nilearn,
  matplotlib, h5py and teneto.
- MATLAB with the Parallel Computing Toolbox, plus
  [GenLouvain](https://github.com/GenLouvain/GenLouvain) and
  [cocoanCORE](https://github.com/cocoanlab/cocoanCORE) in the MATLAB user folder.

## Data layout

Set `ANOREXIA_ROOT` to the folder that holds the data; by default the scripts use
the folder that contains this repository.

```
<ANOREXIA_ROOT>/
  data/an_patients/, data/hc_patients/   AFNI residual time series (errts*.nii.gz)
  data/atlas/                            Schaefer-200, Tian S1, combined labels,
                                         subcortical network maps
  data/analysis/                         FC and MLCD outputs (written by the pipeline)
  output/results/                        statistics (written by the pipeline)
```

Preprocessing (AFNI afni_proc.py: despiking, motion correction, nonlinear
normalization, 6-mm smoothing, nuisance regression with fast ANATICOR) is not
part of this repository.

## References

- Schaefer A, et al. Cereb Cortex. 2018;28:3095-3114.
- Tian Y, et al. Nat Neurosci. 2020;23:1421-1432.
- Mucha PJ, et al. Science. 2010;328:876-878.
- Bassett DS, et al. Proc Natl Acad Sci USA. 2011;108:7641-7646.
- Papadopoulos L, et al. Phys Rev E. 2016;94:032908.
