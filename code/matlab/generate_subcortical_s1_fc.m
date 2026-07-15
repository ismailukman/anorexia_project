% generate_subcortical_s1_fc.m
%
% Extract the Tian Scale I (16-ROI) subcortical sliding-window FC from
% the already-computed combined 216-region FC matrices.
%
% The combined atlas ordering is:
%   Rows/cols 1–200   : Schaefer-200 cortical (Yeo-7)
%   Rows/cols 201–216 : Tian Scale I subcortical (16 bilateral ROIs)
%                       201 HIP-rh  202 AMY-rh  203 pTHA-rh  204 aTHA-rh
%                       205 NAc-rh  206 GP-rh   207 PUT-rh   208 CAU-rh
%                       209 HIP-lh  210 AMY-lh  211 pTHA-lh  212 aTHA-lh
%                       213 NAc-lh  214 GP-lh   215 PUT-lh   216 CAU-lh
%
% Input :  data/analysis/combined_subjs/subj_fc_combined_{tag}_subj{nn}.mat
%          Variable: fc_combined_{tag}_subj{nn}  shape (W, 216, 216)
%
% Output:  data/analysis/subcortical_s1_subjs/subj_fc_s1_{tag}_subj{nn}.mat
%          Variable: fc_s1_{tag}_subj{nn}  shape (W, 16, 16)
%
% Run time: a few seconds per subject (pure indexing, no computation).

clear; clc;

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';
fc_in_dir    = fullfile(PROJECT_ROOT, 'data', 'analysis', 'combined_subjs');
fc_out_dir   = fullfile(PROJECT_ROOT, 'data', 'analysis', 'subcortical_s1_subjs');
if ~exist(fc_out_dir, 'dir'), mkdir(fc_out_dir); end

N_CORTICAL = 200;
N_COMBINED = 216;
sc_idx     = (N_CORTICAL+1):N_COMBINED;   % 201:216

group_tags = {'an_patients', 'hc_patients'};

for g = 1:numel(group_tags)
    tag = group_tags{g};

    % Discover all available subjects for this group
    pattern  = fullfile(fc_in_dir, sprintf('subj_fc_combined_%s_subj*.mat', tag));
    fc_files = dir(pattern);
    n_subj   = numel(fc_files);

    fprintf('\n=== %s  (%d subjects) ===\n', upper(tag), n_subj);

    for s = 1:n_subj
        % Determine subject index from filename
        fname    = fc_files(s).name;
        tok      = regexp(fname, 'subj(\d+)\.mat$', 'tokens');
        subj_idx = str2double(tok{1}{1});

        in_file  = fullfile(fc_in_dir, fname);
        var_in   = sprintf('fc_combined_%s_subj%02d', tag, subj_idx);

        % Load combined FC
        S      = load(in_file, var_in);
        fc_all = S.(var_in);   % (W, 216, 216)  float32

        W = size(fc_all, 1);
        assert(size(fc_all,2) == N_COMBINED && size(fc_all,3) == N_COMBINED, ...
            'Unexpected shape in %s', fname);

        % Extract subcortical block: (W, 16, 16)
        fc_s1 = fc_all(:, sc_idx, sc_idx);

        % Save
        var_out  = sprintf('fc_s1_%s_subj%02d', tag, subj_idx);
        out_file = fullfile(fc_out_dir, ...
            sprintf('subj_fc_s1_%s_subj%02d.mat', tag, subj_idx));

        save_struct.(var_out) = fc_s1;
        save(out_file, '-struct', 'save_struct', '-v7.3');
        clear save_struct fc_all fc_s1 S;

        fprintf('  subj%02d  W=%d  -> %s\n', subj_idx, W, ...
            sprintf('subj_fc_s1_%s_subj%02d.mat', tag, subj_idx));
    end
end

fprintf('\nDone. Files saved to:\n  %s\n', fc_out_dir);
fprintf('Variable name format: fc_s1_{tag}_subj{nn}  shape (W, 16, 16)\n');
fprintf('\nSubcortical ROI ordering (cols 1–16 in output):\n');
fprintf('  1  HIP-rh   2  AMY-rh   3  pTHA-rh  4  aTHA-rh\n');
fprintf('  5  NAc-rh   6  GP-rh    7  PUT-rh   8  CAU-rh\n');
fprintf('  9  HIP-lh  10  AMY-lh  11  pTHA-lh  12  aTHA-lh\n');
fprintf(' 13  NAc-lh  14  GP-lh   15  PUT-lh   16  CAU-lh\n');
