% generate_subcortical_s1_fc_5subj.m
%
% Extract Tian Scale I (16-ROI) subcortical FC for the pilot 5 AN + 5 HC
% subjects from the combined 216-region windowed FC matrices.
%
% Combined atlas ordering:
%   Rows/cols 1–200   : Schaefer-200 cortical
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

clear; clc;

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';
fc_in_dir    = fullfile(PROJECT_ROOT, 'data', 'analysis', 'combined_subjs');
fc_out_dir   = fullfile(PROJECT_ROOT, 'data', 'analysis', 'subcortical_s1_subjs');
if ~exist(fc_out_dir, 'dir'), mkdir(fc_out_dir); end

N_CORTICAL = 200;
N_COMBINED = 216;
N_SUBJ     = 5;
sc_idx     = (N_CORTICAL+1):N_COMBINED;   % 201:216

group_tags = {'an_patients', 'hc_patients'};

fprintf('===== Tian S1 subcortical FC — pilot 5+5 subjects =====\n');
fprintf('Extracting rows/cols 201–216 from combined (W,216,216) matrices\n\n');

for g = 1:numel(group_tags)
    tag = group_tags{g};
    fprintf('--- %s ---\n', upper(tag));

    for subj_i = 1:N_SUBJ
        in_file = fullfile(fc_in_dir, ...
            sprintf('subj_fc_combined_%s_subj%02d.mat', tag, subj_i));

        if ~exist(in_file, 'file')
            warning('File not found, skipping: %s', in_file);
            continue;
        end

        var_in = sprintf('fc_combined_%s_subj%02d', tag, subj_i);
        S      = load(in_file, var_in);
        fc_all = S.(var_in);   % (W, 216, 216)

        W = size(fc_all, 1);
        assert(size(fc_all,2) == N_COMBINED && size(fc_all,3) == N_COMBINED, ...
            'Unexpected combined matrix shape in %s', in_file);

        % Extract 16×16 subcortical block
        fc_s1 = fc_all(:, sc_idx, sc_idx);   % (W, 16, 16)

        % Save
        var_out  = sprintf('fc_s1_%s_subj%02d', tag, subj_i);
        out_file = fullfile(fc_out_dir, ...
            sprintf('subj_fc_s1_%s_subj%02d.mat', tag, subj_i));

        tmp.(var_out) = fc_s1;
        save(out_file, '-struct', 'tmp', '-v7.3');
        clear tmp S fc_all fc_s1;

        fprintf('  subj%02d  W=%d  saved -> subj_fc_s1_%s_subj%02d.mat\n', ...
            subj_i, W, tag, subj_i);
    end
end

fprintf('\nDone. Output: %s\n', fc_out_dir);
fprintf('Variable: fc_s1_{tag}_subj{nn}   shape (W, 16, 16)\n');
