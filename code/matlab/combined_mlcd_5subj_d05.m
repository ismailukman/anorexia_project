% combined_mlcd_5subj_d05.m
% MLCD on the combined Schaefer-200 + Tian S1 atlas (216 regions)
% at 5% edge density, first 5 AN + 5 HC subjects.
%
% Region ordering (must match 01c_combined_preparation.py):
%   1–200   Schaefer-200 cortical   (Yeo-7, LH then RH)
%   201–216 Tian S1 subcortical     (rh then lh):
%           201 HIP-rh  202 AMY-rh  203 pTHA-rh 204 aTHA-rh
%           205 NAc-rh  206 GP-rh   207 PUT-rh  208 CAU-rh
%           209 HIP-lh  210 AMY-lh  211 pTHA-lh 212 aTHA-lh
%           213 NAc-lh  214 GP-lh   215 PUT-lh  216 CAU-lh
%
% N=216, M=216*215/2 = 23220 upper-triangle edges
%   density=0.05  -> K=1161 edges/window

clear; clc; clear global;

addpath(genpath('/Users/ismaila/Documents/MATLAB/GenLouvain'));
addpath('/Users/ismaila/Documents/C-Codes/AnorexiaProject/code/matlab');

%% Parameters
gamma   = 1.0;
omega   = 1.0;
density = 0.05;
DO_PLOT = true;
N_SUBJ  = 5;

N_CORTICAL  = 200;
N_SUBCORT   = 16;
N_COMBINED  = N_CORTICAL + N_SUBCORT;   % 216

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';

%% Load combined atlas labels
label_file = fullfile(PROJECT_ROOT, 'data', 'atlas', 'combined_216', ...
    'combined_216_labels.txt');
fid = fopen(label_file, 'r');
roi_labels = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line)
        line = strtrim(line);
        if isempty(line) || line(1) == '#', continue; end
        parts = strsplit(line);
        if numel(parts) >= 2
            roi_labels{end+1, 1} = parts{2};
        end
    end
end
fclose(fid);
fprintf('Loaded %d ROI labels\n', numel(roi_labels));
assert(numel(roi_labels) == N_COMBINED, ...
    'Expected %d labels, got %d', N_COMBINED, numel(roi_labels));

%% Output directories
out_dir        = fullfile(PROJECT_ROOT, 'data', 'analysis');
mlcd_subj_dir  = fullfile(out_dir, 'mlcd_subjs_combined_5subj_d05');
subjs_mlcd_dir = fullfile(mlcd_subj_dir, 'subjs_mlcd');
fig_dir        = fullfile(PROJECT_ROOT, 'output', 'figures', ...
    'stage2_mlcd_combined_5subj_d05');
for d = {mlcd_subj_dir, subjs_mlcd_dir, fig_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fc_dir = fullfile(out_dir, 'combined_subjs');

group_tags  = {'an_patients', 'hc_patients'};
group_names = {'Anorexia', 'Control'};

fprintf('===== COMBINED 216-ROI MLCD — 5+5 SUBJECTS, density=%.0f%% =====\n', density*100);
fprintf('N=%d (cortical=%d + subcortical=%d)\n', N_COMBINED, N_CORTICAL, N_SUBCORT);
fprintf('M=%d upper-triangle edges  ->  K=%d edges/window at %.0f%%\n', ...
    N_COMBINED*(N_COMBINED-1)/2, round(density*N_COMBINED*(N_COMBINED-1)/2), density*100);
fprintf('Input : %s\n', fc_dir);
fprintf('Output: %s\n', mlcd_subj_dir);

if isempty(gcp('nocreate')), parpool('local'); end

tic;

for grp = 1:2
    tag   = group_tags{grp};
    gname = group_names{grp};

    fprintf('\n===== GROUP: %s (%s) — %d subjects =====\n', gname, tag, N_SUBJ);

    N_all_cell = cell(1, N_SUBJ);
    comm_cell  = cell(1, N_SUBJ);
    Qmod_cell  = cell(1, N_SUBJ);

    for subj_i = 1:N_SUBJ

        %% Load combined windowed FC
        fc_file  = fullfile(fc_dir, sprintf('subj_fc_combined_%s_subj%02d.mat', tag, subj_i));
        if ~exist(fc_file, 'file')
            error('File not found: %s\nRun 01c_combined_preparation.py first.', fc_file);
        end
        S        = load(fc_file);
        var_name = sprintf('fc_combined_%s_subj%02d', tag, subj_i);
        fc_data  = S.(var_name);   % (W, 216, 216)

        %% Orient to (216, 216, W)
        sz = size(fc_data);
        if sz(2) == sz(3) && sz(1) ~= sz(2)
            fc_g = permute(fc_data, [2 3 1]);   % (W,N,N) -> (N,N,W)
        else
            fc_g = fc_data;
        end

        N = size(fc_g, 1);
        W = size(fc_g, 3);
        assert(N == N_COMBINED, 'Expected N=%d, got N=%d', N_COMBINED, N);
        fprintf('  Subj %02d: N=%d | W=%d windows\n', subj_i, N, W);

        %% Clean: symmetrise, zero diagonal, remove non-finite
        fc_g = 0.5 * (fc_g + permute(fc_g, [2 1 3]));
        maskI = repmat(eye(N, 'logical'), 1, 1, W);
        fc_g(maskI) = 0;
        fc_g(~isfinite(fc_g)) = 0;

        %% Threshold: keep top 5% positive edges per window
        upmask  = triu(true(N), 1);
        [ri,ci] = find(upmask);
        M  = numel(ri);
        K  = max(1, round(density * M));

        adj_g = zeros(N, N, W);

        for t = 1:W
            Ak = fc_g(:,:,t);
            Ak(Ak < 0) = 0;
            w   = Ak(upmask);
            pos = find(w > 0);
            if isempty(pos), continue; end
            Kt = min(K, numel(pos));
            [~, ord] = sort(w(pos), 'descend');
            keep = pos(ord(1:Kt));
            Ak2  = zeros(N);
            Ak2(sub2ind([N N], ri(keep), ci(keep))) = w(keep);
            adj_g(:,:,t) = Ak2 + Ak2.';
        end

        %% QC plot — show FC matrix with cortical/subcortical block boundary
        if DO_PLOT
            n_cols   = min(4, W);
            idx_plot = round(linspace(1, W, n_cols));
            f = figure('Visible','off','Color','w', ...
                'Position',[100 100 300*n_cols 600]);
            tl = tiledlayout(2, n_cols, 'Padding','compact','TileSpacing','compact');

            clim_raw = [min(fc_data(:)) max(fc_data(:))];
            clim_adj = [0 max(adj_g(:) + eps)];

            for j = 1:n_cols
                kw = idx_plot(j);

                % Raw FC
                ax = nexttile(j);
                imagesc(ax, squeeze(fc_data(kw,:,:)), clim_raw);
                axis(ax,'image'); axis(ax,'off');
                hold(ax,'on');
                plot(ax, [N_CORTICAL+0.5 N_CORTICAL+0.5], [0.5 N+0.5], 'w-', 'LineWidth',1);
                plot(ax, [0.5 N+0.5], [N_CORTICAL+0.5 N_CORTICAL+0.5], 'w-', 'LineWidth',1);
                title(ax, sprintf('Raw w%d', kw), 'FontSize',7);
                colormap(ax, jet(256));

                % Thresholded
                ax2 = nexttile(n_cols+j);
                imagesc(ax2, adj_g(:,:,kw), clim_adj);
                axis(ax2,'image'); axis(ax2,'off');
                hold(ax2,'on');
                plot(ax2, [N_CORTICAL+0.5 N_CORTICAL+0.5], [0.5 N+0.5], 'w-', 'LineWidth',1);
                plot(ax2, [0.5 N+0.5], [N_CORTICAL+0.5 N_CORTICAL+0.5], 'w-', 'LineWidth',1);
                title(ax2, sprintf('Adj w%d (5%%)', kw), 'FontSize',7);
                colormap(ax2, 'parula');
            end

            title(tl, sprintf('%s subj%02d | Schaefer-200+TianS1 (N=%d) | density=5%%', ...
                gname, subj_i, N), 'FontSize',9, 'FontWeight','bold');
            exportgraphics(f, fullfile(fig_dir, ...
                sprintf('%s_subj%02d_combined_d05.png', tag, subj_i)), 'Resolution',200);
            close(f);
        end

        %% MLCD
        A_subj = squeeze(num2cell(adj_g, [1 2]));
        mc = multilayer_community_detection_individual( ...
            A_subj(:).', 'ord', 'n_repeat', 100, 'thresh_type', 'max', ...
            'gamma', gamma, 'omega', omega);
        sd = mc{1};

        N_all_cell{subj_i} = sd.multi_module_consensus;
        comm_cell{subj_i}  = max(sd.multi_comm_consensus);
        Qmod_cell{subj_i}  = mode(cell2mat(sd.multi_modQ));
        fprintf('  Subj %02d: communities=%d  Q=%.3f\n', ...
            subj_i, comm_cell{subj_i}, Qmod_cell{subj_i});

        %% Save per-subject
        out_s = struct('group',gname,'tag',tag,'subj_index_1based',subj_i, ...
            'gamma',gamma,'omega',omega,'density',density, ...
            'N_all_g',sd.multi_module_consensus, ...
            'comm_cons',max(sd.multi_comm_consensus), ...
            'Qmod',mode(cell2mat(sd.multi_modQ)), ...
            'atlas','Schaefer200_Yeo7+TianS1_216ROIs', ...
            'n_cortical',N_CORTICAL,'n_subcortical',N_SUBCORT);
        save(fullfile(mlcd_subj_dir, ...
            sprintf('mlcd_combined_%s_subj%02d_d05.mat', tag, subj_i)), ...
            '-struct','out_s','-v7.3');
        fprintf('  Saved -> mlcd_combined_%s_subj%02d_d05.mat\n', tag, subj_i);

        clear S fc_data fc_g adj_g A_subj mc sd out_s;
    end

    %% Group-level save
    N_all_g = [N_all_cell{:}];
    if grp == 1
        g = struct('N_all_g_anorexia',N_all_g, ...
                   'Q_g_anorexia',[Qmod_cell{:}], ...
                   'comm_cons_all_g_anorexia',[comm_cell{:}], ...
                   'atlas','Schaefer200_Yeo7+TianS1_216ROIs', ...
                   'n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_combined_anorexia_wins_d05.mat'), ...
            '-struct','g','-v7.3');
        fprintf('Saved mlcd_combined_anorexia_wins_d05.mat\n');
    else
        g = struct('N_all_g_control',N_all_g, ...
                   'Q_g_control',[Qmod_cell{:}], ...
                   'comm_cons_all_g_control',[comm_cell{:}], ...
                   'atlas','Schaefer200_Yeo7+TianS1_216ROIs', ...
                   'n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_combined_control_wins_d05.mat'), ...
            '-struct','g','-v7.3');
        fprintf('Saved mlcd_combined_control_wins_d05.mat\n');
    end

    clear N_all_cell comm_cell Qmod_cell N_all_g;
end

t = toc;
if     t < 60,   fprintf('Done in %.1f sec\n', t);
elseif t < 3600, fprintf('Done in %dm %.1fs\n', floor(t/60), rem(t,60));
else,            fprintf('Done in %dh %dm\n',   floor(t/3600), floor(rem(t,3600)/60));
end
