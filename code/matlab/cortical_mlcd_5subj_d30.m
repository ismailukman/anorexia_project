% cortical_mlcd_5subj_d30.m
% Cortical MLCD for first 5 AN + 5 HC subjects at 30% edge density.
% Mirrors cons_mlcd_win_parallel_an_v2.m but:
%   - N_SUBJ = 5 per group
%   - density = 0.30  (matched to subcortical_mlcd_5subj.m)
%   - outputs to mlcd_subjs_cortical_5subj_d30/ (does NOT touch mlcd_subjs/)

clear; clc; clear global;

%% Toolbox paths
addpath(genpath('/Users/ismaila/Documents/MATLAB/GenLouvain'));
addpath('/Users/ismaila/Documents/C-Codes/AnorexiaProject/code/matlab');

%% Parameters
gamma   = 1.0;
omega   = 1.0;
density = 0.30;   % matched to subcortical for equal comparison
N_SUBJ  = 5;
DO_PLOT = true;

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';
out_dir      = fullfile(PROJECT_ROOT, 'data', 'analysis');

%% Output directories (separate from the 22+22 run)
mlcd_subj_dir  = fullfile(out_dir, 'mlcd_subjs_cortical_5subj_d30');
subjs_mlcd_dir = fullfile(mlcd_subj_dir, 'subjs_mlcd');
fig_dir        = fullfile(PROJECT_ROOT, 'output', 'figures', 'stage2_mlcd_cortical_5subj_d30');

for d = {mlcd_subj_dir, subjs_mlcd_dir, fig_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

%% Input directory (Stage 1 cortical FC windows)
corr_subj_dir = fullfile(out_dir, 'corr_subjs');

group_tags  = {'an_patients', 'hc_patients'};
group_names = {'Anorexia', 'Control'};

fprintf('===== CORTICAL MLCD — 5+5 SUBJECTS, density=%.0f%% =====\n', density*100);
fprintf('gamma=%.1f  omega=%.1f  density=%.0f%%\n', gamma, omega, density*100);
fprintf('Input : %s\n', corr_subj_dir);
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

        %% Load FC windows
        fpat = sprintf('corr_%s_subj%02d_*tr_windows.mat', tag, subj_i);
        d    = dir(fullfile(corr_subj_dir, fpat));
        if isempty(d)
            error('File not found for %s subj%02d. Pattern: %s', tag, subj_i, fpat);
        end
        S = load(fullfile(d(1).folder, d(1).name));

        fn           = fieldnames(S);
        corr_fields  = fn(startsWith(fn,'corr_') & ~strcmp(fn,'__meta__'));
        corr_data    = S.(corr_fields{1});   % (W, N, N) from Python

        %% Orient to (N, N, W)
        sz = size(corr_data);
        if sz(1) ~= sz(2)
            corr_g = permute(corr_data, [2 3 1]);  % (W,N,N) -> (N,N,W)
        else
            corr_g = corr_data;                    % already (N,N,W)
        end

        N = size(corr_g,1);
        W = size(corr_g,3);
        fprintf('  Subj %02d: N=%d regions | W=%d windows\n', subj_i, N, W);

        %% Clean: symmetrise, zero diagonal, remove non-finite
        corr_g = 0.5 * (corr_g + permute(corr_g, [2 1 3]));
        maskI  = repmat(eye(N,'logical'), 1, 1, W);
        corr_g(maskI) = 0;
        corr_g(~isfinite(corr_g)) = 0;

        %% Threshold: keep top density% of positive edges per window
        adj_g  = zeros(N, N, W);
        upmask = triu(true(N), 1);
        [ri, ci] = find(upmask);
        M = numel(ri);
        K = max(1, round(density * M));

        for t = 1:W
            Ak = corr_g(:,:,t);
            Ak(Ak < 0) = 0;
            w  = Ak(upmask);
            pos = find(w > 0);
            if isempty(pos), continue; end
            Kt = min(K, numel(pos));
            [~, ord] = sort(w(pos), 'descend');
            keep = pos(ord(1:Kt));
            Ak2  = zeros(N);
            Ak2(sub2ind([N N], ri(keep), ci(keep))) = w(keep);
            adj_g(:,:,t) = Ak2 + Ak2.';
        end

        %% QC plot
        if DO_PLOT
            idx = 10:10:min(50, W);
            if ~isempty(idx)
                f  = figure('Visible','off','Color','w', ...
                    'Name', sprintf('%s subj%02d cortical d30', gname, subj_i));
                tl = tiledlayout(3, numel(idx), 'Padding','compact','TileSpacing','compact');

                clim_all = [min(corr_data(:)), max(corr_data(:))];
                for j = 1:numel(idx)
                    kw = idx(j);
                    nexttile(j);
                    imagesc(squeeze(corr_data(kw,:,:)), clim_all); axis image off;
                    title(sprintf('Raw w%d', kw), 'FontSize',6);

                    nexttile(j + numel(idx));
                    imagesc(corr_g(:,:,kw), clim_all); axis image off;
                    title(sprintf('Clean w%d', kw), 'FontSize',6);

                    nexttile(j + 2*numel(idx));
                    imagesc(adj_g(:,:,kw)); axis image off;
                    title(sprintf('Adj w%d', kw), 'FontSize',6);
                end
                colormap(parula);
                title(tl, sprintf('%s subj%02d | Schaefer-200 | density=%.0f%%', ...
                    gname, subj_i, density*100), 'FontSize',8,'FontWeight','bold');
                exportgraphics(f, fullfile(fig_dir, ...
                    sprintf('%s_subj%02d_plot.png', tag, subj_i)), 'Resolution',200);
                close(f);
            end
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

        fprintf('  Subj %02d: communities=%d, Q=%.3f\n', ...
            subj_i, comm_cell{subj_i}, Qmod_cell{subj_i});

        %% Save per-subject
        out_s = struct('group',gname,'tag',tag,'subj_index_1based',subj_i, ...
            'gamma',gamma,'omega',omega,'density',density, ...
            'N_all_g',sd.multi_module_consensus, ...
            'comm_cons',max(sd.multi_comm_consensus), ...
            'Qmod',mode(cell2mat(sd.multi_modQ)), ...
            'atlas','Schaefer200_Yeo7');
        save(fullfile(mlcd_subj_dir, sprintf('mlcd_%s_subj%02d.mat', tag, subj_i)), ...
            '-struct','out_s','-v7.3');
        fprintf('  Saved -> mlcd_%s_subj%02d.mat\n', tag, subj_i);

        clear S corr_data corr_g adj_g A_subj mc sd out_s;
    end

    %% Group-level save
    N_all_g = [N_all_cell{:}];
    if grp == 1
        g = struct('N_all_g_anorexia', N_all_g, ...
                   'Q_g_anorexia',     [Qmod_cell{:}], ...
                   'comm_cons_all_g_anorexia', [comm_cell{:}], ...
                   'atlas','Schaefer200_Yeo7','n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_anorexia_wins.mat'),'-struct','g','-v7.3');
        fprintf('Saved mlcd_anorexia_wins.mat\n');
    else
        g = struct('N_all_g_control',  N_all_g, ...
                   'Q_g_control',      [Qmod_cell{:}], ...
                   'comm_cons_all_g_control', [comm_cell{:}], ...
                   'atlas','Schaefer200_Yeo7','n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_control_wins.mat'),'-struct','g','-v7.3');
        fprintf('Saved mlcd_control_wins.mat\n');
    end

    clear N_all_cell comm_cell Qmod_cell N_all_g;
end

t = toc;
if     t < 60,   fprintf('Done in %.1f sec\n', t);
elseif t < 3600, fprintf('Done in %dm %.1fs\n', floor(t/60), rem(t,60));
else,            fprintf('Done in %dh %dm\n', floor(t/3600), floor(rem(t,3600)/60));
end
