% subcortical_mlcd_5subj_d05.m
% Subcortical MLCD for first 5 AN + 5 HC subjects at 5% edge density.
% Mirrors subcortical_mlcd_5subj.m (density=0.30) but uses density=0.05.
% Also produces a comparison plot showing which ROIs survive at 5% vs 30%.
%
% Tian S2 atlas: N=32, M=496 upper-triangle edges
%   density=0.05 → K=25  edges/window  (avg degree ~1.6)
%   density=0.30 → K=149 edges/window  (avg degree ~9.3)

clear; clc; clear global;

addpath(genpath('/Users/ismaila/Documents/MATLAB/GenLouvain'));
addpath('/Users/ismaila/Documents/C-Codes/AnorexiaProject/code/matlab');

%% Parameters
gamma   = 1.0;
omega   = 1.0;
density = 0.05;   % 5% edge density
DO_PLOT = true;
N_SUBJ  = 5;

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';

%% Atlas ROI labels (Tian S2, 32 regions)
label_file = fullfile(PROJECT_ROOT, 'data', 'atlas', 'tian_s2', ...
    'Tian_Subcortex_S2_3T_label.txt');
fid = fopen(label_file, 'r');
roi_labels = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line)
        parts = strsplit(strtrim(line));
        if numel(parts) >= 2
            roi_labels{end+1,1} = parts{2};
        end
    end
end
fclose(fid);
N_roi = numel(roi_labels);

%% Output dirs
out_dir        = fullfile(PROJECT_ROOT, 'data', 'analysis');
mlcd_subj_dir  = fullfile(out_dir, 'mlcd_subjs_subcortical_5subj_d05');
subjs_mlcd_dir = fullfile(mlcd_subj_dir, 'subjs_mlcd');
fig_dir        = fullfile(PROJECT_ROOT, 'output', 'figures', ...
    'stage2_mlcd_subcortical_5subj_d05');
for d = {mlcd_subj_dir, subjs_mlcd_dir, fig_dir}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

fc_dir = fullfile(out_dir, 'subcortical_subjs');

group_tags  = {'an_patients', 'hc_patients'};
group_names = {'Anorexia', 'Control'};

fprintf('===== SUBCORTICAL MLCD — 5+5 SUBJECTS, density=5%% =====\n');
fprintf('density=%.0f%%  gamma=%.1f  omega=%.1f\n', density*100, gamma, omega);
fprintf('Atlas: Tian S2, N=%d ROIs\n', N_roi);
fprintf('Input : %s\n', fc_dir);
fprintf('Output: %s\n', mlcd_subj_dir);

if isempty(gcp('nocreate')), parpool('local'); end

tic;

% Accumulators for comparison plot: each row = one (window × subject) sample
deg5_all  = zeros(0, N_roi);   % degree at  5%
deg30_all = zeros(0, N_roi);   % degree at 30% (computed alongside for free)

for grp = 1:2
    tag   = group_tags{grp};
    gname = group_names{grp};

    fprintf('\n===== GROUP: %s (%s) — %d subjects =====\n', gname, tag, N_SUBJ);

    N_all_cell = cell(1, N_SUBJ);
    comm_cell  = cell(1, N_SUBJ);
    Qmod_cell  = cell(1, N_SUBJ);

    for subj_i = 1:N_SUBJ

        %% Load FC windows
        fc_file  = fullfile(fc_dir, sprintf('subj_fc_windows_%s_subj%02d.mat', tag, subj_i));
        S        = load(fc_file);
        var_name = sprintf('fc_%s_subj%02d', tag, subj_i);
        fc_data  = S.(var_name);   % (W, 32, 32)

        sz = size(fc_data);
        if sz(2) == sz(3) && sz(1) ~= sz(2)
            fc_g = permute(fc_data, [2 3 1]);  % (W,32,32) → (32,32,W)
        else
            fc_g = fc_data;
        end

        N = size(fc_g,1);
        W = size(fc_g,3);
        fprintf('  Subj %02d: N=%d | W=%d windows\n', subj_i, N, W);

        %% Clean: symmetrise, zero diagonal, remove non-finite
        fc_g = 0.5 * (fc_g + permute(fc_g,[2 1 3]));
        maskI = repmat(eye(N,'logical'), 1, 1, W);
        fc_g(maskI) = 0;
        fc_g(~isfinite(fc_g)) = 0;

        %% Pre-compute threshold sizes
        upmask  = triu(true(N), 1);
        [ri,ci] = find(upmask);
        M   = numel(ri);                       % 496 for N=32
        K5  = max(1, round(0.05 * M));         % ~25  edges
        K30 = max(1, round(0.30 * M));         % ~149 edges

        adj_g = zeros(N, N, W);                % thresholded at 5%

        for t = 1:W
            Ak = fc_g(:,:,t);
            Ak(Ak < 0) = 0;
            w   = Ak(upmask);
            pos = find(w > 0);
            if isempty(pos), continue; end

            % Sort edges descending (used for both thresholds)
            [~, ord] = sort(w(pos), 'descend');

            % --- 5% threshold → adjacency for MLCD ---
            Kt5   = min(K5, numel(pos));
            keep5 = pos(ord(1:Kt5));
            Ak5   = zeros(N);
            Ak5(sub2ind([N N], ri(keep5), ci(keep5))) = w(keep5);
            adj_g(:,:,t) = Ak5 + Ak5.';
            deg5_all(end+1, :) = sum(Ak5 + Ak5.' > 0, 2)';   % degree vector

            % --- 30% threshold → degree only (for comparison plot) ---
            Kt30   = min(K30, numel(pos));
            keep30 = pos(ord(1:Kt30));
            Ak30   = zeros(N);
            Ak30(sub2ind([N N], ri(keep30), ci(keep30))) = w(keep30);
            deg30_all(end+1, :) = sum(Ak30 + Ak30.' > 0, 2)';
        end

        %% QC plot (raw vs 5%-thresholded adjacency)
        if DO_PLOT
            n_cols   = min(6, W);
            idx_plot = round(linspace(1, W, n_cols));
            f = figure('Visible','off','Color','w','Position',[100 100 220*n_cols 520]);
            tl = tiledlayout(2, n_cols+1, 'Padding','compact','TileSpacing','compact');

            raw_vals = cell2mat(arrayfun(@(kk) reshape(fc_data(kk,:,:),[],1), ...
                idx_plot,'UniformOutput',false));
            clim_raw = [min(raw_vals(:)) max(raw_vals(:))];
            adj_vals = cell2mat(arrayfun(@(kk) adj_g(:,:,kk), ...
                idx_plot,'UniformOutput',false));
            clim_adj = [0, max(adj_vals(:) + eps)];

            for j = 1:n_cols
                kw = idx_plot(j);
                ax = nexttile(j);
                imagesc(ax, squeeze(fc_data(kw,:,:)), clim_raw);
                axis(ax,'image'); axis(ax,'off');
                title(ax, sprintf('Raw w%d',kw), 'FontSize',7);
            end
            ax_cb1 = nexttile(n_cols+1); axis(ax_cb1,'off');
            cb1 = colorbar(ax_cb1,'Location','west');
            cb1.Limits = clim_raw; cb1.Label.String = 'Pearson r'; cb1.FontSize = 8;
            colormap(ax_cb1, jet(256));

            for j = 1:n_cols
                kw = idx_plot(j);
                ax = nexttile(n_cols+1+j);
                imagesc(ax, adj_g(:,:,kw), clim_adj);
                axis(ax,'image'); axis(ax,'off');
                title(ax, sprintf('Adj w%d (5%%)',kw), 'FontSize',7);
            end
            ax_cb2 = nexttile(2*(n_cols+1)); axis(ax_cb2,'off');
            cb2 = colorbar(ax_cb2,'Location','west');
            cb2.Limits = clim_adj; cb2.Label.String = 'Edge weight'; cb2.FontSize = 8;
            colormap(ax_cb2,'parula');
            colormap(f,'parula');

            title(tl, sprintf('%s subj%02d | Tian S2 (N=%d) | density=5%%', ...
                gname, subj_i, N), 'FontSize',9, 'FontWeight','bold');
            exportgraphics(f, fullfile(fig_dir, ...
                sprintf('%s_subj%02d_d05.png', tag, subj_i)), 'Resolution',200);
            close(f);
        end

        %% MLCD at 5%
        A_subj = squeeze(num2cell(adj_g,[1 2]));
        mc = multilayer_community_detection_individual( ...
            A_subj(:).', 'ord', 'n_repeat', 100, 'thresh_type', 'max', ...
            'gamma', gamma, 'omega', omega);
        sd = mc{1};

        N_all_cell{subj_i} = sd.multi_module_consensus;
        comm_cell{subj_i}  = max(sd.multi_comm_consensus);
        Qmod_cell{subj_i}  = mode(cell2mat(sd.multi_modQ));
        fprintf('  Subj %02d: communities=%d  Q=%.3f\n', ...
            subj_i, comm_cell{subj_i}, Qmod_cell{subj_i});

        out_s = struct('group',gname,'tag',tag,'subj_index_1based',subj_i, ...
            'gamma',gamma,'omega',omega,'density',density, ...
            'N_all_g',sd.multi_module_consensus, ...
            'comm_cons',max(sd.multi_comm_consensus), ...
            'Qmod',mode(cell2mat(sd.multi_modQ)), ...
            'atlas','Tian_Scale2_32ROIs');
        save(fullfile(mlcd_subj_dir, ...
            sprintf('mlcd_subcortical_%s_subj%02d_d05.mat', tag, subj_i)), ...
            '-struct','out_s','-v7.3');
        fprintf('  Saved -> mlcd_subcortical_%s_subj%02d_d05.mat\n', tag, subj_i);

        clear S fc_data fc_g adj_g A_subj mc sd out_s;
    end

    %% Group-level save
    N_all_g = [N_all_cell{:}];
    if grp == 1
        g = struct('N_all_g_anorexia',N_all_g, ...
                   'Q_g_anorexia',[Qmod_cell{:}], ...
                   'comm_cons_all_g_anorexia',[comm_cell{:}], ...
                   'atlas','Tian_Scale2_32ROIs','n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_subcortical_anorexia_wins_d05.mat'), ...
            '-struct','g','-v7.3');
        fprintf('Saved mlcd_subcortical_anorexia_wins_d05.mat\n');
    else
        g = struct('N_all_g_control',N_all_g, ...
                   'Q_g_control',[Qmod_cell{:}], ...
                   'comm_cons_all_g_control',[comm_cell{:}], ...
                   'atlas','Tian_Scale2_32ROIs','n_subj',N_SUBJ,'density',density);
        save(fullfile(subjs_mlcd_dir,'mlcd_subcortical_control_wins_d05.mat'), ...
            '-struct','g','-v7.3');
        fprintf('Saved mlcd_subcortical_control_wins_d05.mat\n');
    end

    clear N_all_cell comm_cell Qmod_cell N_all_g;
end

%% =========================================================
%% Comparison plot: mean degree per ROI at 5% vs 30%
%% =========================================================
% For each ROI: mean number of surviving edges across all windows & subjects
mean_d5  = mean(deg5_all,  1);   % 1×32
mean_d30 = mean(deg30_all, 1);   % 1×32

% Fraction of windows where ROI has at least one edge (connectivity rate)
conn5  = mean(deg5_all  > 0, 1) * 100;   % % windows connected at  5%
conn30 = mean(deg30_all > 0, 1) * 100;   % % windows connected at 30%

% Sort ROIs by mean degree at 30% (descending) so highest-hub ROIs are at top
[~, sort_idx] = sort(mean_d30, 'descend');
labels_sorted = roi_labels(sort_idx);
d5_sorted     = mean_d5(sort_idx);
d30_sorted    = mean_d30(sort_idx);
conn5_sorted  = conn5(sort_idx);

fig_cmp = figure('Color','w','Position',[100 100 980 760]);
ax = axes('Parent', fig_cmp);
hold(ax,'on');

% Horizontal grouped bar
ypos = 1:N_roi;
bh   = barh(ax, ypos, [d5_sorted(:), d30_sorted(:)], 'grouped');
bh(1).FaceColor = [0.18 0.55 0.90];   % blue  = 5%
bh(2).FaceColor = [0.88 0.35 0.20];   % red   = 30%
bh(1).EdgeColor = 'none';
bh(2).EdgeColor = 'none';
bh(1).BarWidth  = 0.75;
bh(2).BarWidth  = 0.75;

% Connectivity rate annotation next to 5% bar (% windows ROI is connected)
bar_offset = max(d30_sorted) * 0.015;
for k = 1:N_roi
    text(ax, d5_sorted(k) + bar_offset, ypos(k) - 0.21, ...
        sprintf('%.0f%%', conn5_sorted(k)), ...
        'FontSize',6.5, 'Color',[0.10 0.38 0.70], 'VerticalAlignment','middle');
end

% Vertical line at maximum 5% degree for reference
xmax5 = max(d5_sorted);
xline(ax, xmax5, '--', 'Color',[0.18 0.55 0.90], 'LineWidth',0.8, 'Alpha',0.5);

ax.YTick             = ypos;
ax.YTickLabel        = labels_sorted;
ax.TickLabelInterpreter = 'none';
ax.FontSize          = 9;
ax.XLabel.String     = sprintf('Mean degree (surviving edges per ROI,  N=%d ROIs, M=%d possible)', ...
    N_roi, N_roi*(N_roi-1)/2);
ax.XLabel.FontSize   = 10;
ax.Title.String      = sprintf(['Subcortical ROI Degree: 5%% vs 30%% Edge Density\n' ...
    'Tian S2 (N=%d) — first %d AN + %d HC subjects'], N_roi, N_SUBJ, N_SUBJ);
ax.Title.FontSize    = 11;
grid(ax,'on'); ax.GridAlpha = 0.20; ax.GridLineStyle = ':';
ax.XLim = [0, max(d30_sorted) * 1.18];
ax.YLim = [0.2, N_roi + 0.8];
ax.YDir = 'reverse';   % highest-hub ROI at top

legend(ax, {'5% density','30% density'}, ...
    'Location','southeast','FontSize',10,'Box','on');

% Second x-axis on top showing density-equivalent degree range
ax2 = axes('Position',ax.Position,'XAxisLocation','top','Color','none');
ax2.XLim   = ax.XLim / (N_roi-1) * 100;   % convert degree → % of max possible
ax2.XLabel.String   = 'Degree as % of max possible (N-1=31)';
ax2.XLabel.FontSize = 9;
ax2.YTick  = [];
linkaxes([ax ax2],'x');

exportgraphics(fig_cmp, fullfile(fig_dir, 'subcortical_roi_degree_5vs30pct.png'), ...
    'Resolution',200);
fprintf('\nComparison plot saved → subcortical_roi_degree_5vs30pct.png\n');

%% Elapsed time
t = toc;
if     t < 60,   fprintf('Done in %.1f sec\n', t);
elseif t < 3600, fprintf('Done in %dm %.1fs\n', floor(t/60), rem(t,60));
else,            fprintf('Done in %dh %dm\n', floor(t/3600), floor(rem(t,3600)/60));
end
