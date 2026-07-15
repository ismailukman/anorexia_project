% subcortical_mlcd_5subj.m
% Quick-run: MLCD on first 5 AN + 5 HC subcortical subjects (Tian S2, N=32)

clear; clc; clear global;

addpath(genpath('/Users/ismaila/Documents/MATLAB/GenLouvain'));
addpath('/Users/ismaila/Documents/C-Codes/AnorexiaProject/code/matlab');

%% Parameters
gamma   = 1.0;
omega   = 1.0;
density = 0.30;
DO_PLOT = true;
N_SUBJ  = 5;        % first 5 per group

PROJECT_ROOT = '/Users/ismaila/Documents/C-Codes/AnorexiaProject';

%% Output dirs
out_dir       = fullfile(PROJECT_ROOT, 'data', 'analysis');
mlcd_subj_dir = fullfile(out_dir, 'mlcd_subjs_subcortical_5subj');
subjs_mlcd_dir= fullfile(mlcd_subj_dir, 'subjs_mlcd');
fig_dir       = fullfile(PROJECT_ROOT, 'output', 'figures', 'stage2_mlcd_subcortical_5subj');
for d = {mlcd_subj_dir, subjs_mlcd_dir, fig_dir}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

fc_dir = fullfile(out_dir, 'subcortical_subjs');

group_tags  = {'an_patients', 'hc_patients'};
group_names = {'Anorexia', 'Control'};

fprintf('===== SUBCORTICAL MLCD — 5+5 SUBJECTS =====\n');
fprintf('density=%.0f%%  gamma=%.1f  omega=%.1f\n', density*100, gamma, omega);

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
        fc_file = fullfile(fc_dir, sprintf('subj_fc_windows_%s_subj%02d.mat', tag, subj_i));
        S = load(fc_file);
        var_name = sprintf('fc_%s_subj%02d', tag, subj_i);
        fc_data  = S.(var_name);   % (W, 32, 32)

        sz = size(fc_data);
        if sz(2) == sz(3) && sz(1) ~= sz(2)
            fc_g = permute(fc_data, [2 3 1]);  % → (32, 32, W)
        else
            fc_g = fc_data;
        end

        N = size(fc_g,1);  W = size(fc_g,3);
        fprintf('  Subj %02d: N=%d | W=%d\n', subj_i, N, W);

        fc_g = 0.5*(fc_g + permute(fc_g,[2 1 3]));
        maskI = repmat(eye(N,'logical'),1,1,W);
        fc_g(maskI) = 0;
        fc_g(~isfinite(fc_g)) = 0;

        adj_g  = zeros(N,N,W);
        upmask = triu(true(N),1);
        [ri,ci]= find(upmask);
        M = numel(ri);
        K = max(1, round(density*M));

        for t = 1:W
            Ak = fc_g(:,:,t); Ak(Ak<0)=0;
            w  = Ak(upmask);
            pos= find(w>0);
            if isempty(pos), continue; end
            Kt = min(K,numel(pos));
            [~,ord]= sort(w(pos),'descend');
            keep   = pos(ord(1:Kt));
            Ak2    = zeros(N);
            Ak2(sub2ind([N N],ri(keep),ci(keep))) = w(keep);
            adj_g(:,:,t) = Ak2 + Ak2.';
        end

        % QC plot with colourbars
        if DO_PLOT
            n_cols   = min(6,W);
            idx_plot = round(linspace(1,W,n_cols));
            f = figure('Visible','off','Color','w','Position',[100 100 220*n_cols 520]);
            tl = tiledlayout(2, n_cols+1, 'Padding','compact','TileSpacing','compact');

            raw_vals = cell2mat(arrayfun(@(w) reshape(fc_data(w,:,:),[],1), idx_plot,'UniformOutput',false));
            clim_raw = [min(raw_vals(:)) max(raw_vals(:))];
            adj_vals = cell2mat(arrayfun(@(w) adj_g(:,:,w), idx_plot,'UniformOutput',false));
            clim_adj = [0 max(adj_vals(:)+eps)];

            for j = 1:n_cols
                kw = idx_plot(j);
                ax = nexttile(j);
                imagesc(ax, squeeze(fc_data(kw,:,:)), clim_raw); axis(ax,'image'); axis(ax,'off');
                title(ax, sprintf('Raw w%d',kw),'FontSize',7);
            end
            ax_cb1 = nexttile(n_cols+1); axis(ax_cb1,'off');
            cb1 = colorbar(ax_cb1,'Location','west');
            cb1.Limits = clim_raw; cb1.Label.String='Pearson r'; cb1.FontSize=8;
            colormap(ax_cb1, jet(256));

            for j = 1:n_cols
                kw = idx_plot(j);
                ax = nexttile(n_cols+1+j);
                imagesc(ax, adj_g(:,:,kw), clim_adj); axis(ax,'image'); axis(ax,'off');
                title(ax, sprintf('Adj w%d',kw),'FontSize',7);
            end
            ax_cb2 = nexttile(2*(n_cols+1)); axis(ax_cb2,'off');
            cb2 = colorbar(ax_cb2,'Location','west');
            cb2.Limits = clim_adj; cb2.Label.String='Edge weight'; cb2.FontSize=8;
            colormap(ax_cb2,'parula');
            colormap(f,'parula');

            title(tl, sprintf('%s subj%02d | Tian S2 (N=32) | density=%.0f%%', ...
                gname,subj_i,density*100),'FontSize',9,'FontWeight','bold');
            exportgraphics(f, fullfile(fig_dir, sprintf('%s_subj%02d.png',tag,subj_i)),'Resolution',200);
            close(f);
        end

        % MLCD
        A_subj = squeeze(num2cell(adj_g,[1 2]));
        mc = multilayer_community_detection_individual( ...
            A_subj(:).','ord','n_repeat',100,'thresh_type','max','gamma',gamma,'omega',omega);
        sd = mc{1};

        N_all_cell{subj_i} = sd.multi_module_consensus;
        comm_cell{subj_i}  = max(sd.multi_comm_consensus);
        Qmod_cell{subj_i}  = mode(cell2mat(sd.multi_modQ));

        out_s = struct('group',gname,'tag',tag,'subj_index_1based',subj_i, ...
            'gamma',gamma,'omega',omega,'density',density, ...
            'N_all_g',sd.multi_module_consensus, ...
            'comm_cons',max(sd.multi_comm_consensus), ...
            'Qmod',mode(cell2mat(sd.multi_modQ)),'atlas','Tian_Scale2_32ROIs');
        save(fullfile(mlcd_subj_dir,sprintf('mlcd_subcortical_%s_subj%02d.mat',tag,subj_i)), ...
            '-struct','out_s','-v7.3');
        fprintf('  ✓ subj %02d saved\n', subj_i);
        clear S fc_data fc_g adj_g A_subj mc sd out_s;
    end

    N_all_g = [N_all_cell{:}];

    if grp==1
        g = struct('N_all_g_anorexia',N_all_g, ...
                   'Q_g_anorexia',[Qmod_cell{:}], ...
                   'comm_cons_all_g_anorexia',[comm_cell{:}], ...
                   'atlas','Tian_Scale2_32ROIs','n_subj',N_SUBJ);
        save(fullfile(subjs_mlcd_dir,'mlcd_subcortical_anorexia_wins.mat'),'-struct','g','-v7.3');
        fprintf('Saved mlcd_subcortical_anorexia_wins.mat\n');
    else
        g = struct('N_all_g_control',N_all_g, ...
                   'Q_g_control',[Qmod_cell{:}], ...
                   'comm_cons_all_g_control',[comm_cell{:}], ...
                   'atlas','Tian_Scale2_32ROIs','n_subj',N_SUBJ);
        save(fullfile(subjs_mlcd_dir,'mlcd_subcortical_control_wins.mat'),'-struct','g','-v7.3');
        fprintf('Saved mlcd_subcortical_control_wins.mat\n');
    end
end

t=toc;
if t<60, fprintf('Done in %.1fs\n',t);
else,    fprintf('Done in %dm %.1fs\n',floor(t/60),rem(t,60)); end
