% combined_mlcd_full44_d05_par.m
% Multilayer community detection (MLCD) on the combined Schaefer-200 + Tian S1
% atlas (216 regions) at 5% edge density for all 22 AN and 22 HC participants.
% gamma = 1, omega = 1, 100 GenLouvain runs per participant with iterative
% consensus (thresh_type 'max'). Participants are processed in parallel (parfor).
%
% Requires GenLouvain and cocoanCORE in the MATLAB user folder (userpath).
% Set the environment variable ANOREXIA_ROOT to the folder holding data/;
% otherwise the folder containing this repository is used.
% Region order matches 01c_combined_fc.py.

clear; clc; clear global;

% Path order matters: cocoanCORE has its own modularity.m that must NOT shadow
% GenLouvain's. Add GenLouvain LAST so its HelperFunctions/modularity.m wins.
code_dir    = fileparts(mfilename('fullpath'));
toolbox_dir = char(userpath);
addpath(genpath(fullfile(toolbox_dir, 'cocoanCORE')));   % calc_allegiance etc.
addpath(code_dir);
addpath(genpath(fullfile(toolbox_dir, 'GenLouvain')));

%% Parameters
gamma   = 1.0;
omega   = 1.0;
density = 0.05;
N_SUBJ  = 22;

N_CORTICAL  = 200;
N_SUBCORT   = 16;
N_COMBINED  = N_CORTICAL + N_SUBCORT;   % 216

PROJECT_ROOT = getenv('ANOREXIA_ROOT');
if isempty(PROJECT_ROOT)
    PROJECT_ROOT = fileparts(fileparts(fileparts(code_dir)));
end

out_dir        = fullfile(PROJECT_ROOT, 'data', 'analysis');
mlcd_subj_dir  = fullfile(out_dir, 'mlcd_subjs_combined_full44_d05');
subjs_mlcd_dir = fullfile(mlcd_subj_dir, 'subjs_mlcd');
for d = {mlcd_subj_dir, subjs_mlcd_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end
fc_dir = fullfile(out_dir, 'combined_subjs');

group_tags  = {'an_patients', 'hc_patients'};
group_names = {'Anorexia', 'Control'};

fprintf('===== COMBINED 216-ROI MLCD (parfor) - FULL 44, density=%.0f%% =====\n', density*100);
fprintf('N=%d (cortical=%d + subcortical=%d)\n', N_COMBINED, N_CORTICAL, N_SUBCORT);

%% Parallel pool; propagate paths to workers
if isempty(gcp('nocreate')), parpool('local'); end
pctRunOnAll(sprintf('addpath(genpath(''%s''))', fullfile(toolbox_dir, 'cocoanCORE')));
pctRunOnAll(sprintf('addpath(''%s'')', code_dir));
pctRunOnAll(sprintf('addpath(genpath(''%s''))', fullfile(toolbox_dir, 'GenLouvain')));

tic;

for grp = 1:2
    tag   = group_tags{grp};
    gname = group_names{grp};
    fprintf('\n===== GROUP: %s (%s) - %d subjects (parfor) =====\n', gname, tag, N_SUBJ);

    N_all_cell = cell(1, N_SUBJ);
    comm_cell  = cell(1, N_SUBJ);
    Qmod_cell  = cell(1, N_SUBJ);

    parfor subj_i = 1:N_SUBJ
        [N_all_cell{subj_i}, comm_cell{subj_i}, Qmod_cell{subj_i}] = ...
            process_subject(subj_i, tag, gname, fc_dir, mlcd_subj_dir, ...
                            gamma, omega, density, ...
                            N_COMBINED, N_CORTICAL, N_SUBCORT);
    end

    %% Group-level save (nodes x total windows)
    N_all_g = [N_all_cell{:}];
    if grp == 1
        g = struct('N_all_g_anorexia', N_all_g, ...
                   'Q_g_anorexia', [Qmod_cell{:}], ...
                   'comm_cons_all_g_anorexia', [comm_cell{:}], ...
                   'atlas', 'Schaefer200_Yeo7+TianS1_216ROIs', ...
                   'n_subj', N_SUBJ, 'density', density);
        save(fullfile(subjs_mlcd_dir, 'mlcd_combined_anorexia_wins_full44_d05.mat'), ...
            '-struct', 'g', '-v7.3');
        fprintf('Saved mlcd_combined_anorexia_wins_full44_d05.mat\n');
    else
        g = struct('N_all_g_control', N_all_g, ...
                   'Q_g_control', [Qmod_cell{:}], ...
                   'comm_cons_all_g_control', [comm_cell{:}], ...
                   'atlas', 'Schaefer200_Yeo7+TianS1_216ROIs', ...
                   'n_subj', N_SUBJ, 'density', density);
        save(fullfile(subjs_mlcd_dir, 'mlcd_combined_control_wins_full44_d05.mat'), ...
            '-struct', 'g', '-v7.3');
        fprintf('Saved mlcd_combined_control_wins_full44_d05.mat\n');
    end
end

t = toc;
if     t < 60,   fprintf('Done in %.1f sec\n', t);
elseif t < 3600, fprintf('Done in %dm %.1fs\n', floor(t/60), rem(t,60));
else,            fprintf('Done in %dh %dm\n',   floor(t/3600), floor(rem(t,3600)/60));
end

% ==========================================================================
function [N_all, comm, Q] = process_subject(subj_i, tag, gname, fc_dir, ...
        mlcd_subj_dir, gamma, omega, density, N_COMBINED, N_CORTICAL, N_SUBCORT)

    fc_file  = fullfile(fc_dir, sprintf('subj_fc_combined_%s_subj%02d.mat', tag, subj_i));
    if ~exist(fc_file, 'file')
        error('File not found: %s', fc_file);
    end
    S        = load(fc_file);
    var_name = sprintf('fc_combined_%s_subj%02d', tag, subj_i);
    fc_data  = S.(var_name);   % (W, 216, 216)

    sz = size(fc_data);
    if sz(2) == sz(3) && sz(1) ~= sz(2)
        fc_g = permute(fc_data, [2 3 1]);   % (W,N,N) -> (N,N,W)
    else
        fc_g = fc_data;
    end

    N = size(fc_g, 1);
    W = size(fc_g, 3);
    assert(N == N_COMBINED, 'Expected N=%d, got N=%d', N_COMBINED, N);

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

    %% MLCD
    A_subj = squeeze(num2cell(adj_g, [1 2]));
    mc = multilayer_community_detection_individual( ...
        A_subj(:).', 'ord', 'n_repeat', 100, 'thresh_type', 'max', ...
        'gamma', gamma, 'omega', omega);
    sd = mc{1};

    N_all = sd.multi_module_consensus;
    comm  = max(sd.multi_comm_consensus);
    Q     = mode(cell2mat(sd.multi_modQ));
    fprintf('  [%s] Subj %02d: communities=%d  Q=%.3f\n', tag, subj_i, comm, Q);

    %% Save per-subject
    out_s = struct('group', gname, 'tag', tag, 'subj_index_1based', subj_i, ...
        'gamma', gamma, 'omega', omega, 'density', density, ...
        'N_all_g', sd.multi_module_consensus, ...
        'comm_cons', comm, 'Qmod', Q, ...
        'atlas', 'Schaefer200_Yeo7+TianS1_216ROIs', ...
        'n_cortical', N_CORTICAL, 'n_subcortical', N_SUBCORT);
    save(fullfile(mlcd_subj_dir, ...
        sprintf('mlcd_combined_%s_subj%02d_d05.mat', tag, subj_i)), ...
        '-struct', 'out_s', '-v7.3');
end
