function [best, Q_table] = gridsearch_multilayer_modularity(mat_path, out_dir, tag, varargin)
%GRIDSEARCH_MULTILAYER_MODULARITY
% Grid search over gamma and omega for ONE dataset (.mat with corr_* variable).
%
% Required:
%   mat_path : path to .mat file containing corr_* variable
%   out_dir  : output folder
%   tag      : string label for saving outputs (e.g., 'anorexia' or 'control')
%
% Name-Value options:
%   'gamma_vals'  (default 0.5:0.1:2.5)
%   'omega_vals'  (default 0.1:0.1:1.5)
%   'use_cat'     (default false)  false=multiord, true=multicat
%   'density'     (default [])     e.g., 0.05 for top 5% edges per window
%   'do_parallel' (default true)

% -------------------------
% Parse inputs
% -------------------------
p = inputParser;
p.addParameter('gamma_vals', 0.5:0.1:2.5);
p.addParameter('omega_vals', 0.1:0.1:1.5);
p.addParameter('use_cat', false);
p.addParameter('density', []);
p.addParameter('do_parallel', true);
p.parse(varargin{:});

gamma_vals  = p.Results.gamma_vals;
omega_vals  = p.Results.omega_vals;
use_cat     = p.Results.use_cat;
density     = p.Results.density;
do_parallel = p.Results.do_parallel;

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('\n=== Grid search: %s ===\n', tag);
fprintf('MAT file: %s\n', mat_path);

% -------------------------
% Load corr_* variable (memory-efficient: only one variable)
% -------------------------
S = load(mat_path);
fn = fieldnames(S);
corr_fields = fn(startsWith(fn,'corr_') & ~strcmp(fn,'__meta__'));
if isempty(corr_fields)
    error('No variables named corr_* found in %s', mat_path);
end
corr_data = S.(corr_fields{1});
clear S

if ndims(corr_data) ~= 3
    error('corr_data must be 3D. Got size [%s].', num2str(size(corr_data)));
end

% Convert to (N,N,T) without duplicating if already that shape
sz = size(corr_data);
if sz(1) == sz(2)
    corr_stack = corr_data;               % (N,N,T)
else
    corr_stack = permute(corr_data,[2 3 1]);  % (N,N,T)
end
clear corr_data

% Use single precision to reduce memory
corr_stack = single(corr_stack);

N   = size(corr_stack,1);
T_g = size(corr_stack,3);

fprintf('Using corr field: %s\n', corr_fields{1});
fprintf('Interpreted as: N=%d regions, T=%d layers\n', N, T_g);

% -------------------------
% Clean in-place
% -------------------------
corr_stack = 0.5 * (corr_stack + permute(corr_stack,[2 1 3])); % symmetrize
maskI = repmat(eye(N,'logical'),1,1,T_g);
corr_stack(maskI) = 0;                                         % remove diagonal
corr_stack(~isfinite(corr_stack)) = 0;
corr_stack(corr_stack < 0) = 0;                                % unsigned

% -------------------------
% Optional proportional thresholding (in-place per layer)
% -------------------------
if ~isempty(density)
    if density <= 0 || density > 1
        error('density must be in (0,1].');
    end
    fprintf('Applying proportional thresholding: density = %.3f\n', density);
    corr_stack = proportional_threshold_3d_inplace(corr_stack, density);
end

% Build multilayer cell array A_g (T_g x 1), each NxN
A_g = squeeze(num2cell(corr_stack, [1 2]));
clear corr_stack maskI

% -------------------------
% Grid setup
% -------------------------
[GammaGrid, OmegaGrid] = meshgrid(gamma_vals, omega_vals);
G = numel(GammaGrid);

Q_g_vec   = nan(G,1);
comm_vec  = nan(G,1);
gamma_vec = nan(G,1);
omega_vec = nan(G,1);

% Try to use parfor safely
use_parfor = false;
if do_parallel
    try
        if isempty(gcp('nocreate')), parpool('local'); end
        use_parfor = true;
    catch
        use_parfor = false;
    end
end

fprintf('Running grid search: %d gamma × %d omega = %d points\n', ...
    numel(gamma_vals), numel(omega_vals), G);

% -------------------------
% Run grid search
% -------------------------
if use_parfor
    parfor idx = 1:G
        [Q_g_vec(idx), comm_vec(idx), gamma_vec(idx), omega_vec(idx)] = ...
            one_point(A_g, N, T_g, GammaGrid(idx), OmegaGrid(idx), use_cat);
    end
else
    for idx = 1:G
        [Q_g_vec(idx), comm_vec(idx), gamma_vec(idx), omega_vec(idx)] = ...
            one_point(A_g, N, T_g, GammaGrid(idx), OmegaGrid(idx), use_cat);

        if ~isnan(Q_g_vec(idx))
            fprintf('gamma=%.2f, omega=%.2f, Q=%.4f, comm=%d\n', ...
                gamma_vec(idx), omega_vec(idx), Q_g_vec(idx), comm_vec(idx));
        end
    end
end

Q_table = table(gamma_vec, omega_vec, Q_g_vec, comm_vec, ...
    'VariableNames', {'gamma','omega','Q_g','num_communities'});
Q_table = Q_table(~isnan(Q_table.Q_g), :);

% Best
[maxQ, maxIdx] = max(Q_table.Q_g);
best = struct();
best.gamma = Q_table.gamma(maxIdx);
best.omega = Q_table.omega(maxIdx);
best.Q     = maxQ;
best.num_communities = Q_table.num_communities(maxIdx);

fprintf('\nBEST (%s): Q=%.4f at gamma=%.3f, omega=%.3f | comm=%d\n', ...
    tag, best.Q, best.gamma, best.omega, best.num_communities);

% -------------------------
% Save outputs
% -------------------------
xlsx_path = fullfile(out_dir, sprintf('%s_modularity_gridsearch.xlsx', tag));
mat_out   = fullfile(out_dir, sprintf('%s_modularity_gridsearch.mat', tag));
png_path  = fullfile(out_dir, sprintf('%s_modularity_gridsearch.png', tag));

writetable(Q_table, xlsx_path);
save(mat_out, 'Q_table', 'best', 'gamma_vals', 'omega_vals', 'use_cat', 'density', 'mat_path');

f = figure('Color','w');
scatter(Q_table.gamma, Q_table.omega, 50, Q_table.Q_g, 'filled');
colorbar;
xlabel('\gamma'); ylabel('\omega');
title(sprintf('%s: modularity Q across \\gamma/\\omega (best=%.4f)', tag, best.Q));
grid on;
exportgraphics(f, png_path, 'Resolution', 200);
close(f);

fprintf('Saved: %s\nSaved: %s\nSaved: %s\n', xlsx_path, mat_out, png_path);

end

% =====================================================================
% One grid point
% =====================================================================
function [Q_out, comm_out, gamma_out, omega_out] = one_point(A_g, N, T_g, gamma, omega, use_cat)
Q_out = nan; comm_out = nan;
gamma_out = gamma; omega_out = omega;

try
    if use_cat
        [B, twom] = multicat(A_g, gamma, omega);
        post_fn = @(S) postprocess_categorical_multilayer(S, T_g);
    else
        [B, twom] = multiord(A_g, gamma, omega);
        post_fn = @(S) postprocess_ordinal_multilayer(S, T_g);
    end

    [S, Q] = iterated_genlouvain(B, 10000, 0, 1, 'moverandw', [], post_fn);
    Q = Q / twom;

    S = reshape(S, N, T_g);
    comm_out = max(S, [], 'all');
    Q_out = Q;
catch
    % keep NaNs on failure
end
end

% =====================================================================
% Proportional thresholding (in-place style) - weighted/unsigned
% =====================================================================
function A3 = proportional_threshold_3d_inplace(A3, density)
N = size(A3,1);
T = size(A3,3);

upmask = triu(true(N), 1);
[ri, ci] = find(upmask);
M = numel(ri);
K = max(0, round(density * M));

for t = 1:T
    Ak = A3(:,:,t);
    w  = Ak(upmask);
    pos_idx = find(w > 0);

    if isempty(pos_idx) || K == 0
        A3(:,:,t) = 0;
        continue;
    end

    Kt = min(K, numel(pos_idx));
    [~, ord] = sort(w(pos_idx), 'descend');
    keep_pos = pos_idx(ord(1:Kt));

    Akeep = zeros(N, 'single');
    lin = sub2ind([N N], ri(keep_pos), ci(keep_pos));
    Akeep(lin) = w(keep_pos);
    Akeep = Akeep + Akeep.';
    A3(:,:,t) = Akeep;
end
end
