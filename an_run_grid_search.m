clear; clc;

out_dir = '/Users/ismaila/Documents/C-Codes/AnorexiaProject/data/analysis/modularity_gridsearch/';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% mat_an = '/Users/ismaila/Documents/C-Codes/AnorexiaProject/data/analysis/corr_an_patients_1tr_windows.mat';
% % mat_hc = '/Users/ismaila/Documents/C-Codes/AnorexiaProject/data/analysis/corr_hc_patients_1tr_windows.mat';

mat_an = '/Users/ismaila/Documents/C-Codes/AnorexiaProject/data/analysis/corr_an_patients_19tr_windows.mat';
% mat_hc = '/Users/ismaila/Documents/C-Codes/AnorexiaProject/data/analysis/corr_hc_patients_19tr_windows.mat';

gamma_vals = 0.5:0.1:2.5;
omega_vals = 0.1:0.1:1.5;

% Same settings for both datasets
use_cat = false;     % false=multiord, true=multicat
density = 0.3;      % [] to disable; 0.05 keeps top 5% -30% edges
do_parallel = true;

% --- Grid-search Anorexia ---
[best_an, Q_an] = gridsearch_multilayer_modularity( ...
    mat_an, out_dir, 'anorexia', ...
    'gamma_vals', gamma_vals, 'omega_vals', omega_vals, ...
    'use_cat', use_cat, 'density', density, 'do_parallel', do_parallel);

% % --- Grid-search Control ---
% [best_hc, Q_hc] = gridsearch_multilayer_modularity( ...
%     mat_hc, out_dir, 'control', ...
%     'gamma_vals', gamma_vals, 'omega_vals', omega_vals, ...
%     'use_cat', use_cat, 'density', density, 'do_parallel', do_parallel);

disp(best_an);
% disp(best_hc);
