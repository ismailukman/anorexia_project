function N = multilayer_community_detection_individual(A, coupling_type, varargin)

% This function detects multi-layer community structure within individuals.
%
%
% :Usage:
% ::
%     N = multilayer_community_detection_individual(A, coupling_type, varargin)
%
%
% :Input:
% ::
%   - A                  Adjacency matrix. Data type should be cell.
%                        Rows indicate subjects.
%                        Columns indicate runs.
%                        (e.g., 19 subjects, 4 runs => 19 X 4 cell)
%   - coupling_type      Type of coupling. Either 'ord' or 'cat'.
%                        'ord': Runs are time-continuous.
%                        'cat': Runs are not time-continuous.
%
%
% :Optional Input:
%
%   - n_repeat           The number of repetition for getting consensus
%                        community. (default: 100)
%   - thresh_type        Type of thresholding for consensus community
%                        detection. See 'calc_allegiance' function.
%                        (default: 'max')
%   - gamma              Intra-layer resolution parameter. (default: 1)
%   - omega              Inter-layer coupling parameter. (default: 1)
%
% :Output:
% ::   
%   - N                  Multi-layer community structure within individuals.
%
%
% :Example:
% ::
%
%
%     Author and copyright information:
%
%     Copyright (C) May 2020  Jae-Joong Lee
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
% ..

%%1. Define default variables for multilayer community detection

n_repeat = 100; %Number of iterations for the consensus community detection
thresh_type = 'max'; %Thresholding method for the allegiance matrix
gamma = 1; %Resolution parameter for modularity optimization
omega = 1; %Inter-layer coupling strength in multilayer networks
allegiance_mat_out = 0; %Initialize allegiance matrix output variable

%2. Process optional input arguments (varargin) and update parameters if provided
for i = 1:length(varargin)
    if ischar(varargin{i})
        switch varargin{i}
            case {'n_repeat'}
                n_repeat = varargin{i+1};
            case {'thresh_type'}
                thresh_type = varargin{i+1};
            case {'gamma'}
                gamma = varargin{i+1};
            case {'omega'}
                omega = varargin{i+1};
        end
    end
end

%3. Extract key matrix dimensions
n_subj = size(A,1);
n_run = size(A,2);
n_node = size(A{1,1}, 1);

fprintf('Multilayer community detection... \n');
fprintf('n_subj is %d, n_run is %d\n and n_node is %d \n', n_subj, n_run, n_node);
fprintf('gamma is %.2f, omega is %.2f, coupling_type is %s \n', gamma, omega, coupling_type);

%4. Iterate through each subject and construct the multilayer modularity matrix
for subj_i = 1:n_subj
    
    fprintf('Working on SUBJECT %.3d ... \n', subj_i);

    switch coupling_type
        case 'ord' %Runs are time-continuous.
            [B,twomu] = multiord(A(subj_i,:),gamma,omega);
            fprintf('multiord returns: B [%d x %d], twomu = %.2f\n', size(B,1), size(B,2), twomu);
            PP = @(S) postprocess_ordinal_multilayer(S, n_run);
            % multiord returns B, which is a sparse modularity matrix (often huge), and twomu, a scalar normalization constant.
        case 'cat' %Runs are not time-continuous.
            [B,twomu] = multicat(A(subj_i,:),gamma,omega);
            PP = @(S) postprocess_categorical_multilayer(S, n_run);
        otherwise
            error('Coupling type should be either ''ord'' or ''cat'' !!');
    end
    

     %5. Run multilayer Louvain community detection for `n_repeat` iterations
    for rep_i = 1:n_repeat
        [S, Q, n_iter] = iterated_genlouvain(B, 10000, 0, 1, 'moverandw', [], PP);
        N{subj_i}.multi_modQ{rep_i} = Q/twomu; % modularity value
        N{subj_i}.multi_modmm{rep_i} = twomu; % normalizing factor
        N{subj_i}.multi_niter{rep_i} = n_iter; % number of iterations
        N{subj_i}.multi_module{rep_i} = reshape(S, n_node, n_run); % community assignments
        N{subj_i}.multi_comm{rep_i} = max(S, [], 'all'); % Number of communities
    end

    %6. Iterate through each run for consensus detection
    for run_i = 1:n_run
        for cons_iter = 1:10
            
            %6a. Initialize consensus community matrix with detected communities
            if cons_iter == 1
                cons_comm = cat(3, N{subj_i}.multi_module{:});
                cons_comm = squeeze(cons_comm(:,run_i,:));
            end

            %6b. Compute allegiance matrix and determine permutation threshold
            [allegiance_mat, perm_max(cons_iter, 1)] = calc_allegiance(cons_comm, 'threshold', thresh_type);


            %6c. Identify nodes that participate in communities
            wh_conn = any(allegiance_mat ~= 0); % retuns a logical row vector
            % wh_conn(i) = true if node i has at least one non-zero connection.
            allegiance_mat = allegiance_mat(wh_conn, wh_conn);
            fprintf('allegiance_mat: [%d x %d] \n', size(allegiance_mat,1), size(allegiance_mat,2));
            cons_comm = NaN(n_node, n_repeat);  % Initialize community matrix
            
            %run checks on A and g length
            % fprintf('A, g length check.');
            % disp(size(allegiance_mat, 1));
            % fprintf('check [A(i, j)] %d\n', allegiance_mat_out); % ?
            % disp(length(gamma));
            
            allegiance_mat_out = allegiance_mat; % Store allegiance matrix
            [B, ~] = modularity(allegiance_mat, gamma); % Compute modularity matrix
            fprintf('modularity returns: B [%d x %d] \n', size(B,1), size(B,2));
            allegiance_mat = []; % Clear allegiance matrix to save memory
            
            %6d. Perform Louvain community detection on allegiance matrix
            for rep_i = 1:n_repeat
                [S, ~] = genlouvain(B, 10000, 0);
                cons_comm(wh_conn, rep_i) = S;
            end
            B = []; % Clear modularity matrix


            %6e. Check if all 100 assignments are identical (consensus reached)
            if all(all(diff(cons_comm(wh_conn,:), [], 2) == 0))
                N{subj_i}.multi_module_consensus(:,run_i) = cons_comm(:,1);% Store consensus communities
                N{subj_i}.multi_niter_consensus(:,run_i) = cons_iter; % Store number of iterations taken
                N{subj_i}.multi_permmax_consensus{run_i} = perm_max; % Store permutation results
                N{subj_i}.multi_comm_consensus(run_i) = max(cons_comm(:,1), [], 'all'); % Number of communities in consensus
                perm_max = []; % Clear permutation storage
                break; % Exit loop if consensus is reached
            end

        end
    end
    
    %6f. Match community labels across runs to ensure consistency
    N{subj_i}.multi_module_consensus_sorted = match_community_affiliation(N{subj_i}.multi_module_consensus);
    
end

end