function covariance = HCP_cov_FC_behavior(FC, behavior)

% covariance = ABCD_cov_FC_behavior(FC, behavior)
%
% Given a single FC matrix and a single behavior, calculate the covariance between them.
% If the FC matrix and the behavior vector are from the training set, and the behavior
% vector is the predicted scores, then the covariace represents the brain-behavior 
% association learned by the model.
% If the FC matrix and the behavior vector are from the test set, and the behavior vector 
% is the raw scores, then the covariance represents the real brain-behavior assocition 
% in the test set (perhaps for a specific subgroup, e.g. AA).
%
% Inputs:
%   - FC
%     A matrix of functional connectivity, assumed to be symmetric. 
%     Dimension: #ROIs x #ROIs x #subjects.
%   - behavior
%     A vector or matrix of a behavioral measures. Dimension: #subjects x #behaviors.
%
% Output:
%   - covariance
%     A matrix of covariance between FC and behavior across subjects.
%     Dimension: #ROIs x #ROIs x #behaviors.

%% check dimensions
assert(size(FC,3)==size(behavior,1), '#subjects in FC and behavior not consistent.')

%% reshape FC matrix, select lower-triangular entries
sz = size(FC);
tril_mtx = ones(sz(1:2));
tril_mtx = tril(tril_mtx, -1);
FC = reshape(FC, prod(sz(1:2)), sz(3));
FC = FC(tril_mtx(:)==1, :);   % dim: #edges x #subjects

%% demean across subjects
FC = bsxfun(@minus, FC, mean(FC,2));
behavior = bsxfun(@minus, behavior, mean(behavior));

%% compute covariance
covariance = FC * behavior ./ size(behavior,1);   % dim: #edges x #behaviors

%% reshape covariance matrix
cov_3D = zeros(sz(1), sz(2), size(behavior,2));
cov_3D = reshape(cov_3D, prod(sz(1:2)), size(behavior,2));
cov_3D(tril_mtx(:)==1, :) = covariance;
cov_3D = reshape(cov_3D, sz(1), sz(2), size(behavior,2));
covariance = cov_3D + permute(cov_3D, [2 1 3]);

end