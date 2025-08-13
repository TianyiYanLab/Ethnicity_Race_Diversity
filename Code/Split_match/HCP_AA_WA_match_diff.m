function HCP_AA_WA_match_diff(split_AAdir, split_WAdir, full_subj_ls, FD_txt, bhvr_ls_rstr, ...
    bhvr_ls_unrstr, num_seeds, mat_out, restricted_csv, unrestricted_csv, match_ls)

% HCP_AA_WA_match_diff(split_AAdir, split_WAdir, full_subj_ls, FD_txt, DV_txt, bhvr_ls_rstr, ...
%     bhvr_ls_unrstr, num_seeds, mat_out, restricted_csv, unrestricted_csv, FS_csv)
%
% Compare the demographic & behavioral distributions between selected African Americans and 
% white Americans using two-sample t test and K-S test. Usually K-S test was more conservative 
% (i.e. bigger p values) than t test, based on the author's experiences in this data. Hence the
% final significance was based on the t test.
%
% Inputs:
%   - split_AAdir
%     Full path of the directory storing the split folds of African Americans.
%     It is the output of `HCP_split_AA_rm_hardtomatch.m`.
%   - split_WAdir
%     Full path of the directory storing the white Americans that were matched to African Americans.
%     It is the output of `HCP_match_WA_with_AAfolds.sh`.
%   - full_subj_ls
%     Full path of the list of all subject IDs that were used for this project.
%   - FD_txt
%     Full path of the list of framewise displacement. Each row in this list corresponds to 
%     the FD of a subject in `full_subj_ls`.
%   - DV_txt
%     Full path of the list of DVARS. Each row in this list corresponds to the DVARS of a subject
%     in `full_subj_ls`.
%   - bhvr_ls_rstr
%     Full path of a list of behavioral measures that were contained in the HCP restricted CSV.
%     Pass 'NONE' if all behavioral measure you want to study were in the unrestricted CSV.
%   - bhvr_ls_unrstr
%     Full path of a list of behavioral measures that were contained in the HCP unrestricted CSV.
%   - num_seeds
%     Total number of random splitting repetitions. It needs to be consistent with the maximal 
%     seed used for `HCP_split_AA_rm_hardtomatch.m`.
%   - mat_out
%     Full path of output .mat file.
%   - restricted_csv (optional)
%     Full path of the HCP restricted CSV file. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/restricted_hcp_data/...
%     RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv'
%   - unrestricted_csv (optional)
%     Full path of the HCP unrestricted CSV file. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/subject_measures/...
%     unrestricted_jingweili_12_7_2017_21_0_16_NEO_A_corrected.csv'
%   - FS_csv (optional)
%     Full path of the HCP FreeSurfer stats CSV file. This is used to calculate matching 
%     cost of intracranial volume. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/Morphometricity/Anat_Sim_Matrix/...
%     FS_jingweili_5_9_2017_2_2_24.csv'
%   - match_ls (optional)
%     Full path of the list of confounds to be matched. If not passed in, the confounds to
%     be checked include age, gender, FD. If 'NONE' is passed in, 
%     only the difference in behavioral distributions will be checked.
%
% Author: Jingwei Li

%% set default parameters
HCP_dir = '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200';
if(~exist('restricted_csv', 'var') || isempty(restricted_csv))
    restricted_csv = fullfile(HCP_dir, 'scripts', 'restricted_hcp_data', ...
        'RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv');
end
if(~exist('unrestricted_csv', 'var') || isempty(unrestricted_csv))
    unrestricted_csv = fullfile(HCP_dir, 'scripts', 'subject_measures', ...
        'unrestricted_jingweili_12_7_2017_21_0_16_NEO_A_corrected.csv');
end

if(~exist('match_ls', 'var') || isempty(match_ls))
    match_var = {'age', 'FD',};
elseif(strcmpi(match_ls, 'none'))
    match_var = [];
else
    match_var = CBIG_text2cell(match_ls);
end

%% read demographics and behavioral scores
[subjects, ~] = CBIG_text2cell(full_subj_ls);
subjects = subjects';

if(~strcmpi(bhvr_ls_rstr, 'none'))
    [bhvr_rstr, n_rstr] = CBIG_text2cell(bhvr_ls_rstr);
else
    bhvr_rstr = [];
    n_rstr = 0;
end
[bhvr_unrstr, n_unrstr] = CBIG_text2cell(bhvr_ls_unrstr);
behaviors = [bhvr_rstr bhvr_unrstr];
nbehav = n_rstr + n_unrstr;

[~, age_Educ] = CBIG_parse_delimited_txtfile(restricted_csv, {'Race', 'Family_ID'}, ...
    {'Age_in_Yrs', 'SSAGA_Educ'}, 'Subject', subjects, ',');

if(~strcmpi(bhvr_ls_rstr, 'none'))
    [~, bhvr_rstr_val] = CBIG_parse_delimited_txtfile(restricted_csv, [], ...
        bhvr_rstr, 'Subject', subjects, ',');
else
    bhvr_rstr_val = [];
end
[sex, ~] = CBIG_parse_delimited_txtfile(unrestricted_csv, {'Gender'}, ...
    bhvr_unrstr, 'Subject', subjects, ',');

sex = strcmp(sex, 'M');
bhvr_val = [bhvr_rstr_val sex];

FD = dlmread(FD_txt);
%DV = dlmread(DV_txt);

metric = [];
weigh_idx = [];
for mv = 1:length(match_var)
    switch match_var{mv}
    case 'age'
        metric = [metric age_Educ(:,1)];
%     case 'gender'
%         metric = [metric sex];
    case 'FD'
        metric = [metric FD];
    otherwise
        error('Unknown variable name to be matched')
    end
end

metric = [metric bhvr_val];
no_match_idx = weigh_idx;
if(~isempty(match_var))
    weigh_idx = [weigh_idx length(match_var)+1];
end
metric = metric - mean(metric,1);
metric = metric ./ sqrt(sum(metric.^2,1));

%% statistical testing for each random seed. Correct for multiple comparisons across all seeds.
H_ks = []; p_ks = []; ksstat = [];
H_tt = []; p_tt = [];
meanA = []; meanW = []; varA = []; varW = [];
outlier_AA = {};
all_mAA = cell(nbehav,1);
all_mWA = cell(nbehav,1);
all_mAA_seed = cell(nbehav,num_seeds);
all_mWA_seed = cell(nbehav,num_seeds);
for seed = 1:num_seeds
    load(fullfile(split_AAdir, ['split_AA_5folds_seed' num2str(seed) '.mat']))
    load(fullfile(split_WAdir, ['match_WA_seed' num2str(seed) '.mat']))
    %%%%%%%%%%%%%%%% for my own debugging purpose
    if(~exist('AA_fold', 'var'))
        AA_fold = Afr_fold;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    H_ks_seed = []; p_ks_seed = []; ksstat_seed = [];
    H_tt_seed = []; p_tt_seed = [];
    meanA_seed = []; meanW_seed = []; varA_seed = []; varW_seed = [];
    
    for b = 1:nbehav
        curr_m = metric(:,[1:length(match_var) b+length(match_var)]);
        
        A_metric{b} = [];
        W_metric{b} = [];
        for f = 1:length(AA_fold.sub_perfold)
            [~,Aind] = intersect(subjects, AA_fold.sub_perfold{f});
            [~,Wind] = intersect(subjects, best_assign{b}{f});
            
            A_metric{b} = [A_metric{b}; curr_m(Aind, :)];
            W_metric{b} = [W_metric{b}; curr_m(Wind, :)];
            
            % the calculation of Hungarian matching cost should be the same as in `HCP_match_WA_with_AAfolds.m`
            cost_mat = bsxfun(@minus, reshape(curr_m(Aind,:), [length(Aind) 1 size(curr_m,2)]), ...
                reshape(curr_m(Wind,:), [1 size(curr_m(Wind,:))]));
            cost_mat(:,:,weigh_idx) = cost_mat(:,:,weigh_idx) .*2;
            cost_mat = sum(abs(cost_mat), 3);
            
            [assign, ~] = munkres(cost_mat);
            
            all_mAA{b} = [all_mAA{b}; curr_m(Aind, :)];
            all_mWA{b} = [all_mWA{b}; curr_m(Wind(assign), :)];
            all_mAA_seed{b,seed} = [all_mAA_seed{b,seed}; curr_m(Aind, :)];
            all_mWA_seed{b,seed} = [all_mWA_seed{b,seed}; curr_m(Wind(assign), :)];
            
            % Outlier AAs based on the matching cost
            % It helps to decide which AA to be excluded when using `HCP_split_AA_rm_hardtomatch.m` :
            % the AA IDs appeared as outliers for multiple behavioral measures can be considered to be ignored.
            pair_ind = sub2ind(size(cost_mat), 1:length(Aind), assign);
            outlier = isoutlier(cost_mat(pair_ind));
            fprintf('Outlier African: %s\n', subjects{Aind(outlier==1)})
            outlier_AA = [outlier_AA; subjects(Aind(outlier==1))];
        end
        
        curr_H_ks = []; curr_p_ks = []; curr_ksstats = [];
        curr_H_tt = []; curr_p_tt = [];
        curr_meanA = []; curr_meanW = []; curr_varA = []; curr_varW = [];
        for m = 1:size(curr_m,2)
            [curr_H_ks(m), curr_p_ks(m), curr_ksstats(m)] = kstest2(A_metric{b}(:,m), W_metric{b}(:,m));
            [curr_H_tt(m), curr_p_tt(m)] = ttest2(A_metric{b}(:,m), W_metric{b}(:,m), 'Vartype', 'unequal');
            
            curr_meanA(m) = mean(A_metric{b}(:,m));
            curr_meanW(m) = mean(W_metric{b}(:,m));
            curr_varA(m) = var(A_metric{b}(:,m));
            curr_varW(m) = var(W_metric{b}(:,m));
        end
        
        H_ks_seed = [H_ks_seed curr_H_ks];
        p_ks_seed = [p_ks_seed curr_p_ks];
        ksstat_seed = [ksstat_seed curr_ksstats];
        
        H_tt_seed = [H_tt_seed curr_H_tt];
        p_tt_seed = [p_tt_seed curr_p_tt];
        
        meanA_seed = [meanA_seed curr_meanA];
        meanW_seed = [meanW_seed curr_meanW];
        varA_seed = [varA_seed curr_varA];
        varW_seed = [varW_seed curr_varW];
    end
    H_ks = [H_ks; H_ks_seed];
    p_ks = [p_ks; p_ks_seed];
    ksstat = [ksstat; ksstat_seed];
    
    H_tt = [H_tt; H_tt_seed];
    p_tt = [p_tt; p_tt_seed];
    
    
    meanA = [meanA; meanA_seed];
    meanW = [meanW; meanW_seed];
    varA = [varA; varA_seed];
    varW = [varW; varW_seed];
end

alpha = 0.05;
H_ks_FDR = FDR(p_ks(:), alpha);
H_tt_FDR = FDR(p_tt(:), alpha);

%% check which (seed, behavior) combinations have significant difference 
%  in demographic and behavioral distributions between AA and WA
sig_bool = zeros(size(p_tt));
sig_bool(H_tt_FDR) = 1;
matched_mtx = zeros(num_seeds, nbehav);
for b = 1:nbehav
    curr_bool = sig_bool(:, ((b-1)*(length(match_var)+1)+1):(b*(length(match_var)+1)) );
    if(any(no_match_idx))
        curr_bool(:,no_match_idx) = [];    % Educ and ICV can't be matched in this data
    end
    sum_bool = sum(curr_bool, 2);
    matched = find(sum_bool == 0);
    if(~isempty(matched))
        for i = 1:length(matched)
            matched_mtx(matched(i),b) = 1;
            fprintf('Random seed %d for behavior %s has matched AA & WA in age, FD)\n', ...
                matched(i), behaviors{b});
        end
    end
end

save(mat_out, 'p_ks', 'H_ks', 'H_ks_FDR', 'p_tt', 'H_tt', 'H_tt_FDR', 'matched_mtx', 'outlier_AA')

%% save the statistics into a csv file
header = {'Random split #'};
for b = 1:nbehav
    header = [header {[behaviors{b} ': Age'], ...
        [behaviors{b} ': FD'], behaviors{b}}];
end

data = [meanA(1,:); varA(1,:)];
data(3+(0:2:(num_seeds-1)*2),:) = meanW; 
data(3+(1:2:(num_seeds-1)*2+1),:) = varW;

for row = 1:size(data,1)
    for col = 1:size(data,2)
        data_rows{col}{row} = num2str(data(row, col));
    end
end

data_rows = [cell(1), data_rows];
data_rows{1}{1} = 'All selected African Am (mean)';
data_rows{1}{2} = 'All selected African Am (var)';
for seed = 1:num_seeds
    data_rows{1}{3+(seed-1)*2} = sprintf('Seed %d: matched White (mean)', seed);
    data_rows{1}{4+(seed-1)*2} = sprintf('Seed %d: matched White (var)', seed);
end
CBIG_write_delimited_txtfile(fullfile(split_WAdir, 'mean_var_AA_vs_WA.csv'), header, data_rows);


end
