function HCP_match_WA_with_AAfolds(subj_ls, FD_txt, bhvr_ls_rstr, bhvr_ls_unrstr, ...
    AA_fold_mat, outdir, outstem, restricted_csv, unrestricted_csv, match_ls)

% HCP_match_WA_with_AAfolds(subj_ls, FD_txt, DV_txt, bhvr_ls_rstr, bhvr_ls_unrstr, ...
%     AA_fold_mat, outdir, outstem, restricted_csv, unrestricted_csv, FS_csv, match_ls)
%
% Match white Americans with African Americans within each fold.
%
% Inputs:
%   - subj_ls
%     Full path of the list of all subjects that were involved in this project.
%   - FD_txt
%     Full path of a text file containing the framewise dispalcement for each of the 
%     subjects in `subj_ls`. Each row in this text file corresponds the FD of a subject.
%   - DV_txt
%     Full path of a text file contaning the DVARS for each of the subjects in `subj_ls`.
%     Each row in this text file corresponds to the DVARS of a subject.
%   - bhvr_ls_rstr
%     List of behavioral measures that were in the HCP restricted CSV.
%     Pass 'NONE' if all behavioral measures were in the unrestricted CSV.
%   - bhvr_ls_unrstr
%     List of behavioral measures that were in the HCP unrestricted CSV.
%   - AA_fold_mat
%     Full path of the .mat file containing the split AA folds.
%   - outdir
%     Absolute path of output directory.
%   - outstem
%     Stem of output filename, e.g. 'split_seed'.
%   - restricted_csv (optional)
%     Full path of the HCP restricted CSV file. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/restricted_hcp_data/...
%     RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv'
%   - unrestricted_csv (optional)
%     Full path of the HCP unrestricted CSV file. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/subject_measures/...
%     unrestricted_jingweili_12_7_2017_21_0_16_NEO_A_corrected.csv'
%   - FS_csv (optional)
%     Fullpath of the HCP FreeSurfer stats CSV file. This is used to calculate matching 
%     cost of intracranial volume. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/Morphometricity/Anat_Sim_Matrix/...
%     FS_jingweili_5_9_2017_2_2_24.csv'
%   - match_ls (optional)
%     Full path of the list of variables which need to be matched. If not passed in, the
%     matched variables include age, gender, FD. 
%     If 'NONE' is passed in, only behavioral score is matched.
%
% Author: Jingwei Li
% Modified by Ziteng Han (ziteng.han@outlook.com) on 2023/9/27

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
% if(~exist('FS_csv', 'var') || isempty(FS_csv))
%     FS_csv = fullfile(HCP_dir, 'scripts', 'Morphometricity', 'Anat_Sim_Matrix', ...
%         'FS_jingweili_5_9_2017_2_2_24.csv');
% end
if(~exist('match_ls', 'var') || isempty(match_ls))
    match_var = {'age', 'FD'};
elseif(strcmpi(match_ls, 'none'))
    match_var = [];
else
    match_var = CBIG_text2cell(match_ls);
end

%% read subject IDs and behavioral names
subjects = CBIG_text2cell(subj_ls);

if(~strcmpi(bhvr_ls_rstr, 'none'))
    [bhvr_rstr, n_rstr] = CBIG_text2cell(bhvr_ls_rstr);
else
    bhvr_rstr = [];
    n_rstr = 0;
end

[bhvr_unrstr, n_unrstr] = CBIG_text2cell(bhvr_ls_unrstr);
behaviors = [bhvr_rstr bhvr_unrstr];
nbehav = n_rstr + n_unrstr;

%% read demographics and behavioral scores 
[restricted_str, age_Educ] = CBIG_parse_delimited_txtfile(restricted_csv, {'Race', 'Family_ID'}, ...
    {'Age_in_Yrs', 'SSAGA_Educ'}, 'Subject', subjects, ',');

race = restricted_str(:,1);
iswhite = strcmp(race, 'White');
WA = subjects(iswhite);

families = restricted_str(:,2);
WA_fam = families(iswhite);
uniq_WA_fam = unique(WA_fam);
for f = 1:length(uniq_WA_fam)
    fam_ind = strcmp(WA_fam, uniq_WA_fam{f})==1;
    WA_perfam{f} = WA(fam_ind)';
    nWA_perfam(f) = length(find(fam_ind==1));
end

if(~strcmpi(bhvr_rstr, 'none'))
    [~, bhvr_rstr_val] = CBIG_parse_delimited_txtfile(restricted_csv, [], ...
        bhvr_rstr, 'Subject', subjects, ',');
else
    bhvr_rstr_val = [];
end
[sex, ~] = CBIG_parse_delimited_txtfile(unrestricted_csv, {'Gender'}, ...
    bhvr_unrstr, 'Subject', subjects, ',');

sex = strcmp(sex, 'M');
bhvr_val = [bhvr_rstr_val sex];

%[~, ICV] = CBIG_parse_delimited_txtfile(FS_csv, {}, {'FS_IntraCranial_Vol'}, 'Subject', subjects, ',');

FD = dlmread(FD_txt);
%DV = dlmread(DV_txt);

%% load pre-split AA folds
load(AA_fold_mat)
%%%%% for debuging
if(~exist('AA_fold', 'var'))
    AA_fold = Afr_fold;
end
%%%%%%%%%%%%%%%%%%%
nfolds = length(AA_fold.sub_perfold);
nAA = length(cat(1, AA_fold.sub_perfold{:}));
nsub_perfold = cellfun(@length, AA_fold.sub_perfold);
[~, ~, AA_ind] = intersect(cat(1, AA_fold.sub_perfold{:}), subjects, 'stable');

%% check intermediate output, so it can be rerun from the broken point
tmp_outdir = fullfile(outdir, 'tmp');
if(~exist(tmp_outdir, 'dir'))
    mkdir(tmp_outdir)
end
b_start = 1;
    for b = 1:nbehav
        tmp_out = fullfile(tmp_outdir, [outstem '_' behaviors{b} '.mat']);
        if(exist(tmp_out, 'file'))
            b_start = b+1;
            break
        end
    end
fprintf('b_start: %d\n', b_start)


%% start matching
rng('default');
ntries = 100;
flag = 1; % assume output exists
for b = b_start:nbehav
    fprintf('Behavior: %d. %s\n', b, behaviors{b});
    
    % load pre-saved temporary output of last run
    if(b>1 && b == b_start)
        load(fullfile(tmp_outdir, [outstem '_' behaviors{b-1} '.mat']));
    end
    fprintf('b: %d\n', b)
    
    metric = [];
    weigh_idx = [];
    for mv = 1:length(match_var)
        switch match_var{mv}
        case 'age'
            metric = [metric age_Educ(:,1)];
%         case 'gender'
%             metric = [metric sex];
        case 'FD'
            metric = [metric FD];

        otherwise
            error('Unknown matched variable.')
        end
    end
    metric = [metric bhvr_val(:,b)];
    if(~isempty(match_var))
        weigh_idx = [weigh_idx length(match_var)+1];
    end
    metric = metric - mean(metric,1);
    metric = metric ./ sqrt(sum(metric.^2,1));
    AA_metric = metric(AA_ind,:);
    AA_metric = reshape(AA_metric, [size(AA_metric,1) 1 size(AA_metric,2)]);
    
    highest_cost(b) = 1e16;
    best_assign{b} = [];
    cost_history{b} = [];
    
    rng(1);
    for r = 1:ntries
        fprintf('\tTrial: %d\t', r);
        nWA_sel = 0;
        
        % randomly shuffle WA families, pick the first families that the summed #WA is slightly bigger than #AA
        WAfam_randperm = randperm(length(uniq_WA_fam));
        nWA_randperm = cumsum(nWA_perfam(WAfam_randperm));
        thresh_ind = find(nWA_randperm <= nAA);
        thresh_ind = thresh_ind(end) + 10;
        
        for n = 1:thresh_ind
            WA_sel{n} = WA_perfam{WAfam_randperm(n)};
            nWA_sel_fam(n)= nWA_perfam(WAfam_randperm(n));
            nWA_sel = nWA_sel + nWA_sel_fam(n);
        end
        
        possib_selfam = 1:length(WA_sel);
        WA_perfold = cell(nfolds, 1);
        for fold = 1:nfolds
            % remove WA families which have AA members in current fold
            [~, curr_AA_ind] = intersect(subjects, cat(1, AA_fold.sub_perfold{setdiff(1:nfolds, fold)}), 'stable');
            curr_AA_fam = families(curr_AA_ind);
            curr_AA_fam = unique(curr_AA_fam);
            
            WA_overkap_fam = zeros(length(possib_selfam), 1);
            for caf = 1:length(curr_AA_fam)
                WA_overkap_fam = WA_overkap_fam + strcmp(uniq_WA_fam(WAfam_randperm(possib_selfam)), curr_AA_fam{caf});
            end
            
            if(any(WA_overkap_fam~=0))
                possib_selfam = setdiff(possib_selfam, find(WA_overkap_fam>0));
            end
            
            % for each picked WA family, assign it to an AA fold
            curr_WA = WA_sel{(possib_selfam(1))};
            possib_selfam(1) = [];
            while (length(curr_WA) < nsub_perfold(fold))
                n=0;
                while(~isempty(possib_selfam) )
                    n = n+1;
                    if( (fold<nfolds && length(curr_WA) + length(WA_sel{possib_selfam(n)}) <= nsub_perfold(fold)) ...
                            || (fold == nfolds))
                        curr_WA = [curr_WA; WA_sel{possib_selfam(n)}];
                        possib_selfam(n) = [];
                        break;
                    end
                    
                    if(n == length(possib_selfam))
                        [~,min_ind] = min(nWA_sel_fam(possib_selfam));
                        curr_WA = [curr_WA; WA_sel{possib_selfam(min_ind)}];
                        possib_selfam(min_ind) = [];
                        break;
                    end
                end
            end
            WA_perfold{fold} = curr_WA;
        end
        
        % Hungarian match
        feasible = 1;
        for fold = 1:nfolds
            if(fold==1)
                fold_ind = 1:nsub_perfold(1);
            else
                fold_ind = sum(nsub_perfold(1:fold-1))+1:sum(nsub_perfold(1:fold));
            end
            
            [~, ~, curr_WA_ind] = intersect(WA_perfold{fold}, subjects, 'stable');
            WA_metric = metric(curr_WA_ind, :);
            WA_metric = reshape(WA_metric, [1 size(WA_metric)]);
            cost_mat = bsxfun(@minus, AA_metric(fold_ind,:,:), WA_metric);
            cost_mat(:,:,weigh_idx) = cost_mat(:,:,weigh_idx) .*2;  % put on weights for Educ, ICV, and behavior
            cost_mat = sum(abs(cost_mat),3);
            [assign_WA, cost_fold(fold)] = munkres(cost_mat);
            if(any(assign_WA==0))
                feasible = 0;
                break
            else
                WA_perfold{fold} = WA_perfold{fold}(assign_WA);
            end
        end
        if(feasible == 0)
            continue
        end
        max_cost = max(cost_fold);
        if(max_cost<highest_cost(b))
            highest_cost(b) = max_cost;
            best_assign{b} = WA_perfold;
        end
        cost_history{b} = [cost_history{b}; highest_cost(b)];
        fprintf('Current cost: %f,  Best cost: %f\n', max_cost, highest_cost(b));
        
        clear WA_sel nWA_sel_fam nWA_sel cost_fold
    end
    
    % save intermediate results in temporary folder for the purpose of rerunning when something crashes
    fprintf('Behavior index: %d, Saving to %s\n', b, fullfile(tmp_outdir, [outstem '_' behaviors{b} '.mat']))
    save(fullfile(tmp_outdir, [outstem '_' behaviors{b} '.mat']), '-v7.3', '-regexp','^(?!(b)$).')
    if(b>1)
        system(['rm ' fullfile(tmp_outdir, [outstem '_' behaviors{b-1} '.mat'])])
    end
    
end

%% save final results and delete intermediate output
if(~exist(outdir, 'dir'))
    mkdir(outdir)
end
save(fullfile(outdir, [outstem '.mat']), 'highest_cost', 'best_assign', 'cost_history');

for b = 1:nbehav
    tmp_out = fullfile(outdir, 'tmp', [outstem '_' behaviors{b} '.mat'] );
    if(exist(tmp_out, 'file'))
        system(['rm ' tmp_out])
    end
end

end