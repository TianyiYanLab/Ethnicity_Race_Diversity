function HCP_split_AA_rm_hardtomatch(full_subj_ls, rm_subj_ls, num_folds, seed, outmat, restricted_csv)

% HCP_split_AA_rm_hardtomatch(full_subj_ls, rm_subj_ls, num_folds, seed, outmat, restricted_csv)
%
% Split the African Amerians into `num_folds` folds for cross-validation. The subjects that are 
% hard to be matched with White Americans can be ignored by passing their IDs via `rm_subj_ls`. 
% The whole procedure can be repeated randomly by giving different random seed `seed`.
%
% Inputs:
%   - full_subj_ls
%     String, the absolute path of the full subject list. Each row in the list corresponds to 
%     one subject ID.
%   - rm_subj_ls
%     String, the absolute path of a list of subjects that are hard to match with White Americans.
%     Note that these subjects will still be integrated with other races/ethnicities and participant
%     the predictive model training. 
%     If you don't have subjects to remove (e.g. it's the first time of matching that you haven't 
%     known which subjects are hard to be matched), pass 'NONE' or ''.
%   - num_folds
%     Scalar, the number of folds to be split.
%   - seed
%     Scalar, the random seed used for split.
%   - outmat
%     String, the absolute path for the output .mat file.
%   - restricted_csv (optional)
%     String, the absolute path of the restricted CSV file of the HCP dataset. Default (on CSC HPC):
%     '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/restricted_hcp_data/...
%     RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv'
%
% Author: Jingwei Li

%% parse input
if(~exist('restricted_csv', 'var') || isempty(restricted_csv))
    restricted_csv = fullfile('/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/restricted_hcp_data', ...
        'RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv');
end

if(isempty(rm_subj_ls) || strcmpi(rm_subj_ls, 'none'))
    rm_subj = [];
else
    rm_subj = CBIG_text2cell(rm_subj_ls);
end

outdir = fileparts(outmat);
if(~exist(outdir, 'dir'))
    mkdir(outdir)
end

[full_subj, nsub] = CBIG_text2cell(full_subj_ls);
[y] = CBIG_parse_delimited_txtfile(restricted_csv, {'Race', 'Family_ID'}, ...
    [], 'Subject', full_subj, ',');

Race = y(:,1);
FamilyID = y(:,2);

uniq_race = unique(Race);
uniq_fam = unique(FamilyID);

%%
rm_subj_ind = [];
for s = 1:length(rm_subj)
    rm_subj_ind = [rm_subj_ind; find(strcmp(full_subj, rm_subj{s})==1)];
end

AA_ind = find(strcmp( Race, 'Black or African Am.')==1);
AA_ind = setdiff(AA_ind, rm_subj_ind);
AA = full_subj(AA_ind);
AA_families = FamilyID(AA_ind);
uniq_AA_fam = unique(AA_families);

% get familiy sizes
sub_perfam = cell(length(uniq_AA_fam), 1);
nsub_perfam = zeros(length(uniq_AA_fam), 1);
for i = 1:length(uniq_AA_fam)
    fam_ind = strcmp(AA_families, uniq_AA_fam{i})==1;
    sub_perfam{i} = AA(fam_ind)';
    nsub_perfam(i) = length(find(fam_ind==1));
end
uniq_nsub = sort(unique(nsub_perfam), 'descend');

rng('default'); rng(seed);

% put families with the same size together. The ordering within each family-size group is randomized.
fam_n = cell(length(uniq_nsub), 1);
curr_order = cell(length(uniq_nsub), 1);
curr_order_all = [];
for n = 1:length(uniq_nsub)
    fam_n{n} = find(nsub_perfam == uniq_nsub(n));
    curr_order{n} = datasample(fam_n{n}, length(fam_n{n}), 'replace', false);
    curr_order_all = [curr_order_all; curr_order{n}];
end

%% assign families into folds, based on the size and the ordering within each size group
fam_fold_corr = zeros(length(uniq_AA_fam), num_folds);
eql_nsub = ceil(length(AA) / num_folds);
mod_nsub = num_folds - mod(length(AA), num_folds);
sub_perfold = cell(num_folds, 1);
fam = 1;
while fam <= length(curr_order_all)
    curr_flag = zeros(num_folds, 1);     % for each fold, was it checked for current family
    assigned = 0;
    while(assigned == 0)
        for fold = 1:num_folds
            if(prod(curr_flag)==0)
                % if this is the first time of for loop
                curr_flag(fold) = 1;
                curr_length = cellfun(@length, sub_perfold);
                if(length(sub_perfold{fold}) + nsub_perfam(curr_order_all(fam)) <= eql_nsub)
                    
                    if(length(sub_perfold{fold}) + nsub_perfam(curr_order_all(fam)) < eql_nsub || ...
                            (length(sub_perfold{fold}) + nsub_perfam(curr_order_all(fam)) == eql_nsub ...
                            &&length(find(curr_length<eql_nsub)) > mod_nsub))
                        sub_perfold{fold} = [sub_perfold{fold}; sub_perfam{curr_order_all(fam)}];
                        assigned = 1;
                        fam_fold_corr(curr_order_all(fam), fold) = 1;
                        break
                    else
                        continue
                    end
                end
            else
                % if this is the second time of for loop
                if(length(sub_perfold{fold}) <= eql_nsub)
                    sub_perfold{fold} = [sub_perfold{fold}; sub_perfam{curr_order_all(fam)}];
                    assigned = 1;
                    fam_fold_corr(curr_order_all(fam), fold) = 1;
                    break
                end
            end
        end
    end
    fam = fam + 1;
end

%% save
AA_fold.sub_perfold = sub_perfold;
AA_fold.fam_fold_corr = fam_fold_corr;
save(outmat, 'AA_fold');

end