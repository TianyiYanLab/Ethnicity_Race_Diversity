function KRR_stage1_setup_params(outdir,seeds_final,num_folds,num_sub)

load(seeds_final)
for n = 1:length(seeds_final)
    seed =  seeds_final(n,1);
    folddir = ('/home/user7/Data/5folds');
    
    %param.sub_fold
    fold_list = cell(num_folds,1);
    %AA
    load([folddir '/split_AA_5folds_seed' num2str(seed) '.mat']); %AA_fold
    AA_fold = AA_fold.sub_perfold;
    
    %WA
    load([folddir '/match_WA_with_AAfolds/match_WA_seed' num2str(seed) '.mat']); %WA_fold
    WA_fold = best_assign{1};

    allsub = [];
    for t = 1:num_folds
        allsub = [allsub;WA_fold{t}];
    end
    for t = 1:num_folds
        allsub = [allsub;AA_fold{t}];
    end

    for i = 1:num_folds
            fold_list{i} = [WA_fold{i};AA_fold{i}];
            fold_index{i} = zeros(num_sub,1);
            for j = 1:size(fold_list{i},1)
                fold_index{i} = fold_index{i} + (strcmp(allsub, fold_list{i}{j}));
            end
            fold_index{i} = logical(fold_index{i});
            
    end
    sub_fold = struct('subject_list', fold_list, 'fold_index', fold_index');
    savemat =([outdir '/seed' num2str(seed) '_sub_fold_v3.mat']);
    save(savemat,'sub_fold');

    %param.feature_mat
    feature_mat=[];
    for s = 1:length(allsub)
        sub = str2num(allsub{s});
        corrData=load(['/home/user7/results/all_corr_mat/' num2str(sub) '_corrmap_avg.mat']);      
        feature_mat(:,:,s)=corrData.corr;
    end
    savemat =([outdir '/seed' num2str(seed) '_feature_v3.mat']);
    save(savemat,'feature_mat');

end
