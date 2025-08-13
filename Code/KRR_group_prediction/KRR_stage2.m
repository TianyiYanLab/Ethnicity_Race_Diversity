function KRR_stage2(outdir_ori,workdir,seeds_final,num_folds,num_sub, ...
    sub_ID_profile,aseg_HCP)

%% add the scripts from CBIG to path: https://github.com/ThomasYeoLab/CBIG/tree/master/utilities/matlab/predictive_models/KernelRidgeRegression

load(seeds_final)
for n = 1:length(seeds_final) 
    seed =  seeds_final(n,1);
    
    %param.sub_fold
    load([workdir '/seed' num2str(seed) '_sub_fold_v3.mat']);
    %param.feature_mat
    load([workdir '/seed' num2str(seed) '_feature_v3.mat']);
    %param.ker_param
    load([workdir '/ker_param.mat'])
    %param.lambda_set
    load([workdir '/lambda_set.mat'])
    %param.metric
    load([workdir '/metric.mat'])
    %param.num_inner_folds
    num_inner_folds = num_folds;
    %param.outdir
    outdir =([outdir_ori '/seed' num2str(seed) '_full_cov_X']);
    mkdir(outdir);
    %param.outstem
    load([workdir '/outstem.mat'])
    %param.threshold_set
    threshold_set=[];
    %param.with_bias
    with_bias=1;
    y(1:num_sub/2,1)=0;%WA
    y(num_sub/2+1:num_sub,1)=1;%AA
    %param.covariates
    covariates=[];
    %param.cov_X
    cov_X = extract_cov_X_2groups_v3(seed,num_folds,sub_ID_profile,aseg_HCP,'/home/user7/aData/HCP/restricted_S1200.csv'); % #subjects × #covariates [ICV Edu Income]
    
    
    savemat = ([outdir '/seed' num2str(seed) '_setup_file.mat']);

    save(savemat,'sub_fold','feature_mat','ker_param','lambda_set', ...
        'metric','num_inner_folds','outdir','outstem','threshold_set', ...
        'with_bias','y','covariates','cov_X');
    cd(outdir)
    features([outdir '/seed' num2str(seed) '_setup_file.mat'],0);
end
