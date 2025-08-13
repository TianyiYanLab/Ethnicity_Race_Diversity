function KRR_stage3_prediction_weights(krr_result_dir,type, seeds_final,num_folds,outputdir)
% krr_result_dir = (['/home/7_KRR/AA_WA_CN/output']);

load(seeds_final)

for n = 1:length(seeds_final)
    seed =  seeds_final(n,1);
    workdir = ([krr_result_dir '/seed' num2str(seed) '_' type]);
    
    load([workdir '/seed' num2str(seed) '_setup_file.mat']);
    load([workdir '/final_result.mat'])
     
    for i = 1:num_folds

        %behavior
        y_pred_train_this_fold = y_pred_train{1,i}; % #subjects x #behaviors.
        %FC
        FC=feature_mat(:,:,logical(sub_fold(i).fold_index==0)); %#ROIs x #ROIs x #subjects.
        %compute Haufe-transformed regression weights 
        curr_cov(:,:,i)=squeeze(HCP_cov_FC_behavior(FC,y_pred_train_this_fold)); %#ROIs x #ROIs x #behaviors.
    
    end
    
     learned_cov(:,:,n) =  mean(curr_cov, 3);

end
     avg_learned_cov = squeeze(mean(learned_cov, 3));

     %% save
     mkdir(outputdir)
     outmat =([outputdir '/weights_' type '.mat']);
     save(outmat, 'learned_cov', 'avg_learned_cov', '-v7.3')