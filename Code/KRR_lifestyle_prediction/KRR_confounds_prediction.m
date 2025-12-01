function KRR_confounds_prediction(allsubls,type,outdir_ori,workdir,seeds_num,num_folds,aseg_HCP)
    allsub  = load(allsubls);

    for n = 1:seeds_num
        seed =  n; 
        
        %param.sub_fold
        load([workdir '/seed' num2str(seed) '/no_relative_5_fold_sub_list.mat']); %% output of function 'CBIG_cross_validation_data_split'
        
        %param.feature_mat
        switch type
               case 'FC_indi'
                     load([workdir '/feature.mat']);
               case 'FC_schaefer400'            
               otherwise
                     error('Invalid input value');
        end
        
        %param.ker_param
        load([workdir '/ker_param.mat'])
        %param.lambda_set
        load([workdir '/lambda_set.mat'])
        %param.metric
        load([workdir '/metric.mat'])
        %param.num_inner_folds
        num_inner_folds = num_folds;
        %param.outdir
            switch type
               case 'FC_indi'
                    outdir =([outdir_ori '/seed' num2str(seed) '_full_cov_X']);
                    mkdir(outdir);
               case 'FC_schaefer400'
               otherwise
                    error('Invalid input value');
        end
        
        %param.outstem
        load([workdir '/outstem.mat'])
        %param.threshold_set
        threshold_set=[];
        %param.with_bias
        with_bias=1;
        
        %param.cov_X
        cov_X = extract_cov_X(allsub,aseg_HCP,'/home/user7/aData/HCP/restricted_S1200.csv');
        %param.covariates
        covariates='none';
    
        %param.y
        y = Y_Extraction(allsubls,'/home/user7/aData/HCP/unrestricted_S1200.csv', ...
            '/home/user7/aData/HCP/restricted_S1200.csv');
       
        savemat = ([outdir '/seed' num2str(seed) '_setup_file.mat']);
        save(savemat,'sub_fold','feature_mat','ker_param','lambda_set', ...
            'metric','num_inner_folds','outdir','outstem','threshold_set', ...
            'with_bias','y','covariates','cov_X');
        cd(outdir)
        CBIG_KRR_workflow([outdir '/seed' num2str(seed) '_setup_file.mat'],0);
    end
end


function cov_X = extract_cov_X(allsub,aseg_HCP,restricted_csv)
    sub_ID=load('/home/user7/aPopulation_differences/Data/sub_ID_RMS.mat') ;%sub_ID
    allsub_profiles = [sub_ID.sub_ID{2,1};sub_ID.sub_ID{2,2}];

    load(aseg_HCP);  % aseg_stats_HCP
    HCP_aseg_stats = aseg_stats_HCP.data(:,1:2);
    allsub_aseg_stats = HCP_aseg_stats;

    cov_X=[]; % [age gender RMS ICV Edu Income]

    for i = 1:length(allsub)
        sub  = allsub(i);
        cov_X(i,1:3) = allsub_profiles(logical(allsub_profiles(:,1)==sub),2:end); %[age gender RMS]
        [~, Educ_Income] = CBIG_parse_delimited_txtfile(restricted_csv, {'Gender','Race','Family_ID'}, ...
    {'SSAGA_Educ','SSAGA_Income'},'Subject', {char(num2str(sub))}, ',');        
        cov_X(i,4) = allsub_aseg_stats(logical(allsub_aseg_stats(:,1)==sub),2);%ICV
        cov_X(i,5) = Educ_Income(1);%Edu
        cov_X(i,6) = Educ_Income(2);%Income
     end

end


function Y = Y_Extraction(allsub_ls,unrestricted_csv,restricted_csv)
    
    %% Sleep:PSQI_Score
    
    % Reference:The interrelation of sleep and mental and physical health is 
    % anchored in grey-matter neuroanatomy and under genetic control
    [~, Psqi] = CBIG_parse_delimited_txtfile(unrestricted_csv, {'Gender',}, ...
        {'PSQI_Score',},'Subject', CBIG_text2cell(allsub_ls), ',');
    
        
    %% Substance Use
    [~,Substance_use] = CBIG_parse_delimited_txtfile(restricted_csv, {'Gender'},...
        {'Num_Days_Drank_7days','SSAGA_Alc_D4_Dp_Sx','SSAGA_Alc_D4_Ab_Dx','SSAGA_Alc_D4_Ab_Sx', ...
        'SSAGA_Alc_D4_Dp_Dx','Num_Days_Used_Any_Tobacco_7days','SSAGA_TB_Smoking_History','SSAGA_TB_Still_Smoking','SSAGA_Times_Used_Illicits', ...
        'SSAGA_Times_Used_Cocaine','SSAGA_Times_Used_Hallucinogens','SSAGA_Times_Used_Opiates','SSAGA_Times_Used_Sedatives','SSAGA_Times_Used_Stimulants', ...
        'SSAGA_Mj_Use','SSAGA_Mj_Times_Used'}, ...
        'Subject', CBIG_text2cell(allsub_ls), ',');
    
    rows_with_nan = any(isnan(Substance_use), 2);
    num_rows_with_nan = sum(rows_with_nan);
    fprintf('A total of %d rows contain at least one NaN value.\n', num_rows_with_nan);

    % mean value
    column_means = nanmean(Substance_use);
    % replace the NaN
    for i = 1:size(Substance_use, 2)
        nan_indices = isnan(Substance_use(:, i));
        Substance_use(nan_indices, i) = column_means(i);
    end
    
    Substance_use_avg = mean(Substance_use,2);
    
    %% Physical Activity
    [~, PA] = CBIG_parse_delimited_txtfile(unrestricted_csv, {'Gender',}, ...
        {'Endurance_Unadj', 'GaitSpeed_Comp', 'Dexterity_Unadj', 'Strength_Unadj'}, ...
        'Subject', CBIG_text2cell(allsub_ls), ',');
    PA_avg  = mean(PA,2);
    
    %% Social Relationships
    % Reference: HCP Manual
      [~, Social] = CBIG_parse_delimited_txtfile(unrestricted_csv, {'Gender',}, ...
        {'Friendship_Unadj', 'Loneliness_Unadj', 'PercHostil_Unadj', 'PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj'}, ...
        'Subject', CBIG_text2cell(allsub_ls), ',');
    
    % adjust the sign
        Social(:,2) = -Social(:,2); % Loneliness
        Social(:,3) = -Social(:,3); % Perceived Hostility
        Social(:,4) = -Social(:,4); % Perceived Rejection
        Social_Score = mean(Social, 2); % mean value
      
      
      confounders= {'Sleep','Substance Use','Physical Activity','Social Relationships'};        
      Y = [Psqi Substance_use_avg PA_avg Social_Score];

end



