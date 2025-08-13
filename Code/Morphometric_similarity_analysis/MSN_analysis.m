function MSN_analysis(sub_ID_RMS,AAls,WAls,MSN_AA,MSN_WA,nregs,unrestricted_csv,restricted_csv)
    %load data
    load(sub_ID_RMS);
    load(MSN_AA);meanMS_regional_AA = meanMS_regional;subj_MSN_5_AA=subj_MSN_5;
    load(MSN_WA);meanMS_regional_WA = meanMS_regional;subj_MSN_5_WA=subj_MSN_5;
    AAs=load(AAls);
    WAs=load(WAls);
    load('/home/user7/Data/5folds/aseg_stats_HCP.mat');
    allsub_aseg_stats = aseg_stats_HCP.data(:,1:2);
    clear mytstat_weighted mypval_weighted 

        %AA
        for ij = 1:length(AAs)
            sub  = AAs(ij);
            AAs_ICV(ij,1) = allsub_aseg_stats(logical(allsub_aseg_stats(:,1)==sub),2);%ICV
        end
        [~,Loc1] =ismember(AAs,sub_ID{2,1}(:,1));
        AAs_age = sub_ID{2,1}(Loc1,2);
        AAs_sex = sub_ID{2,1}(Loc1,3);
        AAs_RMS = sub_ID{2,1}(Loc1,4);
        AAs_MS = meanMS_regional_AA(Loc1,:);
        %WA
        for ij = 1:length(WAs)
            sub  = WAs(ij);
            WAs_ICV(ij,1) = allsub_aseg_stats(logical(allsub_aseg_stats(:,1)==sub),2);%ICV
        end
        [~,Loc2] =ismember(WAs,sub_ID{2,2}(:,1));
        WAs_age = sub_ID{2,2}(Loc2,2);
        WAs_sex = sub_ID{2,2}(Loc2,3);
        WAs_RMS = sub_ID{2,2}(Loc2,4);
        WAs_MS = meanMS_regional_WA(Loc2,:);
        
        % COV: age、sex、RMS、ICV
        age = [WAs_age;AAs_age];
        sex = [WAs_sex;AAs_sex];
        RMS = [WAs_RMS;AAs_RMS];
        ICV = [WAs_ICV;AAs_ICV];

        group(1:length(WAs),1) = 1;
        group(length(WAs)+1:length(AAs)+length(WAs),1) = 2;
        % assign weihgts to the two groups to account for the
        % unbalanced sample sizes between them
        weights = zeros(size(group));
        weights(group == 1) = 1 / length(WAs);  
        weights(group == 2) = 1 / length(AAs);     
        % normalize the weights
        weights = weights / sum(weights);

        dummy =[WAs_MS;AAs_MS];

        confounds=Lifestyle_Factor_Extraction(WAls,AAls,unrestricted_csv,restricted_csv);
        %confounds = #subjects * #covariates [education_z income_z psqi_z Substance_use_avg PA_avg Social_Score Personality_avg MentalHealth_avg];

        %% Compute regional differences in MS
        for region=1:nregs

          tbl = table(age, sex, group, RMS, ICV, confounds(:,1), confounds(:,2), ...
                dummy(:, region), 'VariableNames', {'age', 'sex', 'group','RMS','ICV','education', 'income','response'});
              
          tbl.sex = categorical(tbl.sex);
          tbl.group = categorical(tbl.group);

          % Weighted linear model
 
          lm_weighted = fitlm(tbl, 'response~age*sex+education*income+ICV+RMS+ group', 'Weights', weights);
          mytstat_weighted(region) = lm_weighted.Coefficients{4,3}; 
          mypval_weighted(region) = lm_weighted.Coefficients{4,4};

        end

          mytstat_weighted = transpose(mytstat_weighted);
          mypval_weighted = transpose(mypval_weighted);
          
          mypval_weighted_fdr = mafdr(mypval_weighted, 'BHFDR', true);


    % save 
    savemat = (['/home/user7/AA_WA721/regional_diff_v8.mat']);
    save(savemat,'mytstat_weighted','mypval_weighted','mypval_weighted_fdr');