function Regional_differences_graidient_2g_AA_WA721_v2_majorRe_edu(AAls,WAls,HCPYA_data,nregs,name)
    %load data
   
    AAs=load(AAls);
    WAs=load(WAls);
    data_src = readtable(HCPYA_data);
    clear mytstat_weighted mypval_weighted 

    age = table2array(data_src(:,1));
    sex = table2array(data_src(:,2));
    RMS = table2array(data_src(:,3));
    ICV = table2array(data_src(:,4));
    group= table2array(data_src(:,5));
    education = table2array(data_src(:,6));
    income = table2array(data_src(:,7));
    sleep = table2array(data_src(:,8));
    substance = table2array(data_src(:,9));
    activity = table2array(data_src(:,10));
    social = table2array(data_src(:,11));

    gradient_value = table2array(data_src(:,13:412));

    group(1:length(WAs),1) = 1;
    group(length(WAs)+1:length(AAs)+length(WAs),1) = 2;
   
    % assign weihgts to the two groups to account for the
    % unbalanced sample sizes between them
    weights = zeros(size(group));
    weights(group == 1) = 1 / length(WAs);  
    weights(group == 2) = 1 / length(AAs); 
    weights = weights / sum(weights);

    %% Compute regional differences in gradient
    for region=1:nregs

      tbl = table(age, sex, group, RMS, ICV, education, income, sleep, substance, activity, social, gradient_value(:, region), ...
           'VariableNames', {'age', 'sex', 'group', 'RMS', 'ICV', 'education', 'income', 'sleep','substance','activity','social','response'});

      tbl.sex = categorical(tbl.sex);
      tbl.group = categorical(tbl.group);

      % Weighted linear model
     lm_weighted = fitlm(tbl, ['response ~ age*sex + RMS+ICV + group:education'], 'Weights', weights); 
     mytstat_confounds_weighted(region,1) = lm_weighted.Coefficients{7,3};
     mypval_confounds_weighted(region,1) = lm_weighted.Coefficients{7,4};
     
    end

      
      for i = 1:size(mypval_confounds_weighted,2)
        mypval_confounds_weighted_fdr(:,i) =mafdr(mypval_confounds_weighted(:,i), 'BHFDR', true); 
      end

    % Save the results
    savemat = (['D:\OneDrive\GraduateStudent_Phd1\8_MSN_GLM_2groups_2nd\SEM\majorRe\Intersection_effects_modelling\results_v70_edu_majorRe_signed_sp10' name '.mat']);
    save(savemat,'mypval_confounds_weighted_fdr','mytstat_confounds_weighted');