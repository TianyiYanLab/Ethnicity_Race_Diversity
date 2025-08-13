function model1(AAls,WAls,HCPYA_data,nregs)
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
    gradient_value = table2array(data_src(:,12:411));
    % assign weihgts to the two groups to account for the
    % unbalanced sample sizes between them
    group(1:length(WAs),1) = 1;
    group(length(WAs)+1:length(AAs)+length(WAs),1) = 2;
    % 
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

      lm_weighted = fitlm(tbl, ['response ~ age*sex + RMS+ICV + group:substance'], 'Weights', weights); 
      mytstat_confounds_weighted(region,1) = lm_weighted.Coefficients{3,3};
      mypval_confounds_weighted(region,1) = lm_weighted.Coefficients{3,4};
                
    end
      
      for i = 1:size(mypval_confounds_weighted,2)
        mypval_confounds_weighted_fdr(:,i) =mafdr(mypval_confounds_weighted(:,i), 'BHFDR', true); 
      end

    % save 
    savemat = (['D:\OneDrive\finalexps\Intersection_effects_modelling\results_v33.mat']);
    save(savemat,'mypval_confounds_weighted_fdr','mytstat_confounds_weighted');