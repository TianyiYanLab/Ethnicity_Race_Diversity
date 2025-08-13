# read data 
setwd("D:/OneDrive/8_MSN_GLM_2groups_2nd/SEM/only_func_v2/finalexps")
hcp_data <- read.csv("HCPYA_data_v3_ICV_RMS_z.csv") 
hcp_rsfc_data <- read.csv("HCPYA_SU_top100_v1.csv")

#combine data
data_combined <- cbind(hcp_data,hcp_rsfc_data)

#categorical sex and group

data_combined$sex <- as.factor(data_combined$sex)
data_combined$group <- as.factor(data_combined$group)

#construct SEM
library(lavaan)

sem_model <- '
   #  measurement model
      RSFC=~ RSFCRegion1+RSFCRegion2+RSFCRegion3+RSFCRegion4+RSFCRegion5+RSFCRegion6+RSFCRegion7+RSFCRegion8+RSFCRegion9+RSFCRegion10+RSFCRegion11+RSFCRegion12+RSFCRegion13+RSFCRegion14+RSFCRegion15
      SES =~ education + income

   #  regressions
      RSFC ~ age + sex + c*group01 + SES + b*substance + RMS + ICV
      substance ~ a*group01 + age +sex + SES
      
         ab := a*b                                       # mediation effect (a*b path)
         direct_effect := c
         total_effect := c + (ab) 
         PercentageMediated := ((ab) / (c + ab)) * 100


'

fit <- sem(sem_model, data = data_combined, meanstructure = TRUE, se = "robust")
summary(fit, fit.measures=TRUE, standard=TRUE)

