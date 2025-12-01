# read data 
setwd("D:/OneDrive/GraduateStudent_Phd1/4_Population_differences_BrainParcellation/Population_differences_BrainParcellation/8_MSN_GLM_2groups_2nd/SEM/majorRe")
hcp_data <- read.csv("HCPYA_data_v3_ICV_RMS_z_majorRe.csv")
#hcp_data <- read.csv("Validation_KRRoutput.csv")


#hcp_rsfc_data <- read.csv("D:/OneDrive/GraduateStudent_Phd1/4_Population_differences_BrainParcellation/Population_differences_BrainParcellation/8_MSN_GLM_2groups_2nd/SEM/majorRe/pc1_signed/HCPYA_SR_z_top100_v1.csv") # OK
hcp_rsfc_data <- read.csv("D:/OneDrive/GraduateStudent_Phd1/4_Population_differences_BrainParcellation/Population_differences_BrainParcellation/8_MSN_GLM_2groups_2nd/SEM/majorRe/pca1_sp10_signed/HCPYA_SR_z_top100_v1.csv")# OK



#combine data
data_combined <- cbind(hcp_data,hcp_rsfc_data)

#categorical sex and group
data_combined$sex <- as.factor(data_combined$sex)
data_combined$group01 <- as.factor(data_combined$group01)

rsfc_cols <- c(
  "RSFCRegion1", "RSFCRegion2", "RSFCRegion3", "RSFCRegion4", "RSFCRegion5",
  "RSFCRegion6", "RSFCRegion7", "RSFCRegion8", "RSFCRegion9", "RSFCRegion10",
  "RSFCRegion11", "RSFCRegion12", "RSFCRegion13", "RSFCRegion14", "RSFCRegion15"
)
data_combined$RSFC_mean <- rowMeans(data_combined[, rsfc_cols])

data_combined$SES    <- rowMeans(data_combined[, c("education", "income")])

#construct SEM
library(lavaan)

sem_model <- '
    
  # regressions
    RSFC_mean ~ age + sex + c*group01 + SES + b*social + RMS + ICV
    social ~ a*group01 + age +sex + SES + RMS + ICV
    
    ab := a*b                          # mediation effect (a*b path)

'

fit <- sem(sem_model, data = data_combined, meanstructure = TRUE, se = "bootstrap",bootstrap = 1000)
parameterEstimates(fit, boot.ci.type = "bca.simple", standardized = TRUE)
summary(fit, fit.measures=TRUE, standard=TRUE)

#fit <- sem(sem_model, data = data_combined, meanstructure = TRUE, se = "robust")


