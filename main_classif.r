rm(list = ls())

args = commandArgs(trailingOnly=TRUE)
pathx = args[1]
pathy = args[2]
pathx_test = args[3]

X <- read.csv(file=pathx, header=TRUE, sep=",")
y <- read.csv(file=pathy, header=FALSE, sep=",")
y <- y$V1
X_test <- read.csv(file=pathx_test, header=TRUE, sep=",")

data_train = data.frame(X=X, Y=Y)

library(RWeka)
library(sirus)
##############################
##          RIPPER          ##
##############################
ripper <- JRip(Y ~ ., data = data_train) # Fit
# ripper.m <- evaluate_Weka_classifier(ripper) # Calculation of error
ripper_nbrules = ripper$classifier$measureNumRules()
temp = ripper$classifier$toString()
int_ripper = length(str_split(temp,'X.')) - 1
ripper_rules = str_split(temp,'\n')[[1]]
ripper_rules = ripper_rules[ripper_rules != '']
ripper_pred <- predict(ripper, newdata=data.frame(X=X_test)) # Predict

##############################
##          OneR            ##
##############################
# Not relevant in the study
# oner <- OneR(Y ~ ., data = data_train) # Fit
# oner.m <- evaluate_Weka_classifier(oner) # Calculation of error
# oner_pred <- predict(oner, newdata=data.frame(X=X_test)) # Predict

##############################
##          PART            ##
##############################
part <- PART(Y ~ ., data = data_train) # Fit
# part.m <- evaluate_Weka_classifier(part) # Calculation of error
part_nbrules = part$classifier$measureNumRules()
temp = part$classifier$toString()
int_part = length(str_split(temp,'X.')) - 1
part_rules = str_split(temp,'\n')[[1]]

part_pred <- predict(part, newdata=data.frame(X=X_test)) # Predict

##############################
##          SIRUS           ##
##############################
sirus.m <- sirus.fit(X, y, verbose=FALSE)
sirus_nbrules = length(sirus.m$rules)
int_sirus <- 0
for(i in 1:length(sirus.m$rules)){int_sirus = int_sirus + length(sirus.m$rules[[i]])}

sirus_pred <- sirus.predict(sirus.m, X_test)


##############################
##       Save Results       ##
##############################
path_ripper_rules <- gsub('X.csv', 'ripper_rules.csv', pathx)
path_part_rules <- gsub('X.csv', 'par_rules.csv', pathx)
write.csv(ripper_rules, path_ripper_rules,  row.names=FALSE)
write.csv(part_rules, path_part_rules,  row.names=FALSE)


path_sirus <- gsub('X.csv', 'sirus_pred.csv', pathx)
path_ripper <- gsub('X.csv', 'ripper_pred.csv', pathx)
path_part <- gsub('X.csv', 'part_pred.csv', pathx)

write.csv(sirus_pred, path_sirus,  row.names=FALSE)
write.csv(ripper_pred, path_ripper,  row.names=FALSE)
write.csv(part_pred, path_part,  row.names=FALSE)

path_int <- gsub('X.csv', 'int.csv', pathx)
t <- data.frame('Sirus' = c(sirus_nbrules, int_sirus),
                'RIPPER' = c(ripper_nbrules, int_ripper),
                'PART' = c(part_nbrules, int_part),
                row.names=c('nb_rules', 'int'))
write.csv(t, path_int, row.names=FALSE)
