library(stringr)
rm(list = ls())

args = commandArgs(trailingOnly=TRUE)
pathx = args[1]
pathy = args[2]
pathx_test = args[3]
do_pred = args[4]

X <- read.csv(file=pathx, header=TRUE, sep=",")
y <- read.csv(file=pathy, header=FALSE, sep=",")
y <- y$V1

if(do_pred){X_test <- read.csv(file=pathx_test, header=TRUE, sep=",")}

data_train = data.frame(X=X, Y=y)

library(RWeka)
##############################
##          RIPPER          ##
##############################
ripper <- JRip(Y ~ ., data = data_train) # Fit
# ripper.m <- evaluate_Weka_classifier(ripper) # Calculation of error

temp = ripper$classifier$toString()
ripper_rules = str_split(temp,'\n')[[1]]
ripper_rules = ripper_rules[ripper_rules != '']
ripper_rules = ripper_rules[4:length(ripper_rules)-1]
ripper_rules = gsub("and", "AND", ripper_rules)
ripper_rules = gsub("X.", "", ripper_rules)
ripper_rules = gsub("\\(", "", ripper_rules)
ripper_rules = gsub("\\)", "", ripper_rules)
ripper_rules = gsub(" =>.*", "", ripper_rules)
ripper_rules = gsub(" <= ", " in -Inf;", ripper_rules)
ripper_rules = gsub(" < ", " in -Inf;", ripper_rules)

for(i in 1:length(ripper_rules))
{
    r = ripper_rules[i]
    temp = strsplit(r," AND ")[[1]]
    if(length(temp) > 0)
    {
        for(j in 1:length(temp))
        {
            if(grepl('>=', temp[j], fixed=TRUE))
            {
                feat = strsplit(temp[j],' >= ')[[1]][1]
                val = strsplit(temp[j],' >= ')[[1]][2]
                ripper_rules[i] = gsub(paste(feat, " >= ", val,  sep=''), paste(feat, " in ", val, ';Inf ', sep=''), ripper_rules[i])
            }
            if(grepl('>', temp[j], fixed=TRUE))
            {
                feat = strsplit(temp[j],' > ')[[1]][1]
                val = strsplit(temp[j],' > ')[[1]][2]
                ripper_rules[i] = gsub(paste(feat, " > ", val,  sep=''), paste(feat, " in ", val, ';Inf ', sep=''), ripper_rules[i])
            }
        }
    }
}
ripper_rules <- data.frame('Rules'=matrix(unlist(ripper_rules), nrow=length(ripper_rules), byrow=T))

##############################
##          PART            ##
##############################
part <- PART(Y ~ ., data = data_train) # Fit
# part.m <- evaluate_Weka_classifier(part) # Calculation of error
temp = part$classifier$toString()
int_part = length(str_split(temp,'X.')) - 1
part_rules = str_split(temp,'\n\n')[[1]]
part_rules = part_rules[3:length(part_rules)-1]
part_rules = gsub("\n", " ", part_rules)
part_rules = gsub("X.", "", part_rules)
part_rules = gsub(":.*", " ", part_rules)
part_rules = gsub(" < ", " in -Inf;", part_rules)
part_rules = gsub(" <= ", " in -Inf;", part_rules)

for(i in 1:length(part_rules))
{
    r = part_rules[i]
    temp = strsplit(r," AND ")[[1]]
    if(length(temp) > 0)
    {
        for(j in 1:length(temp))
        {
            if(grepl('>=', temp[j], fixed=TRUE))
            {
                feat = strsplit(temp[j],' >= ')[[1]][1]
                val = strsplit(temp[j],' >= ')[[1]][2]
                part_rules[i] = gsub(paste(feat, " >= ", val,  sep=''), paste(feat, " in ", val, ';Inf ', sep=''), part_rules[i])
            }
            if(grepl('>', temp[j], fixed=TRUE))
            {
                feat = strsplit(temp[j],' > ')[[1]][1]
                val = strsplit(temp[j],' > ')[[1]][2]
                part_rules[i] = gsub(paste(feat, " > ", val,  sep=''), paste(feat, " in ", val, ';Inf ', sep=''), part_rules[i])
            }
        }
    }
}
part_rules <- data.frame('Rules'=matrix(unlist(part_rules), nrow=length(part_rules), byrow=T))


##############################
##       Save Results       ##
##############################
if(do_pred){
ripper_pred <- predict(ripper, newdata=data.frame(X=X_test)) # Predict
part_pred <- predict(part, newdata=data.frame(X=X_test))

path_ripper <- gsub('X.csv', 'ripper_pred.csv', pathx)
path_part <- gsub('X.csv', 'part_pred.csv', pathx)
write.csv(ripper_pred, path_ripper,  row.names=FALSE)
write.csv(part_pred, path_part,  row.names=FALSE)
}
path_ripper_rules <- gsub('X.csv', 'ripper_rules.csv', pathx)
path_part_rules <- gsub('X.csv', 'part_rules.csv', pathx)
write.csv(ripper_rules, path_ripper_rules,  row.names=FALSE)
write.csv(part_rules, path_part_rules,  row.names=FALSE)


# y[y=='+'] = 1
# y[y=='-'] = 0
# library(sirus)
# ##############################
# ##          SIRUS           ##
# ##############################
# sirus.m <- sirus.fit(X, y, verbose=FALSE)
# sirus_nbrules = length(sirus.m$rules)
# int_sirus <- 0
# for(i in 1:length(sirus.m$rules)){int_sirus = int_sirus + length(sirus.m$rules[[i]])}
#
# sirus_rules = list()
# for(i in 1:length(sirus.m$rules)){
# rl = ''
# dep = length(sirus.m$rules[[i]])
# for(j in 1:dep){
# rl = paste(rl, sirus.m$rules[[i]][[j]][1], sep='')
# if(sirus.m$rules[[i]][[j]][2] == '<'){rl = paste(rl, ' in ', '-Inf;' ,sirus.m$rules[[i]][[j]][3], sep='')}
# else{rl = paste(rl, ' in ', sirus.m$rules[[i]][[j]][3], ';Inf' , sep='')}
# if(j < dep){rl = paste(rl, ' AND ')}
# }
# sirus_rules[i] = rl
# }
#
# sirus_rules <- data.frame('Rules'=matrix(unlist(sirus_rules), nrow=length(sirus_rules), byrow=T))
#
# library(nodeHarvest)
# ##############################
# ##       nodeharvest        ##
# ##############################
# NH <- nodeHarvest(X, y, silent=TRUE)
# nh_nbrules = length(NH$nodes)
# int_nh <- 0
# for(i in 1:length(NH$nodes)){int_nh = int_nh + attr(NH$nodes[[i]], 'depth')}
#
# nh_rules = list()
# for(i in 1:length(NH$nodes)){
# rl = ''
# dep = max(1, attr(NH$nodes[[i]], 'depth'))
# for(j in 1:dep){
# rl = paste(rl, NH$varnames[NH$nodes[[i]][j]], ' in ', NH$nodes[[i]][j + dep], ';', NH$nodes[[i]][j + 2*dep], sep='')
# if(j < dep){rl = paste(rl, 'AND ')}
# }
# nh_rules[i] = rl
# }
# nh_rules <- data.frame('Rules'=matrix(unlist(nh_rules), nrow=length(nh_rules), byrow=T))
#
#
# ##############################
# ##       Save Results       ##
# ##############################
# if(do_pred){
# sirus_pred <- sirus.predict(sirus.m, X_test)
# nh_pred <- predict(NH, X_test)
#
# path_sirus <- gsub('X.csv', 'sirus_pred.csv', pathx)
# nh_part <- gsub('X.csv', 'nh_pred.csv', pathx)
# write.csv(sirus_pred, path_sirus,  row.names=FALSE)
# write.csv(nh_pred, nh_sirus,  row.names=FALSE)
# }
#
# path_sirus_rules <- gsub('X.csv', 'sirus_rules.csv', pathx)
# path_nh_rules <- gsub('X.csv', 'nh_rules.csv', pathx)
# write.csv(sirus_rules, path_sirus_rules,  row.names=FALSE)
# write.csv(nh_rules, path_nh_rules,  row.names=FALSE)
