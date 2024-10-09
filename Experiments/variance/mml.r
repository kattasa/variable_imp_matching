setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

library(MatchIt)
library(dbarts)
library(ggplot2)
library(grf)


## get command line arguments
args_df = read.csv('Experiments/variance/args.csv')

## read in data
args = args_df[1, ]

tr_sub = read.csv(paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/df_train.csv'
  ))
calib = read.csv(paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/df_calib.csv'
  ))
tr = rbind(tr_sub, calib)
ts = read.csv(paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/df_est.csv'
  ))
p = args$n_imp + args$n_unimp

## Bart
bart_fit = bart(tr[, c(paste0('X', 0:(p-1)), 'T')], tr$Y, keeptrees = TRUE, ntree = 50, nskip=1000)
Y1_bart = colMeans(predict(bart_fit, data.frame(ts[, 1:p], T=1)))
Y0_bart = colMeans(predict(bart_fit, data.frame(ts[, 1:p], T=0)))
cate_bart = Y1_bart - Y0_bart
# print(mean(abs((Y1_bart - Y0_bart) - (ts$cate))))
#plot((Y1_bart - Y0_bart), (ts$cate))

## M-ML


## Normalizing constants for phi are estimated from the training data
Y1_bart_tr = colMeans(predict(bart_fit, data.frame(tr[, 1:p], T=1)))
Y0_bart_tr = colMeans(predict(bart_fit, data.frame(tr[, 1:p], T=0)))
Y1_dist_tr = abs(rowSums(expand.grid(Y1_bart_tr, -Y1_bart_tr)))
Y0_dist_tr = abs(rowSums(expand.grid(Y0_bart_tr, -Y0_bart_tr)))
fphix1 = approxfun(density(Y1_bart_tr))
fphix0 = approxfun(density(Y0_bart_tr))
ephix1 = glm(tr$T ~ Y1_bart_tr, family = binomial(link="logit"))
ephix0 = glm((1-tr$T) ~ Y0_bart_tr, family = binomial(link="logit"))
c1 = quantile(Y1_dist_tr, .001)
c0 = quantile(Y0_dist_tr, .001)

Gamman = (nrow(ts) * 0.25)
K1 = sum(ts$T) * 0.25
K0 = sum(1-ts$T) * 0.25

phi1 = Y1_bart
phi0 = Y0_bart

Y1_dist = abs(rowSums(expand.grid(phi1, -phi1)))
Y1_dist = matrix(Y1_dist, nrow(ts), nrow(ts),byrow=T)
Y0_dist = abs(rowSums(expand.grid(phi0, -phi0)))
Y0_dist = matrix(Y0_dist, nrow(ts), nrow(ts),byrow=T)

nuphi1 = fphix1(phi1) * predict(ephix1, newdata=data.frame(Y1_bart_tr=phi1), type="response") * c1
nuphi0 = fphix0(phi0) * predict(ephix0, newdata=data.frame(Y0_bart_tr=phi0), type="response") * c0

N_test = nrow(ts)
Y1_MML = rep(NA, N_test)
Y0_MML = rep(NA, N_test)
Y1_KNN = rep(NA, N_test)
Y0_KNN = rep(NA, N_test)

sigmasq1_KNN = rep(NA, N_test)
sigmasq0_KNN = rep(NA, N_test)
sigmasq1 = rep(NA, N_test)
sigmasq0 = rep(NA, N_test)
for (i in 1:N_test){
  MGi1 = ts[which(Y1_dist[i, -i] <= Gamman * c1), ]
  MGi0 = ts[which(Y0_dist[i, -i] <= Gamman * c0), ]
  Y1_MML[i] = mean(MGi1$Y[MGi1$T==1])
  Y0_MML[i] = mean(MGi0$Y[MGi0$T==0])
  sigmasq1[i] = var(MGi1$Y[MGi1$T==1])
  sigmasq0[i] = var(MGi0$Y[MGi0$T==1])
  
  MGi1 = ts[order(Y1_dist[i, -i])[1:K1], ]
  MGi0 = ts[order(Y0_dist[i, -i])[1:K0], ]
  Y1_KNN[i] = mean(MGi1$Y[MGi1$T==1])
  Y0_KNN[i] = mean(MGi0$Y[MGi0$T==0])
  sigmasq1_KNN[i] = var(MGi1$Y[MGi1$T==1])
  sigmasq0_KNN[i] = var(MGi0$Y[MGi0$T==1])
  
}
cate_MML = (Y1_MML - Y0_MML)
mean(abs((cate_MML - ts$cate)), na.rm=T)

cate_KNN = (Y1_KNN - Y0_KNN)
mean(abs((cate_KNN - ts$cate)), na.rm=T)

### Causal Forest
cf_fit = causal_forest(X = tr[, 1:p], Y = tr$Y, W = tr$T)
cf_cate = predict(cf_fit, ts[, 1:p], estimate.variance=TRUE)
cf_hi = cf_cate$predictions + qnorm(0.975) * sqrt(cf_cate$variance.estimates)
cf_lo = cf_cate$predictions - qnorm(0.975) * sqrt(cf_cate$variance.estimates)
print(mean((ts$cate <= cf_hi) & (ts$cate >= cf_lo)))

## X-Learner
# create the hte object using honest Random Forests (RF)
xl_rf <- X_RF(feat = tr[, 1:p], tr = as.numeric(tr$T), yobs = tr$Y)
cate_esti_rf <- EstimateCate(xl_rf, ts[,1:p])