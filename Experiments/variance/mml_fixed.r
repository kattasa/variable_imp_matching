setwd('/usr/project/xtmp/sk787/variable_imp_matching/')
library(dbarts)
library(grf)

slurm_id <- as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID'))
if(is.na(slurm_id))
    slurm_id = 0
set.seed(slurm_id)

# --- --- -------------------------------------------------------------- #
source('./Experiments/variance/M-ML.R', local=TRUE)


## Get counterfactual intervals by X-learner
xlearner_Cf_CI <- function(X, Y, T, Xtest,
                           B = 50){
    if (B == 0){
        df_tau <- df_Y <- list(cr = NA, len = NA)
        return(list(tau = df_tau, Y1 = df_Y))
    }

    xl_rf <- causalToolbox::X_RF(feat = X, tr = T, yobs = Y, nthread = 0)
    cate_esti_rf <- causalToolbox::EstimateCate(xl_rf, Xtest)
    CI <- causalToolbox::CateCI(xl_rf, Xtest, B = B,
                                verbose = FALSE, nthread = 1)[, 2:3]
    return(list(tau = CI, Y = CI))
}

## Get counterfactual intervals by BART
bart_Cf_CI <- function(X, Y, Xtest){
    ids <- !is.na(Y)
    X <- as.data.frame(X)[ids, ]
    y <- Y[ids]
    Xtest <- as.data.frame(Xtest)
    fit <- bartMachine::bartMachine(X, y, verbose = FALSE)
    CI_tau <- bartMachine::calc_credible_intervals(fit, new_data = Xtest, ci_conf = 0.95)
    CI_Y <- bartMachine::calc_prediction_intervals(fit, new_data = Xtest, pi_conf = 0.95)$interval
    return(list(tau = CI_tau, Y = CI_Y))
}

run_iter = function(args){
    message("Reading Data...")
        
    tr_sub = read.csv(paste0(
        './Experiments/variance/output_files/dgp_', args$dgp,
        '/n_train_', args$n_train,
        '/n_est_', args$n_est,
        '/n_imp_', args$n_imp,
        '/n_unimp_', args$n_unimp,
        '/k_', args$k,
        '/seed_', args$seed,
        '/df_train_sub.csv'
    ))
    calib = read.csv(paste0(
        './Experiments/variance/output_files/dgp_', args$dgp,
        '/n_train_', args$n_train,
        '/n_est_', args$n_est,
        '/n_imp_', args$n_imp,
        '/n_unimp_', args$n_unimp,
        '/k_', args$k,
        '/seed_', args$seed,
        '/df_calib.csv'
    ))
    tr = rbind(tr_sub, calib)
    ms = read.csv(paste0(
        './Experiments/variance/output_files/dgp_', args$dgp,
        '/n_train_', args$n_train,
        '/n_est_', args$n_est,
        '/n_imp_', args$n_imp,
        '/n_unimp_', args$n_unimp,
        '/k_', args$k,
        '/seed_', args$seed,
        '/df_est.csv'
    ))

    ts = read.csv(paste0(
        './Experiments/variance/output_files/dgp_', args$dgp,
        '/n_imp_', args$n_imp, 
        '/n_unimp_', args$n_unimp,
        '/query_x.csv'
    ))

    true_cate = read.csv(paste0(
        './Experiments/variance/output_files/dgp_', args$dgp,
        '/n_imp_', args$n_imp, 
        '/n_unimp_', args$n_unimp,
        '/cate_true.csv'
    ))$TE
    xcols = colnames(tr)[grepl(colnames(tr), pattern = 'X')]

    ### Estimation ###

    

    message("Running MML BART...")
    bart_fit = bart(tr[, c(xcols, 'T')], tr$Y, keeptrees = TRUE, 
                    ntree = 50, nskip=100, verbose = FALSE, nthread=1)
    Y1_bart_ms = colMeans(predict(bart_fit, data.frame(ms[, xcols], T=1)))
    Y0_bart_ms = colMeans(predict(bart_fit, data.frame(ms[, xcols], T=0)))
    Y1_bart_ts = colMeans(predict(bart_fit, data.frame(ts[, xcols], T=1)))
    Y0_bart_ts = colMeans(predict(bart_fit, data.frame(ts[, xcols], T=0)))
    k1 = floor(sum(ms$T)^(1/4))
    k0 = floor(sum(1- ms$T)^(1/4))
    Y1_MML = data.frame(MML_knn_crfs(Y1_bart_ts, Y1_bart_ms, ms$Y, ms$T, 1, k1, include_variance=TRUE))
    Y0_MML = data.frame(MML_knn_crfs(Y0_bart_ts, Y0_bart_ms, ms$Y, ms$T, 0, k0, include_variance=TRUE))
    MML_cate = Y1_MML$est - Y0_MML$est
    #### need to make k^(1/4) to account for BART convergence rate
    MML_lo = MML_cate - qnorm(0.975) * sqrt((Y1_MML$var/sqrt(k1) + Y0_MML$var/sqrt(k0)))
    MML_hi = MML_cate + qnorm(0.975) * sqrt((Y1_MML$var/sqrt(k1) + Y0_MML$var/sqrt(k0)))

    mml_return_df = data.frame(
        Y0_mean = Y0_MML$est,
        Y1_mean = Y1_MML$est,
        dist_0 = NA,
        dist_1 = NA,
        CATE_mean = MML_cate,
        CATE_true = true_cate,
        CATE_error_bound = MML_hi - MML_cate,
        CATE_lb = MML_lo,
        CATE_ub = MML_hi,
        contains_true_cate = (MML_lo <= true_cate) * (MML_hi >= true_cate),
        se = (MML_cate - true_cate)^2,
        fit = 'mml_bart',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )
    # ### Causal Forest
    message("Running CF...")
    cf_fit = causal_forest(X = tr[, xcols], Y = tr$Y, W = tr$T, num.threads=1)
    cf_cate = predict(cf_fit, ts[, xcols], estimate.variance=TRUE)

    cf_lo = cf_cate$predictions - qnorm(0.975) * sqrt(cf_cate$variance.estimates)
    cf_hi = cf_cate$predictions + qnorm(0.975) * sqrt(cf_cate$variance.estimates)

    cf_return_df = data.frame(
        Y0_mean = NA,
        Y1_mean = NA,
        dist_0 = NA,
        dist_1 = NA,
        CATE_mean = cf_cate$predictions,
        CATE_true = true_cate,
        CATE_error_bound = cf_hi - cf_cate$predictions,
        CATE_lb = cf_lo,
        CATE_ub = cf_hi,
        contains_true_cate = (cf_lo <= true_cate) * (cf_hi >= true_cate),
        se = (cf_cate$predictions - true_cate)^2,
        fit = 'causal_forest_r',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )

    # message("Running X Learner...")
    # xl_rf <- causalToolbox::X_RF(feat = tr[, xcols], tr = tr$T, yobs = tr$Y, nthread = 1)
    # CI <- causalToolbox::CATE_CI(xl_rf, ts[, xcols], B = 50)
    # xlo = CI[, 2]
    # xhi = CI[, 3]

    # xlearner_return_df = data.frame(
    #     Y0_mean = NA,
    #     Y1_mean = NA,
    #     dist_0 = NA,
    #     dist_1 = NA,
    #     CATE_mean = cf_cate$predictions,
    #     CATE_true = true_cate,
    #     CATE_error_bound = cf_hi - cf_cate$predictions,
    #     CATE_lb = cf_lo,
    #     CATE_ub = cf_hi,
    #     contains_true_cate = (cf_lo <= true_cate) * (cf_hi >= true_cate),
    #     se = (cf_cate$predictions - true_cate)^2,
    #     fit = 'causal_forest_r',
    #     seed = args$seed,
    #     dgp = args$dgp,
    #     n_train = args$n_train,
    #     n_est = args$n_est
    # )

    message("Running MML CF...")
    cf_ms = predict(cf_fit, ms[, xcols], estimate.variance=FALSE)$predictions
    cf_ts = predict(cf_fit, ts[, xcols], estimate.variance=FALSE)$predictions
    k1 = floor(sum(ms$T)^(1/4))
    k0 = floor(sum(1- ms$T)^(1/4))
    Y1_MML = data.frame(MML_knn_crfs(cf_ts, cf_ms[which(ms$T == 1)], ms$Y, ms$T, 1, k1, include_variance=TRUE))
    Y0_MML = data.frame(MML_knn_crfs(cf_ts, cf_ms[which(ms$T == 0)], ms$Y, ms$T, 0, k0, include_variance=TRUE))
    MML_cate = Y1_MML$est - Y0_MML$est
    #### need to make k^(1/4) to account for BART convergence rate
    MML_lo = MML_cate - qnorm(0.975) * sqrt((Y1_MML$var/sqrt(k1) + Y0_MML$var/sqrt(k0)))
    MML_hi = MML_cate + qnorm(0.975) * sqrt((Y1_MML$var/sqrt(k1) + Y0_MML$var/sqrt(k0)))

    mml_cf_return_df = data.frame(
        Y0_mean = Y0_MML$est,
        Y1_mean = Y1_MML$est,
        dist_0 = NA,
        dist_1 = NA,
        CATE_mean = MML_cate,
        CATE_true = true_cate,
        CATE_error_bound = MML_hi - MML_cate,
        CATE_lb = MML_lo,
        CATE_ub = MML_hi,
        contains_true_cate = (MML_lo <= true_cate) * (MML_hi >= true_cate),
        se = (MML_cate - true_cate)^2,
        fit = 'mml_cf',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )


    message("Running CQR...")
    library('cfcausal')
    conformal_tr = rbind(tr, ms)
    CIfun = conformalIte(conformal_tr[, xcols], conformal_tr$Y, conformal_tr$T, alpha = 0.05,
                 algo = 'nest', exact = TRUE, type = 'CQR', quantiles = c(0.05, 0.95), outfun = 'quantRF', useCV = FALSE
    )
    conformal_cis = CIfun(ts[, xcols])
    conformal_return_df = data.frame(
        Y0_mean = NA,
        Y1_mean = NA,
        dist_0 = NA,
        dist_1 = NA,
        CATE_mean = NA,
        CATE_true = true_cate,
        CATE_error_bound = abs(conformal_cis$upper - conformal_cis$lower)/2,
        CATE_lb = conformal_cis$lower,
        CATE_ub = conformal_cis$upper,
        contains_true_cate = (conformal_cis$lower <= true_cate) * (conformal_cis$upper >= true_cate),
        se = NA,
        fit = 'conformal',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )
    return(list(
        mml = mml_return_df, 
        # slearner = slearner_return_df,
        # tlearner = tlearner_return_df,
        # xlearner = xlearner_return_df,
        mml_cf = mml_cf_return_df,
        cf = cf_return_df,
        conformal = conformal_return_df
        ))
}

### read in command line arguments
library(optparse)

option_list <- list(make_option(c("--task_id"), type="integer", default = NULL, help="Task ID"))

parser <- OptionParser(option_list = option_list)
opts <- parse_args(parser)
print(paste('R task id', opts$task_id))
args_df = read.csv('./Experiments/variance/args.csv')
if(opts$task_id > nrow(args_df)) {
    message('task id is out of bounds.')
    quit()
}
args = args_df[opts$task_id, ]

### run/save baselines

res_cp = run_iter(args)
write.csv(res_cp$mml, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/mml_bart.csv'), row.names=FALSE)
write.csv(res_cp$cf, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/causal_forest_r.csv'), row.names=FALSE)
write.csv(res_cp$mml_cf, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/mml_cf.csv'), row.names=FALSE)
write.csv(res_cp$conformal, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/conformal.csv'), row.names=FALSE)
# write.csv(res_cp$tlearner, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/tlearner_r.csv'), row.names=FALSE)
# write.csv(res_cp$xlearner, paste0('./Experiments/variance/output_files/dgp_',args$dgp, '/n_train_', args$n_train, '/n_est_', args$n_est, '/n_imp_', args$n_imp, '/n_unimp_', args$n_unimp, '/k_', args$k, '/seed_', args$seed, '/xlearner_r.csv'), row.names=FALSE)

