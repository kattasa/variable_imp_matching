setwd('/usr/project/xtmp/sk787/variable_imp_matching/')
library(dbarts)
library(grf)

slurm_id <- as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID'))
if(is.na(slurm_id))
    slurm_id = 0
set.seed(slurm_id)

# --- --- -------------------------------------------------------------- #
source('./Experiments/variance/M-ML.R', local=TRUE)

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
        '/df_train.csv'
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
        '/est.csv'
    ))

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
        CATE_true = ts$CATE_true,
        CATE_error_bound = MML_hi - MML_cate,
        CATE_lb = MML_lo,
        CATE_ub = MML_hi,
        contains_true_cate = (MML_lo <= ts$CATE_true) * (MML_hi >= ts$CATE_true),
        se = (MML_cate - ts$CATE_true)^2,
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
        CATE_true = ts$CATE_true,
        CATE_error_bound = cf_hi - cf_cate$predictions,
        CATE_lb = cf_lo,
        CATE_ub = cf_hi,
        contains_true_cate = (cf_lo <= ts$CATE_true) * (cf_hi >= ts$CATE_true),
        se = (cf_cate$predictions - ts$CATE_true)^2,
        fit = 'causal_forest_r',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )

    mml_return_df = data.frame(
        Y0_mean = Y0_MML$est,
        Y1_mean = Y1_MML$est,
        dist_0 = NA,
        dist_1 = NA,
        CATE_mean = MML_cate,
        CATE_true = ts$CATE_true,
        CATE_error_bound = MML_hi - MML_cate,
        CATE_lb = MML_lo,
        CATE_ub = MML_hi,
        contains_true_cate = (MML_lo <= ts$CATE_true) * (MML_hi >= ts$CATE_true),
        se = (MML_cate - ts$CATE_true)^2,
        fit = 'mml_bart',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )

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
        CATE_true = ts$CATE_true,
        CATE_error_bound = MML_hi - MML_cate,
        CATE_lb = MML_lo,
        CATE_ub = MML_hi,
        contains_true_cate = (MML_lo <= ts$CATE_true) * (MML_hi >= ts$CATE_true),
        se = (MML_cate - ts$CATE_true)^2,
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
        CATE_true = ts$CATE_true,
        CATE_error_bound = abs(conformal_cis$upper - conformal_cis$lower)/2,
        CATE_lb = conformal_cis$lower,
        CATE_ub = conformal_cis$upper,
        contains_true_cate = (conformal_cis$lower <= ts$CATE_true) * (conformal_cis$upper >= ts$CATE_true),
        se = NA,
        fit = 'conformal',
        seed = args$seed,
        dgp = args$dgp,
        n_train = args$n_train,
        n_est = args$n_est
    )

    # # Bootstrapping parameters
    # n_bootstrap = 500  # Number of bootstrap samples

    # bootstrap_cate <- function(learner_func, tr, ts, xcols, Tcol = 'T', Ycol = 'Y') {
    #     cate_estimates = matrix(NA, nrow = n_bootstrap, ncol = nrow(ts))
        
    #     for (i in 1:n_bootstrap) {
    #         # Bootstrap resampling
    #         boot_indices = sample(1:nrow(tr), replace = TRUE)
    #         boot_sample = tr[boot_indices, ]
            
    #         # Run the learner and store CATE estimates
    #         cate_estimates[i, ] = learner_func(boot_sample, ts, xcols, Tcol, Ycol)
    #     }
        
    #     # Calculate mean, lower (2.5th percentile), and upper (97.5th percentile) confidence intervals
    #     cate_mean = colMeans(cate_estimates)
    #     cate_lb = apply(cate_estimates, 2, quantile, probs = 0.025)
    #     cate_ub = apply(cate_estimates, 2, quantile, probs = 0.975)
        
    #     return(list(cate_mean = cate_mean, cate_lb = cate_lb, cate_ub = cate_ub))
    # }

    # # Helper functions for each learner
    # slearner_bart <- function(tr, ts, xcols, Tcol, Ycol) {
    #     # Fit S-learner
    #     slearner_fit = bart(tr[, c(xcols, Tcol)], tr[, Ycol], keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
        
    #     # Predict treatment (T=1) and control (T=0) outcomes
    #     Y1_slearner_ts = colMeans(predict(slearner_fit, data.frame(ts[, xcols], T = 1)))
    #     Y0_slearner_ts = colMeans(predict(slearner_fit, data.frame(ts[, xcols], T = 0)))
        
    #     # CATE estimate as the difference
    #     return(Y1_slearner_ts - Y0_slearner_ts)
    # }

    # tlearner_bart <- function(tr, ts, xcols, Tcol, Ycol) {
    #     # Fit separate BART models for treated and control groups
    #     tlearner_treated_fit = bart(tr[tr[, Tcol] == 1, xcols], tr[tr[, Tcol] == 1, Ycol], keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
    #     tlearner_control_fit = bart(tr[tr[, Tcol] == 0, xcols], tr[tr[, Tcol] == 0, Ycol], keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
        
    #     # Predict outcomes for test set
    #     Y1_tlearner_ts = colMeans(predict(tlearner_treated_fit, data.frame(ts[, xcols])))
    #     Y0_tlearner_ts = colMeans(predict(tlearner_control_fit, data.frame(ts[, xcols])))
        
    #     # CATE estimate as the difference
    #     return(Y1_tlearner_ts - Y0_tlearner_ts)
    # }

    # xlearner_bart <- function(tr, ts, xcols, Tcol, Ycol) {
    #     # Step 1: T-learner to get initial estimates
    #     tlearner_treated_fit = bart(tr[tr[, Tcol] == 1, xcols], tr[tr[, Tcol] == 1, Ycol], keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
    #     tlearner_control_fit = bart(tr[tr[, Tcol] == 0, xcols], tr[tr[, Tcol] == 0, Ycol], keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
        
    #     Y1_xlearner_ms = colMeans(predict(tlearner_treated_fit, data.frame(tr[, xcols])))
    #     Y0_xlearner_ms = colMeans(predict(tlearner_control_fit, data.frame(tr[, xcols])))

    #     # Step 2: Impute counterfactuals
    #     tau_control = tr[tr[, Tcol] == 0, Ycol] + (Y1_xlearner_ms[tr[, Tcol] == 0] - tr[tr[, Tcol] == 0, Ycol])
    #     tau_treated = tr[tr[, Tcol] == 1, Ycol] - (tr[tr[, Tcol] == 1, Ycol] - Y0_xlearner_ms[tr[, Tcol] == 1])

    #     # Step 3: Train models on imputed treatment effects
    #     xlearner_control_fit = bart(tr[tr[, Tcol] == 0, xcols], tau_control, keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)
    #     xlearner_treated_fit = bart(tr[tr[, Tcol] == 1, xcols], tau_treated, keeptrees = TRUE, ntree = 50, nskip = 100, verbose = FALSE, nthread = 1)

    #     # Step 4: Predict treatment effects
    #     xlearner_control_effects = colMeans(predict(xlearner_control_fit, data.frame(ts[, xcols])))
    #     xlearner_treated_effects = colMeans(predict(xlearner_treated_fit, data.frame(ts[, xcols])))

    #     # Step 5: Average treatment effects for treated and control
    #     return((xlearner_control_effects + xlearner_treated_effects) / 2)
    # }

    # ### Bootstrapping for S-Learner ###
    # message("Running Bootstrapped S-Learner...")
    # slearner_result = bootstrap_cate(slearner_bart, tr, ts, xcols)

    # slearner_return_df = data.frame(
    #     CATE_mean = slearner_result$cate_mean,
    #     CATE_true = ts$CATE_true,
    #     CATE_lb = slearner_result$cate_lb,
    #     CATE_ub = slearner_result$cate_ub,
    #     contains_true_cate = (slearner_result$cate_lb <= ts$CATE_true) & (slearner_result$cate_ub >= ts$CATE_true),
    #     se = (slearner_result$cate_mean - ts$CATE_true)^2,
    #     fit = 'slearner_bart_bootstrap'
    # )

    # ### Bootstrapping for T-Learner ###
    # message("Running Bootstrapped T-Learner...")
    # tlearner_result = bootstrap_cate(tlearner_bart, tr, ts, xcols)

    # tlearner_return_df = data.frame(
    #     CATE_mean = tlearner_result$cate_mean,
    #     CATE_true = ts$CATE_true,
    #     CATE_lb = tlearner_result$cate_lb,
    #     CATE_ub = tlearner_result$cate_ub,
    #     contains_true_cate = (tlearner_result$cate_lb <= ts$CATE_true) & (tlearner_result$cate_ub >= ts$CATE_true),
    #     se = (tlearner_result$cate_mean - ts$CATE_true)^2,
    #     fit = 'tlearner_bart_bootstrap'
    # )

    # ### Bootstrapping for X-Learner ###
    # message("Running Bootstrapped X-Learner...")
    # xlearner_result = bootstrap_cate(xlearner_bart, tr, ts, xcols)

    # xlearner_return_df = data.frame(
    #     CATE_mean = xlearner_result$cate_mean,
    #     CATE_true = ts$CATE_true,
    #     CATE_lb = xlearner_result$cate_lb,
    #     CATE_ub = xlearner_result$cate_ub,
    #     contains_true_cate = (xlearner_result$cate_lb <= ts$CATE_true) & (xlearner_result$cate_ub >= ts$CATE_true),
    #     se = (xlearner_result$cate_mean - ts$CATE_true)^2,
    #     fit = 'xlearner_bart_bootstrap'
    # )


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

