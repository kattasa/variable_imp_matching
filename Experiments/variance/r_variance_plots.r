library(dplyr)
library(purrr)
library(readr)
library(ggplot2)
library(progress)

setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

# Function to read CSV with error handling
read_df_w_error <- function(args, counter, summ_df) {
  tryCatch(
    {
      file_path <- paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_est_', args$n_est,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/', args$fit, '.csv'
  )
      data.table::fread(file_path) %>%
      mutate(Y0_mean = as.numeric(Y0_mean),
             Y1_mean = as.numeric(Y1_mean),
             dist_0 = as.numeric(dist_0),
             dist_1 = as.numeric(dist_1),
             CATE_mean = as.numeric(CATE_mean),
             seed = as.numeric(seed),
             n_train = as.numeric(n_train),
             contains_true_cate = as.numeric(contains_true_cate),
             CATE_error_bound = as.numeric(CATE_error_bound)
      ) %>%
      mutate(n_unimp = args$n_unimp,
            range_max = 2 * max(abs(c(summ_df$ymin, -summ_df$ymin, summ_df$ymax, -summ_df$ymax)))
            ) %>%
      group_by(dgp, range_max, fit) %>%
      summarise(contains_true_cate = mean(contains_true_cate),
                CATE_error_bound = mean(CATE_error_bound),
                CATE_error_bound_normalized = mean(CATE_error_bound)/range_max,
                n = n(), .groups = 'drop') %>%
      mutate(n_train = args$n_train, n_imp = args$n_imp, n_unimp = args$n_unimp, n_est = args$n_est) %>%
      return()
    },
    error = function(e) {
      message(paste("oops. file not found:", counter, file_path))
      return(data.frame())
    }
  )
}

# Read the args CSV file
args_df <- read_csv('./Experiments/variance/args.csv') %>% mutate(row = row.names(.))
# fit_add_list <- c('bias_corr', 'bias_corr_betting', 'boost_bias_corr', 'mml_bart', 'causal_forest_r', 'mml_cf', 'conformal')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
fit_add_list <- c('weighted_bias_corr_betting')#, 'bias_corr_betting', 'mml_bart', 'conformal', 'causal_forest_r')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
if(!('fit' %in% colnames(args_df))) {
  args_df$fit = NA
}
for(fit_add in fit_add_list) {
  args_df_fit_add <- unique(args_df[, c('dgp', 'n_train', 'n_est', 'n_imp', 'n_unimp', 'k', 'seed', 'row')])
  args_df_fit_add$fit <- fit_add
  args_df <- rbind(args_df, args_df_fit_add)
}
args_df <- args_df %>% filter(!is.na(fit))

# Create a list of dataframes
list_df <- pmap(args_df, function(...) {
  args <- list(...)
  
  summ_file_path <- paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_est_', args$n_est,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/summaries.csv'
  )
  summ_df = read_csv(summ_file_path, show_col_types = FALSE)

  read_df_w_error(args, args$row, summ_df) %>% return()
})
# Remove NULL elements (if any files were not found)
list_df <- compact(list_df)

# Combine all dataframes
overall_df <- list_df  %>%
bind_rows() %>%
  # filter(fit != 'naive') %>%
  # filter(fit != 'nn') %>%
  mutate(vim_or_not = ifelse(grepl(pattern = 'vim', x = fit), 'VIM', 'Not VIM')) %>%
  mutate(coverage_mg = (lb <= mg_cate) * (mg_cate <= ub))

# Group by and calculate mean
gb_df <- overall_df %>%
  group_by(n_train, n_unimp, n_est, dgp, range_max, fit) %>%
  summarise(contains_true_cate = mean(contains_true_cate),
            CATE_error_bound = mean(CATE_error_bound),
            CATE_error_bound_normalized = mean(CATE_error_bound_normalized),
            n = n(), .groups = 'drop')  %>%
  mutate(linear = ifelse(grepl(pattern = 'linear', dgp), 'Linear', 'Non-linear')) %>%
  mutate(corr = ifelse(grepl(pattern = '_corr', dgp), 'Cor.', 'Unc.')) %>%
  mutate(dim = ifelse(n_unimp == 100, 'High Dim', 'Low Dim')) %>%
  mutate(betting_or_not = ifelse(grepl(pattern = 'betting', x = fit), 'Betting', 'Not Betting')) %>%
  mutate(heteroskedastic = ifelse(grepl(pattern = 'hetero', x = dgp), 'Hetero', ' Homo'))
  # filter(dgp %in% c('linear', 'lihua_corr_hetero', 'lihua_corr_homo'))

png("trash1.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df, aes(x = n_train, y = contains_true_cate, color = fit, fill = fit)) +
    geom_line() +
    # geom_line(aes(x = n_train, y = contains_true_cate, color = fit, fill = fit), alpha = 0.5)
    # ggdist::stat_ribbon(fun = function(y) return(data.frame(ymin = quantile(y, 0), ymax = quantile(y, 100))), geom = 'ribbon') +
    # geom_line(aes(x = n_train, y = contains_true_cate, color = fit, fill = fit), alpha = 0.5) +
    # stat_summary(geom = 'line', fun = mean) +
    # ggdist::stat_ribbon(alpha = 0.5, .width = c(0.5)) +
    # stat_summary(
    # geom = "ribbon",
    # fun.data = function(y) {
    #   return(data.frame(ymin = quantile(y, 0), ymax = quantile(y, 1)))
    # }  ) +
    # geom_line(aes(x = n_train, y = mean(contains_true_cate), color = fit), alpha = 0.7) +
    # geom_boxplot(aes(x = n_est, y = contains_true_cate, color = fit, group = interaction(cut_width(n_est, 1000), fit)), alpha = 0.7) + #, linewidth = 1.5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_est + n_unimp~linear+corr+heteroskedastic,  scales = 'free_x') +
    xlab('#Observations in Training Set') +
    ylab('Marginal Coverage') +
    scale_color_brewer(palette = 'Dark2') +
    scale_fill_brewer(palette = 'Dark2') +
    theme_bw() +
    theme(text = element_text(size = 20),  axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

ggplot(gb_df, aes(x = n_train, y = 2 * CATE_error_bound_normalized, color = fit, fill = fit)) +
    # stat_summary(fun = mean, geom = 'line') +
    # ggdist::stat_ribbon(alpha = 0.5, .width= c(0.5)) +
    geom_line() +
    facet_grid(n_unimp~linear+corr+heteroskedastic,  scales = 'free_x') +
    xlab('#Observations in Training Set') +
    ylab('Normalized Interval Width') +
    scale_color_brewer(palette = 'Dark2') +
    scale_fill_brewer(palette = 'Dark2') +
    theme_bw() +
    theme(text = element_text(size = 20),  axis.text.x = element_text(angle = 45, hjust = 1))


png("trash4.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df) +
    geom_line(aes(x = n_train, y = contains_true_cate, color = fit), alpha = 0.7) + #, linewidth = 1.5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_est~dgp+vim_or_not,  scales = 'free_x') +
    xlab('#Training observations') +
    ylab('Coverage') +
    scale_color_brewer(palette = 'Dark2') +
    theme_bw() +
    theme(text = element_text(size = 20),  axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0.95, 1.1)
dev.off()

png("trash4_est.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_line(aes(x = n_est, y = contains_true_cate, color = fit), alpha = 0.7, linewidth = 1.5) +
      geom_hline(yintercept = 0.95, linetype = 'dashed') +
  facet_grid(dim + corr ~linear + heteroskedastic) +
  theme_bw() +
    xlab('#Matching observations') +
    ylab('Coverage') +
    scale_color_brewer(palette = 'Dark2') +
    theme_bw() +
    theme(text = element_text(size = 20),  axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

png("trash5.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df) +
    geom_line(aes(x = n_est, y = CATE_error_bound, color = fit), alpha = 0.7, linewidth = 1.5) +
    # geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_train~dgp+vim_or_not) +
    xlab('Training set size') +
    ylab('Interval Half-width') +
    theme_bw() +
    theme(text = element_text(size = 20),  axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_brewer(palette = 'Dark2')
dev.off()

p <- overall_df %>%
  group_by(n_train, dgp, fit, vim_or_not) %>%
  summarise(mse = mean(se))
ggplot(p) +
  geom_line(aes(x = n_train, y = mse, color = fit), alpha = 0.5, linewidth = 1.5) +
  facet_grid(~dgp+vim_or_not) +
    scale_color_brewer(palette = 'Set3')


ggplot(gb_df) +
    geom_line(aes(x = n_train, y = se, color = fit), alpha = 0.7, linewidth = 1.5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_est~dgp+vim_or_not) +
    xlab('Training set size') +
    ylab('Mean squared error') +
    theme_bw() +
    theme(text = element_text(size = 20), axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_brewer(palette = 'Dark2')
