library(dplyr)
library(purrr)
library(readr)
library(ggplot2)

setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

# Function to read CSV with error handling
read_df_w_error <- function(file, counter) {
  tryCatch(
    {
      cat(counter)
      read_csv(file, show_col_types = FALSE) %>%
      return()
    },
    error = function(e) {
      message(paste("oops. file not found:", counter, file))
      # message(e)
      return(data.frame())
    }
  )
}



# Read the args CSV file
args_df <- read_csv('./Experiments/variance/randomization_args.csv') %>% mutate(row = row.names(.))
# fit_add_list <- c('bias_corr', 'bias_corr_betting', 'boost_bias_corr', 'mml_bart', 'causal_forest_r', 'mml_cf', 'conformal')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
# fit_add_list <- c('knn_match_true_prop_true_or_true_prog',
# 'knn_match_est_prop_est_or_rf_prog',
# 'causal_forest',
# 'knn_match_est_prop_est_or_rf_prog_no_bias')

# if(!('fit' %in% colnames(args_df))) {
#   args_df$fit = NA
# }
# for(fit_add in fit_add_list) {
#   args_df_fit_add <- unique(args_df)
#   args_df_fit_add$fit <- fit_add
#   args_df <- rbind(args_df, args_df_fit_add)
# }
# args_df <- args_df %>% filter(!is.na(fit))

# Create a list of dataframes

list_df <- pmap(args_df, function(...) {
  args <- list(...)
  file_path <- paste0(
    './Experiments/variance/randomization_files/dgp_', args$dgp, 
    '/n_train_', args$n_train,
    '/n_est_', args$n_est,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp, 
    '/seed_', args$seed,
    '/', args$fit, '_k', args$k, '.csv'
  )
  read_df_w_error(file_path, args$row) %>%
  mutate(
    id = row.names(.),
    dgp = args$dgp,
    n_imp = args$n_imp,
    n_unimp = args$n_unimp,
    n_train = args$n_train,
    n_est = args$n_est,
    # sample_seed = args$sample_seed,
    final_seed = args$seed,
    k = args$k
  )
}
)

# Remove NULL elements (if any files were not found)
list_df <- compact(list_df)

# Combine all dataframes
overall_df <- list_df  %>%
bind_rows() %>%
  # filter(fit != 'naive') %>%
  # filter(fit != 'nn') %>%
  mutate(betting_or_not = ifelse(grepl(pattern = 'betting', x = fit), 'Betting', 'Not Betting')) %>%
  mutate(heteroskedastic = ifelse(grepl(pattern = 'hetero', x = dgp), 'Hetero', 'Homo')) %>%
  mutate(contains_true_cate = (lb <= cate_true) * (cate_true <= ub)) %>%
  mutate(contains_mg_ate = (lb <= mg_ate) * (cate_true <= mg_ate)) %>%
  mutate(interval_length = (ub - lb)/(ymax - ymin)) %>%
  mutate(fit = ifelse(fit == 'causal_forest', 'Causal Forest',
                      ifelse(fit == 'knn_match_true_prop_true_or_true_prog', 'Prog. Matching w/ True Params',
                             ifelse(fit == 'knn_match_est_prop_est_or_rf_prog', 'Prog. Matching w/ Estimated Params',
                                    'Prog. Matching w/out Bias'))
                                    ))

# Group by and calculate mean
gb_df <- overall_df %>%
  mutate(k_coef = k/as.integer(sqrt(n_est)) ) %>%
  group_by(
    n_unimp, 
    n_train, 
    n_est, 
    dgp, 
    fit, 
    k_coef,
    betting_or_not, 
    heteroskedastic
    # id
    ) %>%
  summarise(across(everything(), mean), n = n(), .groups = 'drop') %>%
  mutate(linear = ifelse(grepl(pattern = 'linear', dgp), 'Linear', 'Non-linear')) %>%
  mutate(corr = ifelse(grepl(pattern = '_corr', dgp), 'Correlated', 'Uncorrelated')) %>%
  mutate(dim = ifelse(n_unimp == 100, 'High Dim', 'Low Dim'))

png("./plots/randomization_vary_k.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df) +
  # stat_summary(geom = 'line', fun = 'mean', aes(x = k, y = as.numeric(contains_true_cate), color = fit), size = 1.5) +
  # stat_summary(geom = 'bar', fun = 'mean', aes(x = k, y = as.numeric(contains_true_cate), fill = fit), size = 1.5, position = 'dodge') +
  stat_summary(geom = 'bar', fun = 'mean', aes(x = k, y = as.numeric(interval_length), fill = fit), size = 1.5, position = 'dodge') +
  geom_hline(yintercept = 0.95, linetype = 'dashed') +
  labs(y = 'Marginal Coverage') +
  facet_grid(dim + corr ~ heteroskedastic + n_est, scales = 'free_x') +
  theme_bw() +
  scale_color_brewer(palette = 'Dark2') +
  guides(color = guide_legend(nrow = 2)) +
  theme(text = element_text(size = 30), legend.position = 'top',
        axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

overall_df %>%
filter(dgp == 'lihua_uncorr_heteroskedastic') %>%
filter(n_unimp == 10) %>%
filter(n_est == 1000) %>%
filter(fit == 'Prog. Matching w/ True Params') %>%
filter(k == 16 * as.integer(sqrt(n_est))) %>%
group_by(id) %>%
summarize(across(everything(), mean), n = n(), .groups = 'drop') %>%
ggplot() +
geom_point(aes(x = X_0, y = X_1, color = as.factor(contains_true_cate)), size = 10) +
theme_bw() +
labs(color = 'Coverage')

ggplot(gb_df) +
  stat_summary(geom = 'line', fun = 'mean', aes(x = k_coef, y = as.numeric(contains_mg_ate), color = fit), size = 1.5) +
  geom_hline(yintercept = 0.95, linetype = 'dashed') +
  labs(y = 'Marginal Coverage') +
  facet_grid(dim + corr ~ heteroskedastic + n_est) +
  theme_bw() +
  scale_color_brewer(palette = 'Dark2') +
  guides(color = guide_legend(nrow = 2)) +
  theme(text = element_text(size = 30), legend.position = 'top')

ggplot(gb_df) +
  stat_summary(geom = 'line', fun = 'mean', aes(x = k_coef, y = as.numeric(interval_length), color = fit), size = 1.5) +
  labs(y = 'Interval width') +
  facet_grid(dim + corr ~ heteroskedastic, scales = 'free_y') +
  theme_bw() +
  scale_color_brewer(palette = 'Dark2') +
  guides(color = guide_legend(nrow = 2)) +
  theme(text = element_text(size = 30), legend.position = 'top')


ggplot(gb_df) +
  stat_summary(geom = 'point', fun = 'mean', aes(x = contains_true_cate, y = as.numeric(interval_length), color = fit, shape = factor(k_coef)), size = 1.5) +
  labs(y = 'Interval width') +
  geom_vline(xintercept = 0.95, linetype = 'dashed') +
  facet_grid(dim + corr ~ heteroskedastic + n_est, scales = 'free_y') +
  theme_bw() +
  scale_color_brewer(palette = 'Dark2') +
  # guides(color = guide_legend(nrow = 2)) +
  theme(text = element_text(size = 30))



png("randomization_trash1.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df) +
  geom_line(aes(x = n_est, y = as.numeric(contains_true_cate), color = fit), size = 1.5) +
  geom_hline(yintercept = 0.95, linetype = 'dashed') +
  labs(x = '#Matching Units', y = 'Marginal Coverage') +
  facet_grid(dim + corr ~ heteroskedastic) +
  theme_bw() +
  ylim(0.75, 1) +
  scale_color_brewer(palette = 'Dark2') +
  guides(color = guide_legend(nrow = 2)) +
  theme(text = element_text(size = 30), legend.position = 'top')
dev.off()


ggplot(gb_df) +
  geom_line(aes(x = n_est, y = as.numeric(interval_length), color = fit)) +
  labs(x = '#Matching Units', y = 'Relative Interval Length') +
  facet_grid(dim + corr ~ heteroskedastic) +
  theme_bw() +
  ylim(0, 1) +
  theme(text = element_text(size = 20)) 
