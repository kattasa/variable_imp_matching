library(dplyr)
library(purrr)
library(readr)
library(ggplot2)

setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

# Function to read CSV with error handling
read_df_w_error <- function(file, counter) {
  # tryCatch(
  #   {
      read_csv(file, show_col_types = FALSE) %>%
      return()
    # },
    # error = function(e) {
    #   message(paste("oops. file not found:", counter, file))
    #   # message(e)
    #   return(NULL)
    # }
  # )
}

# Read the args CSV file
args_df <- read_csv('./Experiments/variance/scate_args.csv') %>% mutate(row = row.names(.))
# fit_add_list <- c('bias_corr', 'bias_corr_betting', 'boost_bias_corr', 'mml_bart', 'causal_forest_r', 'mml_cf', 'conformal')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
fit_add_list <- c('bias_corr_betting')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
if(!('fit' %in% colnames(args_df))) {
  args_df$fit = NA
}
for(fit_add in fit_add_list) {
  args_df_fit_add <- unique(args_df)
  args_df_fit_add$fit <- fit_add
  args_df <- rbind(args_df, args_df_fit_add)
}
args_df <- args_df %>% filter(!is.na(fit))

# Create a list of dataframes

list_df <- pmap(args_df, function(...) {
  args <- list(...)
  file_path <- paste0(
    './Experiments/variance/output_files/dgp_', args$dgp, 
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp, 
    '/n_train_', args$n_train,
    '/n_est_', args$n_est,
    '/sample_seed_', args$sample_seed,
    '/final_seed_', args$seed,
    '/', args$fit, '.csv'
  )
  read_df_w_error(file_path, args$row) %>%
  mutate(
    dgp = args$dgp,
    n_imp = args$n_imp,
    n_unimp = args$n_unimp,
    n_train = args$n_train,
    n_est = args$n_est,
    sample_seed = args$sample_seed,
    final_seed = args$seed
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
  mutate(heteroskedastic = ifelse(grepl(pattern = 'hetero', x = dgp), 'Heteroskedastic', 'Homoskedastic'))

# Group by and calculate mean
gb_df <- overall_df %>%
  group_by(n_unimp, n_train, n_est, dgp, fit, seed, betting_or_not, heteroskedastic) %>%
  summarise(across(everything(), mean), n = n(), .groups = 'drop') %>%
  mutate(linear = ifelse(grepl(pattern = 'linear', dgp), 'Linear', 'Non-linear')) %>%
  mutate(corr = ifelse(grepl(pattern = '_corr', dgp), 'Correlated', 'Uncorrelated')) %>%
  mutate(dim = ifelse(n_unimp == 100, 'High Dim', 'Low Dim'))

png("trash1.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df) +
  geom_boxplot(aes(x = fit, y = as.numeric(contains_true_cate), fill = fit)) +
  geom_hline(yintercept = 0.95, linetype = 'dashed') +
  labs(x = 'Estimator', y = 'Marginal Coverage') +
  facet_grid(dim + corr ~linear + heteroskedastic) +
  theme_bw() +
  theme(text = element_text(size = 20))
dev.off()

ggplot(gb_df) +
  geom_boxplot(aes(x = fit, y = as.numeric(CATE_error_bound), linetype = heteroskedastic)) +
  # geom_hline(yintercept = 0.95, linetype = 'dashed') +
  labs(x = 'Estimator', y = 'Interval 1/2 Width') +
  facet_grid(linear ~ dim + corr) +
  theme_bw() +
  theme(text = element_text(size = 20))
