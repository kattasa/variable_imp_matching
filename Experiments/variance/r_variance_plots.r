library(dplyr)
library(purrr)
library(readr)
library(ggplot2)

setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

# Function to read CSV with error handling
read_df_w_error <- function(file, counter) {
  tryCatch(
    {
      read_csv(file, show_col_types = FALSE) %>%
      mutate(Y0_mean = as.numeric(Y0_mean),
             Y1_mean = as.numeric(Y1_mean),
             dist_0 = as.numeric(dist_0),
             dist_1 = as.numeric(dist_1),
             CATE_mean = as.numeric(CATE_mean),
             seed = as.numeric(seed),
             n_train = as.numeric(n_train)
      ) %>%
      # mutate(index = row.names()) %>%
      return()
    },
    error = function(e) {
      message(paste("oops. file not found:", counter, file))
      # message(e)
      return(NULL)
    }
  )
}

# Read the args CSV file
args_df <- read_csv('./Experiments/variance/args.csv') %>% mutate(row = row.names(.))
# fit_add_list <- c('bias_corr', 'bias_corr_betting', 'boost_bias_corr', 'mml_bart', 'causal_forest_r', 'mml_cf', 'conformal')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
fit_add_list <- c('bias_corr_betting')#, 'xlearner_r', 'slearner_r', 'tlearner_r')
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
  read_df_w_error(file_path, args$row)
})
# Remove NULL elements (if any files were not found)
list_df <- compact(list_df)

# Combine all dataframes
overall_df <- list_df  %>%
bind_rows() %>%
  # filter(fit != 'naive') %>%
  # filter(fit != 'nn') %>%
  mutate(vim_or_not = ifelse(grepl(pattern = 'vim', x = fit), 'VIM', 'Not VIM'))

# Group by and calculate mean
gb_df <- overall_df %>%
  group_by(n_train, n_est, dgp, fit, vim_or_not) %>%
  summarise(across(everything(), mean), n = n(), .groups = 'drop')


png("trash4.png", width = 1500, height = 1000, res = 100, units = 'px')
ggplot(gb_df %>% filter(fit != 'mml_cf')) +
    geom_line(aes(x = n_train, y = contains_true_cate, color = fit), alpha = 0.7, linewidth = 1.5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_est~dgp+vim_or_not,  scales = 'free_x') +
    xlab('#Training observations') +
    ylab('Coverage') +
    scale_color_brewer(palette = 'Dark2') +
    theme_bw() +
    theme(text = element_text(size = 20))
dev.off()

png("trash4_est.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df %>% filter(dgp == 'exp')) +
    geom_line(aes(x = n_est, y = contains_true_cate, color = fit), alpha = 0.7, linewidth = 1.5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(n_train~dgp+vim_or_not,  scales = 'free_x') +
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
