library(dplyr)
library(purrr)
library(readr)
library(ggplot2)

setwd('/usr/project/xtmp/sk787/variable_imp_matching/')

# Function to read CSV with error handling
read_df_w_error <- function(file) {
  tryCatch(
    {
      read_csv(file, show_col_types = FALSE)
    },
    error = function(e) {
      message(paste("oops. file not found:", file))
      return(NULL)
    }
  )
}

# Read the args CSV file
args_df <- read_csv('./Experiments/variance/args.csv')

# Create a list of dataframes
list_df <- pmap(args_df, function(...) {
  args <- list(...)
  file_path <- paste0(
    './Experiments/variance/output_files/dgp_', args$dgp,
    '/n_train_', args$n_train,
    '/n_imp_', args$n_imp,
    '/n_unimp_', args$n_unimp,
    '/k_', args$k,
    '/seed_', args$seed,
    '/', args$fit, '.csv'
  )
  read_df_w_error(file_path)
})

# Remove NULL elements (if any files were not found)
list_df <- compact(list_df)

# Combine all dataframes
overall_df <- bind_rows(list_df) %>%
  # filter(fit != 'naive') %>%
  filter(fit != 'nn') %>%
  mutate(vim_or_not = ifelse(grepl(pattern = 'vim', x = fit), 'VIM', 'Not VIM'))

# Group by and calculate mean
gb_df <- overall_df %>%
  group_by(dgp, seed, fit, vim_or_not) %>%
  summarise(across(everything(), mean), .groups = 'drop')

# If you need labels (commented out in the original code)
# labels <- unique(gb_df$fit)


png("trash2.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_point(aes(x = CATE_error_bound, y = contains_true_cate, color = fit, shape = vim_or_not), alpha = 0.7, size = 5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_wrap(~dgp,  scales = 'free_x') +
    xlab('Radius of confidence interval') +
    ylab('Coverage')
dev.off()

png("trash2_fit_v_coverage.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_col(aes(x = fit, y = contains_true_cate), alpha = 0.7, size = 5) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_grid(vim_or_not~dgp,  scales = 'free_y') +
    ylab('Coverage') +
    xlab('Estimator') +
    coord_flip()
dev.off()

png("trash2_fit_v_width.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_col(aes(x = fit, y = CATE_error_bound, fill = (contains_true_cate >= 0.95)), alpha = 0.7, size = 5) +
    facet_grid(vim_or_not~dgp,  scales = 'free_y') +
    xlab('Estimator') +
    ylab('Radius of confidence interval') +
    coord_flip()
dev.off()

# Group by and calculate mean
gb_df <- overall_df %>%
  group_by(n_train, dgp, fit, vim_or_not) %>%
  summarise(across(everything(), mean), .groups = 'drop')


png("trash4.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_line(aes(x = n_train, y = contains_true_cate, color = fit), alpha = 0.7) +
    geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_wrap(~dgp+vim_or_not,  scales = 'free_x') +
    xlab('#Training observations') +
    ylab('Coverage')
dev.off()

png("trash5.png", width = 1500, height = 1000, res = 300, units = 'px')
ggplot(gb_df) +
    geom_line(aes(x = n_train, y = CATE_error_bound, color = fit), alpha = 0.7) +
    # geom_hline(yintercept = 0.95, linetype = 'dashed') +
    facet_wrap(~dgp+vim_or_not,  scales = 'free_x') +
    xlab('Training set size') +
    ylab('Coverage') +
    ylim(0, 80)
dev.off()