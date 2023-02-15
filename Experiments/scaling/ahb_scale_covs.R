library('AHB')

data_folder <- Sys.getenv("RESULTS_FOLDER")
n_covs <- as.integer(Sys.getenv("N_COVS"))

df <- read.table(paste(data_folder, "df.csv", sep="/"), sep=",", header=TRUE, nrows=2048, colClasses=c(rep("numeric", 1), rep("integer", 1), rep("numeric", n_covs), rep("NULL", 1024-n_covs)))
subset <- sample(seq_len(nrow(df)), size = floor(0.5 * nrow(df)))
df1 <- df[subset, ]
df2 <- df[-subset, ]

start <- Sys.time()
suppressMessages(ahb1 <- AHB_fast_match(df1, holdout=df2, treated_column_name = "T", outcome_column_name = "Y"))
suppressMessages(ahb2 <- AHB_fast_match(df2, holdout=df1, treated_column_name = "T", outcome_column_name = "Y"))
write(difftime(Sys.time(), start, units='secs'), stdout())