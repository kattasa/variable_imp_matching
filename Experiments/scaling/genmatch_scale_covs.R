suppressWarnings(library('MatchIt'))

data_folder <- Sys.getenv("RESULTS_FOLDER")
n_covs <- as.integer(Sys.getenv("N_COVS"))

df <- read.table(paste(data_folder, "df.csv", sep="/"), sep=",", header=TRUE, nrows=2048, colClasses=c(rep("numeric", 1), rep("integer", 1), rep("numeric", n_covs), rep("NULL", 1024-n_covs)))
cols <- colnames(df)
cols <- paste(cols, collapse=" + ")
cols <- substr(cols, 9, nchar(cols))

start <- Sys.time()
m <- matchit(as.formula(paste("T ~ ", cols)), data=df, method="genetic")
write(difftime(Sys.time(), start, units='secs'), stdout())
