suppressWarnings(library(reticulate))
suppressWarnings(library(dbarts))

RESULTS_FOLDER <- Sys.getenv("RESULTS_FOLDER")
ACIC_FOLDER <- Sys.getenv("ACIC_FOLDER")
SPLIT_NUM <- Sys.getenv("SPLIT_NUM")


acic_results_folder <- paste(RESULTS_FOLDER, ACIC_FOLDER, sep="/")
split_num = strtoi(SPLIT_NUM)

source_python("/hpc/home/qml/linear_coef_matching/Experiments/acic_error_and_runtime/runtimes/pickle_load_split.py")

idx <- pickle_load_split(acic_results_folder, split_num)

df_train <- read.csv(paste(acic_results_folder, "df_data.csv", sep=""))[unlist(idx[2]),]
df_train <- subset(df_train, select = -c(X) )
binary <- length(unique(df_train[["Y"]])) == 2

# For some reason bart can only do prediction with >1 sample for binary outcomes
if (binary) {
  this_sample <- sample(unlist(idx[2]), 2)
} else {
  this_sample <-  sample(unlist(idx[2]), 1)
}

sample <- read.csv(paste(acic_results_folder, "df_data.csv", sep=""))[this_sample,]
sample <- subset(sample, select = -c(X,T,Y) )


start <- Sys.time()
Xc <- subset(df_train[df_train$T == 0,], select = -c(T,Y))
Yc <- subset(df_train[df_train$T == 0,], select = c(Y))
Xt <- subset(df_train[df_train$T == 1,], select = -c(T,Y))
Yt <- subset(df_train[df_train$T == 1,], select = c(Y))
if (binary) {
  cate <- mean(pnorm(dbarts::bart(as.matrix(Xt), as.matrix(Yt), as.matrix(sample), verbose=FALSE)[3][[1]][,1])) - mean(pnorm(dbarts::bart(as.matrix(Xc), as.matrix(Yc), as.matrix(sample), verbose=FALSE)[3][[1]][,1]))
} else {
  cate <- mean(dbarts::bart(as.matrix(Xt), as.matrix(Yt), as.matrix(sample), verbose=FALSE)[8][1]) - mean(dbarts::bart(as.matrix(Xc), as.matrix(Yc), as.matrix(sample), verbose=FALSE)[8][1])
}
total_time <- Sys.time() - start
cat(as.character(as.numeric(total_time)))
