suppressWarnings(library(reticulate))
suppressWarnings(library(grf))

RESULTS_FOLDER <- Sys.getenv("RESULTS_FOLDER")
ACIC_FOLDER <- Sys.getenv("ACIC_FOLDER")
SPLIT_NUM <- Sys.getenv("SPLIT_NUM")
RANDOM_STATE <- Sys.getenv("RANDOM_STATE")

acic_results_folder <- paste(RESULTS_FOLDER, ACIC_FOLDER, sep="/")
split_num = strtoi(SPLIT_NUM)

source_python("/hpc/home/qml/linear_coef_matching/Experiments/acic_error_and_runtime/runtimes/pickle_load_split.py")

idx <- pickle_load_split(acic_results_folder, split_num)

df_train <- read.csv(paste(acic_results_folder, "df_dummy_data.csv", sep=""))[unlist(idx[2]),]
df_train <- subset(df_train, select = -c(X) )

sample <- read.csv(paste(acic_results_folder, "df_dummy_data.csv", sep=""))[sample(unlist(idx[2]), 1),]
sample <- subset(sample, select = -c(X,T,Y) )


start <- Sys.time()

Ycrf = df_train$Y
Tcrf = df_train$T
X = subset(df_train, select = -c(T,Y))
crf <- grf::causal_forest(X, Ycrf, Tcrf, seed=RANDOM_STATE)
tauhat = predict(crf, sample)

cat(as.character(as.numeric(Sys.time() - start)), '\n')
