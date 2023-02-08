suppressWarnings(library('MatchIt'))

save_folder <- Sys.getenv("SAVE_FOLDER")

df <- read.csv(paste(save_folder, "df.csv", sep="/"))
cols <- colnames(df)
cols <- paste(cols, collapse=" + ")
cols <- substr(cols, 1, nchar(cols)-8)

start <- Sys.time()
m <- matchit(as.formula(paste("T ~ ", cols)), data=df, method="genetic")
write(Sys.time() - start, stdout())
