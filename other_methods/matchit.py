"""
Created on Sat Apr 18 15:20:12 2020
@author: Harsh
"""

import pandas as pd
import numpy as np

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()


def matchit(outcome, treatment, data, method='nearest', replace=False,
            k_est=1):
    if replace:
        replace = 'TRUE'
    else:
        replace = 'FALSE'
    data.to_csv('data.csv', index=False)
    formula_cov = treatment + ' ~ '
    i = 0
    for cov in data.columns:
        if cov != outcome and cov != treatment:
            if i != 0:
                formula_cov += '+'
            formula_cov += str(cov)
            i += 1
    print(formula_cov)
    string = """
    library('MatchIt')
    data <- read.csv('data.csv')
    r <- matchit( %s, method = "%s", data = data, replace = %s)
    matrix <- r$match.matrix[,]
    names <- as.numeric(names(r$match.matrix[,]))
    mtch <- data[as.numeric(names(r$match.matrix[,])),]
    hh <- data[as.numeric(names(r$match.matrix[,])),'%s']- data[as.numeric(r$match.matrix[,]),'%s']
    data2 <- data
    data2$%s <- 1 - data2$%s
    r2 <- matchit( %s, method = "%s", data = data2, replace = %s)
    matrix2 <- r2$match.matrix[,]
    names2 <- as.numeric(names(r2$match.matrix[,]))
    mtch2 <- data2[as.numeric(names(r2$match.matrix[,])),]
    hh2 <- data2[as.numeric(r2$match.matrix[,]),'%s'] - data2[as.numeric(names(r2$match.matrix[,])),'%s']
    """ % (formula_cov, method, replace, outcome, outcome, treatment, treatment, formula_cov, method, replace, outcome,
           outcome)

    # string = """
    # library('MatchIt')
    # data <- read.csv('data.csv')
    # r <- matchit( %s, method = "%s", data = data, replace = %s, ratio = %d)
    # matrix <- r$match.matrix[,]
    # names <- as.numeric(rownames(r$match.matrix[,]))
    # mtch <- data[names, '%s']
    # # cmtch <- rowMeans(array(data[as.numeric(matrix), '%s'], dim=dim(matrix)))
    # # hh <- mtch - cmtch
    #
    # data2 <- data
    # data2$%s <- 1 - data2$%s
    # r2 <- matchit( %s, method = "%s", data = data2, replace = %s, ratio = %d)
    # matrix2 <- r2$match.matrix[,]
    # names2 <- as.numeric(rownames(r2$match.matrix[,]))
    # mtch2 <- data2[names2, '%s']
    # # cmtch2 <- rowMeans(array(data2[as.numeric(matrix2), '%s'], dim=dim(matrix2)))
    # # hh2 <- mtch2 - cmtch2
    # """ % (
    # formula_cov, method, replace, k_est, outcome, outcome, treatment, treatment,
    # formula_cov, method, replace, k_est, outcome, outcome)

    psnn = SignatureTranslatedAnonymousPackage(string, "powerpack")
    match = psnn.mtch
    match2 = psnn.mtch2
    t_hat = pd.DataFrame(np.hstack((np.array(psnn.hh), np.array(psnn.hh2))),
                         index=list(psnn.names.astype(int)) + list(
                             psnn.names2.astype(int)),
                         columns=['CATE'])
    ate = np.mean(t_hat['CATE'])
    return ate, t_hat