MML_maha = function(phi, weights=NULL){
  if(is.null(dim(phi))){
    if(is.null(weights))
      w = 1
    sapply(phi, function(x) sqrt(((phi-x)/w)^2))
  }else{
    if(is.null(weights))
      weights=diag(ncol(phi))
    apply(phi, 1, function(x) sqrt(mahalanobis(phi, x, weights, inverted = T)))
  }
}

MML_two_sample_maha = function(A, B, w=1){
  maha = function(a, B, w){sqrt(rowSums(t((a-t(B))/w)^2))}
  
  if (!is.matrix(A)){
    A = matrix(A, 1, length(A))
  }
  if (!is.matrix(B)){
    B = matrix(B, 1, length(B))
  }
  if(ncol(A) != ncol(B))
    stop(paste("Unable to compute distance. A has dimension:", 
               ncol(A), "and B has dimension:", ncol(B)))
  if (length(w) != ncol(A) && length(w) > 1)
    stop(paste("Weight vector has dimension:", length(w), 
               "and data has dimension", ncol(A)))
  t(apply(A, 1, maha, B, w))
}

MML_c = function(phi, q, Gamman=1){
  if(is.null(dim(phi)))
    ds = abs(phi - sample(phi, length(phi)))
  else
    ds = sqrt(rowSums((phi - phi[sample(nrow(phi), nrow(phi)), ])^2))
  
  quantile(ds, q) * Gamman
}

caliper_MG = function(i, distmat, caliper){
  ids = which(distmat[i, ] <= caliper)
  ids
}



MML_crf = function(MG, Y, Z, z){
  tr = MG[Z[MG]==z]
  mean(Y[tr])
}

MML_cate = function(MG, Y, Z){
  MML_crf(MG, Y, Z, 1) - MML_crf(MG, Y, Z, 0)
}

MML_pscore = function(MG, Z, z){
  mean(Z[MG]==z)
}

MML_caliper_cates = function(phi, Y, Z, weights=NULL, 
                             caliper=NULL, Gamman=1, q=0.01){
  n = length(Y)
  distmat = MML_maha(phi, weights)
  if(is.null(caliper))
    caliper = MML_c(phi, q, Gamman)
  cates = sapply(1:n, function(i){
    MG = caliper_MG(i, distmat, caliper)
    MML_cate(MG, Y, Z)
  })
  cates
}


MML_ate = function(phi, Y, Z, weights=NULL, caliper=NULL, 
                   Gamman=1, q=0.01){
  n = length(Y)
  distmat = MML_maha(phi, weights)
  if(is.null(caliper))
    caliper = MML_c(phi, q, Gamman)
  vals = sapply(1:n, function(i){
    MG = caliper_MG(i, distmat, caliper)
    c(MML_crf(MG, Y, Z, 1),
      MML_crf(MG, Y, Z, 0),
      MML_pscore(MG, Z, 1))
  })
  DR_ate(Y, Z, vals[1,], vals[2, ], vals[3, ])
}


knn_MG = function(i, distmat, k=NULL, Z=1, z=1){
  if(is.null(k))
    k = floor(sum(rep(1, ncol(distmat)) * (Z==z))^{1/4})
  m = max(distmat[i, ]) * (Z!=z)
  ids = order(distmat[i, ] + m)[1:k]
  ids
}

MML_knn_MGs = function(phi_estimation, phi_matching, Z, z, k=NULL, weights=1){
  if(!is.matrix(phi_estimation))
    phi_estimation = matrix(phi_estimation, length(phi_estimation), 1)
  if(!is.matrix(phi_matching))
    phi_matching = matrix(phi_matching, length(phi_matching), 1)
  
  distmat = MML_two_sample_maha(phi_estimation, phi_matching, weights)
  
  sapply(1:nrow(distmat), knn_MG, distmat, k, Z, z, simplify = FALSE)
}

MML_crf_from_MGs = function(MGs, Y, include_variance=FALSE){
  if (!include_variance)
    sapply(MGs, function(mg) mean(Y[mg]))
  else
    t(sapply(MGs, function(mg) c('est'=mean(Y[mg]), 'var'=var(Y[mg]))))
}

MML_knn_crfs = function(phi_estimation, phi_matching, Y, Z, z, k=NULL, 
                        weights=1, include_variance=FALSE){
  MGs = MML_knn_MGs(phi_estimation, phi_matching, Z, z, k, weights)
  MML_crf_from_MGs(MGs, Y, include_variance)
}

MML_knn_cates = function(phi_estimation1, phi_matching1, 
                         phi_estimation0, phi_matching0,
                         Y,Z, z1=1, z0=0, 
                         k1=NULL, k0=NULL, weights1=1, weights0=1){
  MGs1 = MML_knn_MGs(phi_estimation1, phi_matching1, Z, z1, k1, weights1)
  MGs0 = MML_knn_MGs(phi_estimation0, phi_matching0, Z, z0, k0, weights0)
  crf1 = MML_crf_from_MGs(MGs1, Y)
  crf0 = MML_crf_from_MGs(MGs0, Y)
  crf1-crf0
}

MML_knn_ate = function(phi_est1, phi_mat1, 
                       phi_est0, phi_mat0, 
                       Y_est, Y_mat,
                       Z_est, Z_mat,
                       e_est, 
                       z1=1, z0=0, k1=NULL, k0=NULL,
                       weights1=1, weights0=1, 
                       include_variance = FALSE){
  
  MGs1 = MML_knn_MGs(phi_est1, phi_mat1, Z_mat, z1, k1, weights1)
  MGs0 = MML_knn_MGs(phi_est0, phi_mat0, Z_mat, z0, k0, weights0)
  crf1 = MML_crf_from_MGs(MGs1, Y_mat)
  crf0 = MML_crf_from_MGs(MGs0, Y_mat)
  
  if(include_variance)
    return(DR_ate_and_var(Y_est, Z_est, crf1, crf0, e_est))
  else
    return(DR_ate(Y_est, Z_est, crf1, crf0, e_est))
}



MML_ate_cross_fit = function(phi1, phi0, Y, Z, pscore,
                             n_folds = 10, z1=1, z0=0, 
                             k1=NULL, k0=NULL,  
                             w1=1, w0=1, 
                             include_variance=FALSE){
  n = length(Y)
  shuf = sample(1:n, n, replace=FALSE)
  folds = cut(1:n, breaks = n_folds, labels=FALSE)
  if(include_variance)
    ates = data.frame('est'=rep(NA, n_folds), 'var'=rep(NA, n_folds))
  else
    ates = data.frame('est'=rep(NA, n_folds))
  
  for (fld in unique(folds)){
    est_idx = shuf[folds == fld]
    mat_idx = shuf[folds != fld]
    
    phi_est1 = phi1[est_idx]
    phi_mat1 = phi1[mat_idx]
    phi_est0 = phi0[est_idx]
    phi_mat0 = phi0[mat_idx]
    Y_est = Y[est_idx]
    Y_mat = Y[mat_idx]
    Z_est = Z[est_idx]
    Z_mat = Z[mat_idx]
    e_est = pscore[est_idx]
    
    ates[fld, ] = MML_knn_ate(phi_est1, phi_mat1, 
                            phi_est0, phi_mat0, 
                            Y_est, Y_mat,
                            Z_est, Z_mat,
                            e_est,
                            z1=z1, z0=z0, k1=k1, k0=k0, ke=ke, 
                            weights1=w1, weights0=w0, weightse=we,
                            include_variance=include_variance)
  }
    colMeans(ates)
}


# MML_knn_ate = function(phi, Y, Z, weights=NULL, k1=NULL, 
#                        k0=NULL, z1=1, z0=0){
#   n = length(Y)
# 
#   if(is.null(k1))
#     k1 = floor(sum(Z==z1)^{1/4})
#   if(is.null(k0))
#     k0 = floor(sum(Z==z0)^{1/4})
#   
#   vals = sapply(1:n, function(i){
#     MG1 = knn_MG(i, distmat, k1, Z, z1)
#     MG0 = knn_MG(i, distmat, k0, Z, z0)
#     MG =  knn_MG(i, distmat, (k0 + k1)/2, 1, 1)
#     c(MML_crf(MG1, Y, Z, z1),
#       MML_crf(MG0, Y, Z, z0),
#       MML_pscore(MG, Z, 1))
#   })
#   DR_ate(Y, Z, vals[1,], vals[2, ], vals[3, ])
# }
# 
# MML_knn_MGs = function(phi, Z=1, z=1, weights=NULL, k=NULL){
#   if(is.null(dim(phi)))
#     n = length(phi)
#   else
#     n = nrow(phi)
#   
#   if(is.null(k))
#     k = floor(sum(Z==z)^{1/4})
#   
#   distmat = MML_maha(phi, weights)
#   
#   sapply(1:n, knn_MG, i, distmat, k, Z, z)
# }
#   
# 
# MML_knn_crf = function(phi, Y, Z, weights=NULL, k=NULL, z=NULL){
#   n = length(Y)
#   distmat = MML_maha(phi, weights)
#   
#   if(is.null(k))
#     k = floor(sum(Z==z)^{1/4})
#   
#   vals = sapply(1:n, function(i){
#     MG = knn_MG(i, distmat, k, Z, z)
#     MML_crf(MG, Y, Z, z)
#   })
#   vals
# }
# 
# MML_knn_ate = function(phi1, phi0, phips, Y, Z, weights1=NULL, 
#                        weights0=NULL, weightsps=NULL, k1=NULL, 
#                        k0=NULL, kps=NULL, z1=1, z0=0){
#   
#   crf1 = MML_knn_crf(phi1, Y, Z, weights1, k1, z1)
#   crf0 = MML_knn_crf(phi0, Y, Z, weights0, k0, z0)
#   pscore = MML_knn_crf(phips, Z==z1, 1, weightsps, kps, 1)
# 
#   DR_ate(Y, Z, crf1, crf0, pscore)
# }


DR_ate = function(Y, Z, Y1hat, Y0hat, ehat){
  mean((Y1hat + Z*(Y-Y1hat)/ehat) - (Y0hat + (1-Z)*(Y-Y0hat)/(1-ehat)), na.rm = T)
}

compute_nuphi = function(phi, Z, c, gamma){
  library(kde)
  if (ncol(phi) > 3)
    fphi = kde(phi[,1:3], verbose = T, eval.points = phi[,1:3])
  else if (ncol(phi) == 2)
    fphi = kde(phi[,1:2], verbose = T, eval.points = phi[,1:2])
  else if (is.null(ncol(phi)))
    fphi = kde(phi, verbose = T, eval.points = phi)
}


DR_ate_and_var = function(Y, Z, Y1hat, Y0hat, ehat){
  psi = (Y1hat + Z*(Y-Y1hat)/ehat) - (Y0hat + (1-Z)*(Y-Y0hat)/(1-ehat))
  c("Estimate" = mean(psi, na.rm=T), "Std" = sd(psi, na.rm=T))
}
