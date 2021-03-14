setwd('/home/gustavo/repos/cuda')

# IMPORTS
library(collections)
library(Rcpp)
library(torch)
library(pracma)
sourceCpp('distmat_cpp.cpp')

dyn.load("distmat_c.so")

# =============================================================================
distmat_imp <- function(arr) {
  size <- dim(arr)[1]
  dist <- matrix(0, size, size)
  for (i in seq(size)) {
    for (j in seq(size)) {
      dist[i, j] <- sqrt(sum((arr[i,] - arr[j,])^2))
    }
  }
  return(dist)
}

distmat_c <- function(mat) {
  nfeats <- dim(mat)[2]
  nsamples <- dim(mat)[1]
  # column first arr
  arr <- as.vector(mat)
  result <- .C(
    "distmat_c",
    as.double(arr),
    as.integer(nfeats),
    as.integer(nsamples),
    res = double(nsamples * nsamples)
  )$res
  matrix(result, nrow=nsamples)
}

distmat_cuda_dyn <- function(mat) {
  nfeats <- dim(mat)[2]
  nsamples <- dim(mat)[1]
  # column first arr
  arr <- as.vector(mat)
  result <- .C(
    "distmat_cuda",
    as.double(arr),
    as.integer(nfeats),
    as.integer(nsamples),
    res = double(nsamples * nsamples)
  )$res
  matrix(result, nrow=nsamples)
}

distmat_torch <- function(mat) {
  result <- as_array(torch_pdist(torch_tensor(mat)))
  squareform(result)
}
# =============================================================================

# SETUP
size <- 1000
feats <- 10
feat_1D <- matrix(abs((9 * rnorm(size))), nrow=size)
feat_2D <- matrix(abs((9 * rnorm(size * feats))), nrow=size)
times <- dict()

# =============================================================================
### CALCULATE DISTANCE MATRIX
# native R
st <- Sys.time()
nat_1D <- as.matrix(dist(feat_1D, method='euclidean', diag=TRUE, upper=TRUE))
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('nati-1d', aux)
sprintf('Native 1D Time Elapsed (s): %.06f', aux)

st <- Sys.time()
nat_2D <- as.matrix(dist(feat_2D, method='euclidean', diag=TRUE, upper=TRUE))
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('nati-2d', aux)
sprintf('Native 2D Time Elapsed (s): %.06f', aux)

# implemented
#st <- Sys.time()
#imp_1D <- distmat_imp(feat_1D)
#en <- Sys.time()
#aux <- difftime(en, st, units='secs')[[1]]
#times$set('impl-1d', aux)
#sprintf('Implem 1D Time Elapsed (s): %.06f', aux)
#
#st <- Sys.time()
#imp_2D <- distmat_imp(feat_2D)
#en <- Sys.time()
#aux <- difftime(en, st, units='secs')[[1]]
#times$set('impl-2d', aux)
#sprintf('Implem 2D Time Elapsed (s): %.06f', aux)

# Rcpp
st <- Sys.time()
cpp_1D <- cpp_distmat(feat_1D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('rcpp-1d', aux)
sprintf('Rcpp 1D Time Elapsed (s): %.06f', aux)

st <- Sys.time()
cpp_2D <- cpp_distmat(feat_2D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('rcpp-2d', aux)
sprintf('Rcpp 2D Time Elapsed (s): %.06f', aux)

# C
st <- Sys.time()
dyn_1D <- distmat_c(feat_1D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('dynl-1d', aux)
sprintf('DynLoad 1D Time Elapsed (s): %.06f', aux)

st <- Sys.time()
dyn_2D <- distmat_c(feat_2D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('dynl-2d', aux)
sprintf('DynLoad 2D Time Elapsed (s): %.06f', aux)

# CUDADyn
st <- Sys.time()
dyn_1D <- distmat_c(feat_1D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('dynl-1d', aux)
sprintf('DynLoad 1D Time Elapsed (s): %.06f', aux)

st <- Sys.time()
dyn_2D <- distmat_c(feat_2D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('dynl-2d', aux)
sprintf('DynLoad 2D Time Elapsed (s): %.06f', aux)

# TORCH
st <- Sys.time()
trc_1D <- distmat_torch(feat_1D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('trch-1d', aux)
sprintf('Torch 1D Time Elapsed (s): %.06f', aux)

st <- Sys.time()
trc_2D <- distmat_torch(feat_2D)
en <- Sys.time()
aux <- difftime(en, st, units='secs')[[1]]
times$set('trch-2d', aux)
sprintf('Torch 2D Time Elapsed (s): %.06f', aux)

# =============================================================================
# ASSERT CORRECT DISTANCE RESULTS
#all(nat_1D == imp_1D)
#all(nat_2D == imp_2D)

all(nat_1D == cpp_1D)
all(nat_2D == cpp_2D)

all(nat_1D == dyn_1D)
all(nat_2D == dyn_2D)

all(nat_1D == trc_1D)
all(nat_2D == trc_2D)

print_dict <- function (d) {
  for (key in d$keys()) {
    print(paste0(key, ': ', d$get(key)))
  }
}
print_dict(times)
