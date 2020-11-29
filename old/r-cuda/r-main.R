# REQUIREMENTS
# Rcpp
#   http://www.rcpp.org/
#   install.packages("Rcpp")

# IMPORTS
library(Rcpp)
sourceCpp('c-main.cpp')

# =============================================================================
# HOST FUNCTION DEFINITION - R
h_distmat <- function(arr) {
    size = length(arr)
    dist <- matrix(0, size, size)
    for (i in seq(length(feat))) {
        for (j in seq(length(feat))) {
            dist[i, j] = sqrt((feat[j] - feat[i])^2)
        }
    }
    return(dist)
}

# =============================================================================

# SETUP
size = 5
feat <- abs(as.integer(10 * rnorm(size)))

# =============================================================================
### CALCULATE DISTANCE MATRIX
# native R
st <- Sys.time()
nat <- as.matrix(dist(feat, method='euclidean', diag=TRUE, upper=TRUE))
en <- Sys.time()
sprintf('Time Elapsed (s): %.06f', difftime(en, st, units='secs')[[1]])

# implemented
st <- Sys.time()
imp <- h_distmat(feat)
en <- Sys.time()
sprintf('Time Elapsed (s): %.06f', difftime(en, st, units="secs")[[1]])

# =============================================================================
# ASSERT CORRECT DISTANCE RESULTS
all(nat == imp)
