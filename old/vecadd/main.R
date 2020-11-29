# SETUP
size = 5
a <- abs(as.integer(10 * rnorm(size)))
b <- abs(as.integer(10 * rnorm(size)))

# =============================================================================
### CALCULATE DISTANCE MATRIX
# native R
st <- Sys.time()
nat = a + b
en <- Sys.time()
sprintf('Time Elapsed (s): %.06f', difftime(en, st, units='secs')[[1]])

# cpp
library(Rcpp)
sourceCpp('vecadd.cpp')

st <- Sys.time()
cpp = vecadd_cpp(a, b, size)
en <- Sys.time()
sprintf('Time Elapsed (s): %.06f', difftime(en, st, units='secs')[[1]])

# cuda


# =============================================================================
# ASSERT CORRECT DISTANCE RESULTS
all(nat == cpp)
