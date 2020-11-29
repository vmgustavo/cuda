#include "Rcpp.h"

// [[Rcpp::export]]
Rcpp::NumericVector vecadd_cpp(const Rcpp::NumericVector& a, const Rcpp::NumericVector& b, int size) {
    // Declare loop counters, and vector sizes
    int i;

    // Create vector filled with 0
    Rcpp::NumericVector res(size);

    // Crux of the algorithm
    for (i = 0; i < size; i++) {
        res = a[i] + b[i];
    }

    // Return result
    return res;
}
