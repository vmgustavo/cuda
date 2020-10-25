#include "Rcpp.h"

// [[Rcpp::export]]
Rcpp::NumericVector convolve_cpp(const Rcpp::NumericVector& a, const Rcpp::NumericVector& b) {
    // Declare loop counters, and vector sizes
    int i, j;
    int na = a.size();
    int nb = b.size();
    int nab = na + nb -1;

    // Create vector filled with 0
    Rcpp::NumericVector ab(nab);

    // Crux of the algorithm
    for (i = 0; i < na; i++) {
        for (j = 0; j < nb; j++) {
            ab[i + j] += a[i] * b[j];
            }
        }

    // Return result
    return ab;
}
