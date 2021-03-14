#include <math.h>
#include "Rcpp.h"

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_distmat(const Rcpp::NumericMatrix& mat) {
    // Declare loop counters, and vector sizes
    int i, j, k;
    int samples = mat.nrow();
    int features = mat.ncol();

    Rcpp::NumericMatrix res(samples, samples);

    // Crux of the algorithm
    for (i = 0; i < samples; i++) {
        for (j = 0; j < samples; j++) {
            if (j > i) {
                double aux = 0;
                for (k = 0; k < features; k++) {
                    aux += pow(mat(i, k) - mat(j, k), 2);
                }
                res(i, j) = sqrt(aux);
            }
        }
    }

    // Return result
    return res;
}
