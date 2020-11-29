#include <R.h>
#include <math.h>

void distmat_c(double *arr, int *feats, int *samples, double *res) {
    int i, j, k;
    for (i = 0; i < *samples; i++) {
        for (j = 0; j < *samples; j++) {
            double aux = 0;
            for (k = 0; k < *feats; k++) {
                // loop through features
                double diff = arr[i + *samples * k] - arr[j + *samples * k];
                aux += pow(diff, 2);
            }
            res[i + *samples * j] = sqrt(aux);
        }
    }
}
