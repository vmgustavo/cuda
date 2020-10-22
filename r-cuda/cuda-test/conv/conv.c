void cconv(int *l, double *x, int *n, double *s) {
    double  *y = x + (*n - *l), *z = x + *l, *u = x;
    while ( u < y)
        *s += *u++ * *z++;
}
