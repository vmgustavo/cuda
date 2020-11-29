ksmooth3 <- function(x, xpts, h) {
    n <- length(x)
    nxpts <- length(xpts)
    dens <- .C(
        "kernel_smooth", as.double(x), as.integer(n),
        as.double(xpts), as.integer(nxpts), as.double(h),
        result = double(length(xpts))
    )
    dens[["result"]]
}
