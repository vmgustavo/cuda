R_conv <- function(lag,x) {
    .C(
        "cconv", as.integer(lag), as.double(x), as.integer(length(x)), as.double(0.0)
    )[[4]]
}
