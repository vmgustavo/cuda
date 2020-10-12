R_hello <- function(n) {
    .C("hello", as.integer(n))
}
