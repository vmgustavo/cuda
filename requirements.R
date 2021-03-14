packages <- c(
  "torch", "Rcpp", "ggplot2", "collections", "pracma"
)

for (package in packages) {
  if(package %in% rownames(installed.packages())){
    print(paste0(package, " is installed"))
  } else {
    install.packages(package)
  }
}
