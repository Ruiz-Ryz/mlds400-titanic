# jsonlite writes the metrics file. Model fitting uses stats from base R.
required_version <- "2.0.0"

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    as.character(utils::packageVersion("jsonlite")) != required_version) {
  archive <- tempfile(fileext = ".tar.gz")
  urls <- c(
    paste0("https://cran.r-project.org/src/contrib/jsonlite_", required_version, ".tar.gz"),
    paste0("https://cran.r-project.org/src/contrib/Archive/jsonlite/jsonlite_", required_version, ".tar.gz")
  )
  downloaded <- FALSE
  for (url in urls) {
    downloaded <- tryCatch({
      utils::download.file(url, archive, mode = "wb", quiet = TRUE)
      TRUE
    }, error = function(e) FALSE)
    if (downloaded) break
  }
  if (!downloaded) stop("Could not download the pinned jsonlite source from CRAN.")
  utils::install.packages(archive, repos = NULL, type = "source")
}

stopifnot(as.character(utils::packageVersion("jsonlite")) == required_version)
cat("Installed/verified jsonlite", required_version, "\n")
