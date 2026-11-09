# Part 4: repeat the Titanic loading, preprocessing, and modeling workflow in R.

numeric_features <- c("Age", "SibSp", "Parch", "Fare")
categorical_features <- c("Pclass", "Sex", "Embarked")
feature_names <- c(numeric_features, categorical_features)

parse_args <- function() {
  values <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% values) {
    cat("Usage: Rscript src/r/main.R [--data-dir src/data] [--output-dir outputs/r]\n")
    quit(status = 0)
  }
  args <- list(data_dir = "src/data", output_dir = "outputs/r")
  if (length(values) %% 2 != 0) stop("Each option must be followed by its path value.")
  if (length(values) > 0) {
    for (i in seq(1, length(values), by = 2)) {
      if (!values[i] %in% c("--data-dir", "--output-dir")) stop(paste("Unknown option:", values[i]))
      args[[gsub("-", "_", substring(values[i], 3))]] <- values[i + 1]
    }
  }
  args
}

read_input <- function(directory, filename) {
  path <- file.path(directory, filename)
  if (!file.exists(path)) stop(paste("Missing", path, "— download the CSVs as described in README.md."))
  data <- utils::read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
  required <- c("PassengerId", feature_names)
  if (filename == "train.csv") required <- c(required, "Survived")
  missing <- setdiff(required, names(data))
  if (length(missing)) stop(paste(filename, "is missing columns:", paste(missing, collapse = ", ")))
  if (!nrow(data) || anyNA(data$PassengerId) || anyDuplicated(data$PassengerId)) {
    stop(paste(filename, "must contain rows with unique, nonmissing PassengerId values."))
  }
  data
}

select_features <- function(data) {
  result <- data[feature_names]
  for (name in numeric_features) result[[name]] <- suppressWarnings(as.numeric(result[[name]]))
  for (name in categorical_features) {
    result[[name]] <- trimws(as.character(result[[name]]))
    result[[name]][!is.na(result[[name]]) & result[[name]] == ""] <- NA_character_
  }
  result
}

fit_preprocessor <- function(data) {
  medians <- sapply(data[numeric_features], stats::median, na.rm = TRUE)
  if (any(!is.finite(medians))) stop("A training numeric feature has no finite values.")
  modes <- sapply(data[categorical_features], function(x) {
    counts <- table(x)
    if (!length(counts)) stop("A training categorical feature is entirely missing.")
    names(counts)[which.max(counts)]
  })
  levels <- lapply(data[categorical_features], function(x) sort(unique(x[!is.na(x)])))
  filled <- data[numeric_features]
  for (name in numeric_features) filled[[name]][is.na(filled[[name]])] <- medians[[name]]
  centers <- sapply(filled, mean)
  scales <- sapply(filled, stats::sd)
  scales[is.na(scales) | scales == 0] <- 1
  list(medians = medians, modes = modes, levels = levels, centers = centers, scales = scales)
}

apply_preprocessor <- function(data, preprocessing) {
  result <- data
  for (name in numeric_features) {
    result[[name]][is.na(result[[name]])] <- preprocessing$medians[[name]]
    result[[name]] <- (result[[name]] - preprocessing$centers[[name]]) / preprocessing$scales[[name]]
  }
  for (name in categorical_features) {
    unseen_or_missing <- is.na(result[[name]]) | !result[[name]] %in% preprocessing$levels[[name]]
    result[[name]][unseen_or_missing] <- preprocessing$modes[[name]]
    result[[name]] <- factor(result[[name]], levels = preprocessing$levels[[name]])
  }
  result
}

fit_model <- function(data, y) {
  preprocessing <- fit_preprocessor(data)
  prepared <- apply_preprocessor(data, preprocessing)
  prepared$Survived <- y
  model <- stats::glm(
    Survived ~ Age + SibSp + Parch + Fare + Pclass + Sex + Embarked,
    data = prepared, family = stats::binomial(link = "logit"),
    control = stats::glm.control(maxit = 100)
  )
  if (!model$converged) stop("Logistic regression did not converge.")
  list(preprocessing = preprocessing, model = model)
}

classify <- function(fit, data) {
  probabilities <- stats::predict(fit$model, newdata = apply_preprocessor(data, fit$preprocessing), type = "response")
  if (anyNA(probabilities)) stop("Some predictions could not be calculated.")
  as.integer(probabilities >= 0.5)
}

show_preprocessing <- function(fit, label) {
  cat("[Q14]", label, "numeric fill values:\n")
  print(fit$preprocessing$medians)
  cat("[Q14]", label, "categorical fill values:\n")
  print(fit$preprocessing$modes)
}

main <- function() {
  args <- parse_args()
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Run Rscript src/r/install_packages.R first.")
  if (as.character(utils::packageVersion("jsonlite")) != "2.0.0") stop("This project requires jsonlite 2.0.0; run install_packages.R.")
  cat("[Part 4]", R.version.string, "; jsonlite", as.character(utils::packageVersion("jsonlite")), "\n")
  train <- read_input(args$data_dir, "train.csv")
  test <- read_input(args$data_dir, "test.csv")
  if (anyNA(train$Survived) || !setequal(unique(train$Survived), c(0, 1))) {
    stop("Training Survived values must include both classes, coded 0 and 1.")
  }
  cat("[Q13, Q17] Loaded", nrow(train), "training passengers and", nrow(test), "test passengers.\n")
  cat(sprintf("[Q14] Overall training survival rate: %.2f%%\n", 100 * mean(train$Survived)))
  cat("[Q14] Training survival rates by sex:\n")
  print(stats::aggregate(Survived ~ Sex, train, mean))
  cat("[Q14] Training survival rates by passenger class:\n")
  print(stats::aggregate(Survived ~ Pclass, train, mean))

  x <- select_features(train)
  x_test <- select_features(test)
  cat("[Q14] Features retained:", paste(feature_names, collapse = ", "), "\n")
  cat("[Q14] PassengerId, Name, Ticket, and Cabin are excluded from the model.\n")
  cat("[Q14] Training missing counts:\n")
  print(colSums(is.na(x)))
  cat("[Q14] Test missing counts:\n")
  print(colSums(is.na(x_test)))
  cat("[Q14] Numeric: training median imputation and scaling. Categories: training mode imputation and factor encoding.\n")
  cat("[Q14] Rows are retained; new categories are mapped to the fitted training mode.\n")

  set.seed(42)
  fit_rows <- sort(unlist(lapply(c(0, 1), function(label) {
    indices <- which(train$Survived == label)
    sample(indices, size = floor(0.8 * length(indices)))
  })))
  valid_rows <- setdiff(seq_len(nrow(train)), fit_rows)
  validation_fit <- fit_model(x[fit_rows, ], train$Survived[fit_rows])
  show_preprocessing(validation_fit, "Validation-fit")
  validation_accuracy <- mean(classify(validation_fit, x[valid_rows, ]) == train$Survived[valid_rows])
  cat("[Q15] Binomial logistic regression; stratified split:", length(fit_rows), "fit rows /", length(valid_rows), "validation rows; seed 42.\n")
  cat(sprintf("[Q18] Holdout VALIDATION accuracy: %.4f (%.2f%%).\n", validation_accuracy, 100 * validation_accuracy))

  full_fit <- fit_model(x, train$Survived)
  show_preprocessing(full_fit, "Full-training-fit")
  training_accuracy <- mean(classify(full_fit, x) == train$Survived)
  cat(sprintf("[Q16] Full-training accuracy (resubstitution): %.4f (%.2f%%).\n", training_accuracy, 100 * training_accuracy))
  predictions <- data.frame(PassengerId = test$PassengerId, Survived = classify(full_fit, x_test))
  cat("[Q17] Generated", nrow(predictions), "test predictions; threshold 0.5.\n")
  print(utils::head(predictions, 5))

  test_accuracy <- NULL
  if ("Survived" %in% names(test)) {
    if (anyNA(test$Survived) || !all(test$Survived %in% c(0, 1))) stop("Supplied test labels must be complete and coded 0 or 1.")
    test_accuracy <- mean(predictions$Survived == test$Survived)
    cat(sprintf("[Q18] Test accuracy using supplied labels: %.4f.\n", test_accuracy))
  } else {
    cat("[Q18] Kaggle test accuracy: N/A. test.csv has no true Survived labels.\n")
    cat("[Q18] The holdout score is validation accuracy, not Kaggle test accuracy.\n")
    cat("[Q18] gender_submission.csv is an example prediction baseline, not ground truth.\n")
  }

  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(predictions, file.path(args$output_dir, "predictions.csv"), row.names = FALSE)
  metrics <- list(
    language = "R", model = "logistic_regression", seed = 42,
    training_rows = nrow(train), validation_fit_rows = length(fit_rows),
    validation_rows = length(valid_rows), test_rows = nrow(test),
    training_accuracy = training_accuracy, validation_accuracy = validation_accuracy,
    test_accuracy = test_accuracy,
    test_accuracy_note = if (is.null(test_accuracy)) "Unavailable for the original Kaggle test.csv because it has no true labels." else "Evaluated using the supplied test Survived labels."
  )
  jsonlite::write_json(metrics, file.path(args$output_dir, "metrics.json"), auto_unbox = TRUE, pretty = TRUE, null = "null", digits = 8)
  cat("[Q19] Saved predictions.csv and metrics.json in", args$output_dir, "\n")
}

tryCatch(main(), error = function(e) {
  message("ERROR: ", conditionMessage(e))
  quit(status = 1)
})
