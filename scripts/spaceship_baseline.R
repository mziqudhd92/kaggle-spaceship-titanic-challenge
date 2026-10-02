# Spaceship Titanic — baseline model in R
#
# Pipeline (mirrors spaceship_baseline.py):
#   1. Load train/test CSVs.
#   2. Feature engineering: split Cabin, group size from PassengerId, total spend.
#   3. Preprocess: median / most-frequent imputation + one-hot encoding.
#   4. Random Forest (ranger) with 5-fold cross-validation.
#   5. Fit on the full training set and write submission.csv.
#
# Run:  Rscript spaceship_baseline.R

library(data.table)
library(ranger)
set.seed(42)

# Works locally and inside a Kaggle R notebook (Data > Add Input > competition)
DATA_DIR <- if (dir.exists("/kaggle/input/spaceship-titanic")) {
  "/kaggle/input/spaceship-titanic"
} else {
  "/Users/mz/.zcode/workspace/default/spaceship-titanic"
}

# ---- feature engineering -----------------------------------------------------

engineer_features <- function(df) {
  stopifnot(is.data.frame(df))

  # Cabin is "deck/number/side" (side is P=port, S=starboard)
  cabin_parts <- tstrsplit(df$Cabin, "/", fixed = TRUE, fill = NA)
  df[, Deck := cabin_parts[[1]]]
  df[, CabinNum := suppressWarnings(as.numeric(cabin_parts[[2]]))]
  df[, Side := cabin_parts[[3]]]

  # PassengerId is "group_member": e.g. "0002_01" -> group 0002, member 01
  group_id <- sub("_.*$", "", df$PassengerId)
  df[, GroupSize := .N, by = .(group_id)]

  # Spending across all onboard amenities; cryo-sleepers spend ~nothing
  spend_cols <- c("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
  df[, TotalSpend := rowSums(.SD, na.rm = TRUE), .SDcols = spend_cols]
  df[, SpentAnything := as.integer(TotalSpend > 0)]

  df[, IsChild := as.integer(Age < 13)]
  df
}

# ---- preprocessing -----------------------------------------------------------

median_impute <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  x
}

mode_impute <- function(x) {
  x[is.na(x)] <- names(sort(table(x, useNA = "no"), decreasing = TRUE))[1]
  x
}

preprocess <- function(df) {
  df <- copy(df)
  for (col in c("Age", "CabinNum")) df[, (col) := median_impute(get(col))]
  for (col in c("HomePlanet", "CryoSleep", "Destination", "VIP", "Deck",
                "Side")) {
    df[, (col) := mode_impute(as.character(get(col)))]
  }
  # one-hot encode all factor/character features
  design <- model.matrix(
    ~ 0 + HomePlanet + CryoSleep + Destination + VIP + Deck + Side,
    data = df
  )
  design <- as.data.table(design)
  design[, names(design) := lapply(.SD, as.numeric)]
  cbind(df[, c("Age", "CabinNum", "TotalSpend", "GroupSize", "IsChild",
               "SpentAnything"), with = FALSE], design)
}

# ---- main --------------------------------------------------------------------

train <- fread(file.path(DATA_DIR, "train.csv"))
test <- fread(file.path(DATA_DIR, "test.csv"))

cat("train:", nrow(train), "rows | test:", nrow(test), "rows\n\n")

train <- engineer_features(train)
test <- engineer_features(test)

X <- preprocess(train)
y <- train$Transported
X_test <- preprocess(test)

# 5-fold cross-validation
folds <- sample(rep(1:5, length.out = nrow(X)))
cv_scores <- vapply(seq_len(5), function(k) {
  fit <- ranger(
    x = X[folds != k], y = factor(y[folds != k]),
    num.trees = 300, min.node.size = 2, num.threads = 0
  )
  preds <- as.logical(predict(fit, data = X[folds == k])$predictions)
  mean(preds == y[folds == k])
}, numeric(1))

cat(sprintf("5-fold CV accuracy: %.4f (+/- %.4f)\n",
            mean(cv_scores), sd(cv_scores)))
cat("Per fold:", sprintf("%.4f", cv_scores), "\n")

# final model on the full training set
final_fit <- ranger(
  x = X, y = factor(y),
  num.trees = 300, min.node.size = 2, num.threads = 0
)
preds <- as.logical(predict(final_fit, data = X_test)$predictions)

submission <- data.table(PassengerId = test$PassengerId,
                         Transported = ifelse(preds, "True", "False"))
# Kaggle reads submissions from /kaggle/working; local runs use the project dir
out_dir <- if (dir.exists("/kaggle/working")) "/kaggle/working" else DATA_DIR
fwrite(submission, file.path(out_dir, "submission_r.csv"))
cat(sprintf("\nWrote submission_r.csv in %s (%d rows)\n", out_dir, nrow(submission)))
