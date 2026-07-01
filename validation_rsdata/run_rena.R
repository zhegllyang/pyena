# run_rena.R
# Run rENA pipeline on RS.data and export each stage's output as CSV
# for downstream comparison against pyena.
#
# Note: rENA 0.3.1 uses `connection.counts` (not `adjacency.vectors`)
# and stores unit IDs in `meta.data$ENA_UNIT`.

library(rENA)
data(RS.data)

cat("RS.data dimensions:", nrow(RS.data), "rows x", ncol(RS.data), "cols\n")
cat("Conditions:", unique(as.character(RS.data$Condition)), "\n")
cat("Unique GroupNames:", length(unique(RS.data$GroupName)), "\n")
cat("Unique UserNames:", length(unique(RS.data$UserName)), "\n")

code_cols <- c("Data", "Technical.Constraints", "Performance.Parameters",
               "Client.and.Consultant.Requests", "Design.Reasoning",
               "Collaboration")
unit_cols <- c("Condition", "UserName")
conv_cols <- c("Condition", "GroupName")

# Stage 1: Accumulate
cat("\n[Stage 1] ena.accumulate.data ...\n")
accum <- ena.accumulate.data(
  units    = RS.data[, unit_cols],
  conversation = RS.data[, conv_cols],
  codes    = RS.data[, code_cols],
  window.size.back = 4
)
cat("  connection.counts dim:", dim(accum$connection.counts), "\n")
cat("  meta.data dim:", dim(accum$meta.data), "\n")

# Stage 2: SVD projection (no rotation)
cat("\n[Stage 2] ena.make.set (SVD) ...\n")
set_svd <- ena.make.set(accum)
cat("  points dim:", dim(set_svd$points), "\n")

# Stage 3: Means rotation
cat("\n[Stage 3] ena.make.set (means rotation) ...\n")
set_mr <- ena.make.set(
  accum,
  rotation.by     = ena.rotate.by.mean,
  rotation.params = list(
    accum$meta.data$Condition == "FirstGame",
    accum$meta.data$Condition == "SecondGame"
  )
)
cat("  MR points dim:", dim(set_mr$points), "\n")

cat("\n[Export] Writing CSVs ...\n")

# Helper: convert data.table with ENA-specific column types to a plain
# data.frame with numeric/character columns suitable for write.csv.
to_plain_df <- function(dt) {
  df <- as.data.frame(dt)
  # Convert ena.metadata columns to character
  for (col in names(df)) {
    if (inherits(df[[col]], "ena.metadata")) {
      df[[col]] <- as.character(df[[col]])
    }
    # Convert ena.co.occurrence columns to numeric
    if (inherits(df[[col]], "ena.co.occurrence")) {
      df[[col]] <- as.numeric(df[[col]])
    }
  }
  df
}

# Unit metadata
meta_df <- to_plain_df(accum$meta.data)
write.csv(meta_df, "rena_units.csv", row.names = FALSE)
cat("  rena_units.csv:", nrow(meta_df), "rows x", ncol(meta_df), "cols\n")

# Adjacency / connection counts (rename from connection.counts → adjacency for consistency)
adj_df <- to_plain_df(accum$connection.counts)
write.csv(adj_df, "rena_adjacency.csv", row.names = FALSE)
cat("  rena_adjacency.csv:", nrow(adj_df), "rows x", ncol(adj_df), "cols\n")

# SVD coordinates
svd_df <- to_plain_df(set_svd$points)
write.csv(svd_df, "rena_svd.csv", row.names = FALSE)
cat("  rena_svd.csv:", nrow(svd_df), "rows x", ncol(svd_df), "cols\n")

# Means rotation coordinates
mr_df <- to_plain_df(set_mr$points)
write.csv(mr_df, "rena_mr.csv", row.names = FALSE)
cat("  rena_mr.csv:", nrow(mr_df), "rows x", ncol(mr_df), "cols\n")

# Raw data (for pyena to read same input)
write.csv(RS.data, "RS_data.csv", row.names = FALSE)
cat("  RS_data.csv:", nrow(RS.data), "rows\n")

cat("\nDone.\n")
