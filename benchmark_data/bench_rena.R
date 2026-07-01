# bench_rena.R — run rENA on each synthetic dataset, record stage timings,
# and export adjacency for parity checking. Reads/writes in benchmark_data/.
#
# Usage:  cd benchmark_data && Rscript bench_rena.R
suppressMessages(library(rENA))

N_WARMUP <- 1L
N_RUNS   <- 5L

manifest <- read.csv("manifest.csv", stringsAsFactors = FALSE)

# robust median timer: warmup then N_RUNS, return median seconds
time_median <- function(expr_fn) {
  for (i in seq_len(N_WARMUP)) expr_fn()
  ts <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0 <- proc.time()[["elapsed"]]
    expr_fn()
    ts[i] <- proc.time()[["elapsed"]] - t0
  }
  median(ts)
}

to_plain_df <- function(dt) {
  df <- as.data.frame(dt)
  for (col in names(df)) {
    if (inherits(df[[col]], "ena.metadata"))      df[[col]] <- as.character(df[[col]])
    if (inherits(df[[col]], "ena.co.occurrence")) df[[col]] <- as.numeric(df[[col]])
  }
  df
}

results <- data.frame()

for (r in seq_len(nrow(manifest))) {
  name <- manifest$name[r]
  path <- paste0(name, ".csv")
  df <- read.csv(path, stringsAsFactors = FALSE)

  code_cols <- grep("^Code", names(df), value = TRUE)
  units <- data.frame(unit = df$unit_id)
  conv  <- data.frame(conv = df$conv_id)
  codes <- df[, code_cols]

  # time accumulation (stage i) and full set incl. SVD (stage ii)
  t_accum <- time_median(function() {
    ena.accumulate.data(units = units, conversation = conv,
                        codes = codes, window.size.back = 4)
  })
  accum <- ena.accumulate.data(units = units, conversation = conv,
                               codes = codes, window.size.back = 4)
  t_svd <- time_median(function() ena.make.set(accum))

  # export adjacency once, for parity check against pyena
  adj_df <- to_plain_df(accum$connection.counts)
  write.csv(adj_df, paste0(name, "_rena_adj.csv"), row.names = FALSE)

  results <- rbind(results, data.frame(
    name = name, rows = nrow(df), units = length(unique(df$unit_id)),
    codes = length(code_cols),
    t_accum_s = round(t_accum, 4),
    t_full_s  = round(t_accum + t_svd, 4),  # accumulate + make.set
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  %-12s rows=%6d  accum=%.4fs  full=%.4fs\n",
              name, nrow(df), t_accum, t_accum + t_svd))
}

write.csv(results, "bench_rena_timings.csv", row.names = FALSE)
cat("\nwrote bench_rena_timings.csv and *_rena_adj.csv\n")
