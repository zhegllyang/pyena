# bench_rena_big.R — time rENA on the large row-scaling datasets only.
# cd benchmark_data && Rscript bench_rena_big.R
suppressMessages(library(rENA))

N_WARMUP <- 1L
N_RUNS   <- 3L   # fewer runs; these are large

datasets <- c("R_100k", "R_200k", "R_500k")

time_median <- function(fn) {
  for (i in seq_len(N_WARMUP)) fn()
  ts <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0 <- proc.time()[["elapsed"]]; fn(); ts[i] <- proc.time()[["elapsed"]] - t0
  }
  median(ts)
}

results <- data.frame()
for (name in datasets) {
  df <- read.csv(paste0(name, ".csv"), stringsAsFactors = FALSE)
  code_cols <- grep("^Code", names(df), value = TRUE)
  units <- data.frame(unit = df$unit_id)
  conv  <- data.frame(conv = df$conv_id)
  codes <- df[, code_cols]

  t_accum <- time_median(function() {
    ena.accumulate.data(units = units, conversation = conv,
                        codes = codes, window.size.back = 4)
  })
  accum <- ena.accumulate.data(units = units, conversation = conv,
                               codes = codes, window.size.back = 4)
  t_svd <- time_median(function() ena.make.set(accum))

  results <- rbind(results, data.frame(
    name = name, rows = nrow(df), units = length(unique(df$unit_id)),
    t_accum_s = round(t_accum, 4), t_full_s = round(t_accum + t_svd, 4)
  ))
  cat(sprintf("  %-8s rows=%6d  accum=%.4fs  full=%.4fs\n",
              name, nrow(df), t_accum, t_accum + t_svd))
}
write.csv(results, "bench_rena_big_timings.csv", row.names = FALSE)
cat("\nwrote bench_rena_big_timings.csv\n")
