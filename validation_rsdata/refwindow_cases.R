# refwindow_cases.R
# Ground-truth for rENA's ref_window_df, exported to CSV (no rpy2).
# Three cases chosen to pin the derived window formula:
#   result(a,b) = (a,b both in anchor row) OR (one in anchor, one in look-back)
#   i.e. pairs *between two earlier rows* are excluded (counted at their own anchor).
suppressMessages(library(rENA))

run_case <- function(name, mat, wsize) {
  df  <- as.data.frame(mat)
  res <- as.matrix(rENA:::ref_window_df(df, windowSize = wsize,
                                        windowForward = 0, binary = TRUE))
  colnames(res) <- c("C1C2", "C1C3", "C2C3")   # vector_to_ut order for 3 codes
  data.frame(case = name, windowSize = as.character(wsize),
             row = seq_len(nrow(res)) - 1L, res, row.names = NULL)
}

# Case A: basic back window, B=2. Tests within-row vs one-step cross-row.
A <- matrix(c(1,1,0,
              0,0,1,
              1,0,0), nrow = 3, byrow = TRUE)

# Case B: same rows, infinite look-back. Should differ from A at row 2.
# (row2 anchor C1 reaches BOTH earlier rows under Inf.)

# Case C: multi-code rows + binary saturation, B=2.
C <- matrix(c(1,1,0,
              1,0,1), nrow = 2, byrow = TRUE)

out <- rbind(
  run_case("A_back2",  A, 2),
  run_case("B_backInf", A, Inf),
  run_case("C_multicode_back2", C, 2)
)

dir.create("validation_algspec", showWarnings = FALSE)
write.csv(out, "validation_algspec/refwindow_cases.csv", row.names = FALSE)
print(out)
