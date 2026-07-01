library(rENA)
data(RS.data)

code_cols <- c("Data", "Technical.Constraints", "Performance.Parameters",
               "Client.and.Consultant.Requests", "Design.Reasoning",
               "Collaboration")
unit_cols <- c("Condition", "UserName")
conv_cols <- c("Condition", "GroupName")

accum <- ena.accumulate.data(
  units    = RS.data[, unit_cols],
  conversation = RS.data[, conv_cols],
  codes    = RS.data[, code_cols],
  window.size.back = 4
)

cat("\n=== Top-level structure of accum ===\n")
cat("Class:", class(accum), "\n")
cat("Names:\n")
print(names(accum))

cat("\n=== adjacency.vectors ===\n")
av <- accum$adjacency.vectors
cat("Class:", class(av), "\n")
cat("Type:", typeof(av), "\n")
cat("Length (if vector):", length(av), "\n")
cat("Names (if list):\n")
print(names(av))
cat("Dim (if any):\n")
print(dim(av))
cat("First few elements:\n")
print(head(av, 3))

cat("\n=== meta.data ===\n")
md <- accum$meta.data
cat("Class:", class(md), "\n")
cat("Dim:", dim(md), "\n")
cat("Names:\n")
print(names(md))

cat("\n=== connection.counts (if exists) ===\n")
if ("connection.counts" %in% names(accum)) {
  cc <- accum$connection.counts
  cat("Class:", class(cc), "\n")
  cat("Dim:", dim(cc), "\n")
  print(head(cc, 2))
} else {
  cat("(not found)\n")
}

cat("\n=== Looking for unit-level adjacency anywhere ===\n")
for (nm in names(accum)) {
  obj <- accum[[nm]]
  cat(sprintf("  %-30s class=%-20s ", nm, paste(class(obj), collapse="/")))
  if (is.data.frame(obj) || is.matrix(obj)) {
    cat(sprintf("dim=[%d x %d]", nrow(obj), ncol(obj)))
  }
  cat("\n")
}
