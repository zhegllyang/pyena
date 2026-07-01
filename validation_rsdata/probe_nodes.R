# probe_nodes.R — find where rENA stores network node positions
suppressMessages(library(rENA))
data(RS.data)

codes <- c("Data","Technical.Constraints","Performance.Parameters",
           "Client.and.Consultant.Requests","Design.Reasoning","Collaboration")

accum <- ena.accumulate.data(
  units = RS.data[, c("Condition","UserName")],
  conversation = RS.data[, c("Condition","GroupName")],
  codes = RS.data[, codes],
  window.size.back = 4
)
set <- ena.make.set(accum)

cat("=== names(set) ===\n"); print(names(set))
cat("\n=== set$rotation names ===\n"); print(names(set$rotation))
cat("\n=== does set$rotation$nodes exist? ===\n")
print(tryCatch(dim(set$rotation$nodes), error=function(e) "no"))
print(tryCatch(head(as.data.frame(set$rotation$nodes)), error=function(e) "n/a"))

cat("\n=== set$points names / class ===\n")
print(class(set$points)); print(dim(set$points))

# rENA often stores node positions under $rotation$nodes or as an attr;
# also check the 'ena.nodes' / line.weights surface
cat("\n=== any node-like components ===\n")
for (nm in names(set)) {
  obj <- set[[nm]]
  if (!is.null(dim(obj)) && nrow(as.data.frame(obj)) == length(codes)) {
    cat("  candidate:", nm, "-> dim", paste(dim(obj), collapse="x"), "\n")
  }
}
