# export_nodes.R — save rENA node positions + locate the placement algorithm
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

# export node positions (first 2 dims are what we render)
nodes <- as.data.frame(set$rotation$nodes)
write.csv(nodes, "rena_nodes_svd.csv", row.names = FALSE)
cat("wrote rena_nodes_svd.csv:", nrow(nodes), "nodes\n")

# also need line.weights (edge weights per unit) for edge thickness
lw <- as.data.frame(set$line.weights)
write.csv(lw, "rena_line_weights.csv", row.names = FALSE)
cat("wrote rena_line_weights.csv:", nrow(lw), "x", ncol(lw), "\n")

# --- locate the placement function ---
cat("\n=== ena.make.set: where nodes come from ===\n")
# find which function computes rotation$nodes
cat("body of ena.make.set (node-related lines):\n")
src <- deparse(body(ena.make.set))
print(grep("node|position|lws|rotation\\$", src, value=TRUE, ignore.case=TRUE))
