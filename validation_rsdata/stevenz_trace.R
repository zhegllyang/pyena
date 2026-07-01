# stevenz_trace.R — reconcile rENA(16) vs old-pyena(17)
# FirstGame::steven z, window.size.back = 4, pinned.
suppressMessages(library(rENA))
data(RS.data)

codes <- c("Data","Technical.Constraints","Performance.Parameters",
           "Client.and.Consultant.Requests","Design.Reasoning","Collaboration")

groups <- unique(RS.data$GroupName[RS.data$Condition == "FirstGame" &
                                   RS.data$UserName  == "steven z"])
cat("steven z group(s):", paste(groups, collapse=", "),
    "| n_groups =", length(groups), "\n")

# whole conversation (all speakers), original data order = turn order
conv <- RS.data[RS.data$Condition == "FirstGame" &
                RS.data$GroupName %in% groups, ]
cat("conv rows:", nrow(conv),
    "| steven z rows:", sum(conv$UserName == "steven z"), "\n")

# rENA per-row adjacency over the WHOLE conversation, window.size.back = 4
adj <- as.matrix(rENA:::ref_window_df(conv[, codes],
                                      windowSize = 4, windowForward = 0,
                                      binary = TRUE))
colnames(adj) <- apply(combn(codes, 2), 2, paste, collapse = " & ")

write.csv(cbind(UserName = conv$UserName, conv[, codes]),
          "stevenz_conv_codes.csv", row.names = FALSE)
write.csv(cbind(UserName = conv$UserName, as.data.frame(adj)),
          "stevenz_conv_refwindow_w4.csv", row.names = FALSE)

# immediate reality check: rENA cross-speaker count for steven z
cat("\nrENA cross-speaker  Data & TC =",
    sum(adj[conv$UserName == "steven z", "Data & Technical.Constraints"]), "\n")
