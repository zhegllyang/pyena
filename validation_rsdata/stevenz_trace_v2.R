# stevenz_trace_v2.R — per-conversation windowing (fixes cross-group leak)
suppressMessages(library(rENA))
data(RS.data)

codes <- c("Data","Technical.Constraints","Performance.Parameters",
           "Client.and.Consultant.Requests","Design.Reasoning","Collaboration")
pair  <- "Data & Technical.Constraints"

sz <- RS.data[RS.data$Condition=="FirstGame" & RS.data$UserName=="steven z", ]
cat("steven z rows:", nrow(sz), "\n")
print(table(droplevels(factor(sz$GroupName))))
groups <- unique(as.character(sz$GroupName))

ref_pair <- function(d) {
  a <- as.matrix(rENA:::ref_window_df(d[, codes], windowSize=4,
                                      windowForward=0, binary=TRUE))
  colnames(a) <- apply(combn(codes,2),2,paste,collapse=" & ")
  a
}

# (1) CROSS-SPEAKER: each conversation windowed over ALL its speakers, then
#     sum steven z's own rows across his conversations.
cross <- 0
for (g in groups) {
  conv <- RS.data[RS.data$Condition=="FirstGame" & RS.data$GroupName==g, ]
  cnt  <- sum(ref_pair(conv)[conv$UserName=="steven z", pair])
  cat(sprintf("  [cross] FirstGame::%-10s convrows=%3d szrows=%2d Data&TC=%d\n",
              g, nrow(conv), sum(conv$UserName=="steven z"), cnt))
  cross <- cross + cnt
}
cat("CROSS-SPEAKER (per-conv)  Data & TC =", cross, "\n\n")

# (2) WITHIN-SPEAKER: steven z's rows ONLY, still split per conversation
#     (reproduces old pyena's within-speaker stanza window).
within <- 0
for (g in groups) {
  szc <- RS.data[RS.data$Condition=="FirstGame" & RS.data$GroupName==g &
                 RS.data$UserName=="steven z", ]
  cnt <- sum(ref_pair(szc)[, pair])
  cat(sprintf("  [within] FirstGame::%-10s szrows=%2d Data&TC=%d\n",
              g, nrow(szc), cnt))
  within <- within + cnt
}
cat("WITHIN-SPEAKER (sz rows only) Data & TC =", within, "\n")
