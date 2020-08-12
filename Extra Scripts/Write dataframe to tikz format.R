# DSS_Write archive into Tiks readable code
workingDirectory <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/Skripsie DSS/DSS"
setwd(workingDirectory)
rm(workingDirectory)

source("./DSS_Functions.R")
source("./DSS_Visualisation_Functions.R")

load(paste("./SavedRData/workspace6Routes2019-05-08a.Rdata", sep = ""))

activeArchive <- archive2

archiveDF <- createArchiveDF(activeArchive)

orderedArchive <- archiveDF[order(archiveDF$f2,FALSE),]

Tikz_text <- NULL

for (i in 1:nrow(orderedArchive)) {
  Tikz_text <- paste(Tikz_text,"(",orderedArchive[i,2],",",orderedArchive[i,3],") ", sep = "")
}
Tikz_text

