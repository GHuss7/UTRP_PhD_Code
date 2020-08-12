library(multiplex)

# workingDirectory <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/Skripsie DSS/DSS"
workingDirectory <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/ORSSA/Article/Article/fig/6_case_study/PlotGenerator"
setwd(workingDirectory)
rm(workingDirectory)

loadSpecificArchive <- FALSE
loadSpecificWorkspace <- FALSE

# Write in different folders
folderPath <- as.vector(NULL)
#folderPath[1] <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/Skripsie DSS/DSS/Data_files"
#folderPath[2] <- "C:/Users/Günther/OneDrive - Stellenbosch University/Skripsie Presentation - ORSSA/slides/Numerical_results/Data_files"
folderPath[1] <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/ORSSA/Article/Article/fig/6_case_study/PlotGenerator"


# Load a specific archive
if(loadSpecificArchive){ # Load the archive with the given number of routes 
  load(paste("./SavedRData/archive6Routes17092018a.Rdata", sep = ""))
}else{
  archive <- as.list(NULL)
}

if(loadSpecificWorkspace){ # Load the workspace with the given number of routes 
  load(paste("./SavedRData/workspace6Routes2019-03-13d.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace6Routes2019-03-08a.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace6Routes17092018a.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace6Routes08102018a.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace6Routes2018-10-17a.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace8Routes2018-10-17b.Rdata", sep = ""))
  #load(paste("./SavedRData/workspace6Routes2018-10-19a.Rdata", sep = ""))
  
}

create.Dat <- function(archiveDF,normalised,path){
  
  if(!normalised){
    plotData <-archiveDF[,-c(1,4,5)]
  }else{
    plotData <-archiveDF[,-c(1,2,3)]
  }
  write.dat(plotData,path)
} # end creat.Dat


addUndomFrontsTogether <- function(attFront1,attFront2){
counter <- 1
archiveComb <- as.list(NULL)

  for (j in 1:length(attFront1)) {
    archiveComb[[counter]] <- attFront1[[j]]
    counter <- counter + 1
  }
  
  for (j in 1:length(attFront2)) {
    archiveComb[[counter]] <- attFront2[[j]]
    counter <- counter + 1
}
return(archiveComb)
}

#### Take the adding attainment fronts approach ------

### Old
# combinedAtt <- as.list(NULL)
#  for (i in 1:(length(archiveList)-1)) {
#      combinedAtt[[i]] <- addUndomFrontsTogether(archiveList[[i]],archiveList[[i+1]])
#   }
#  removedAtt <- as.list(NULL)
#  for (i in 1:length(combinedAtt)) {
#      removedAtt <- removeAllDominatedSolutionsMinMin(combinedAtt[[i]])
#    }

## Adding the fronts together =======
 
 # Pareto front (get values)
 archiveDFPareto <- createArchiveDF(archive)
 archiveMaxf1DF <- archiveDFPareto[which.max(archiveDFPareto$f1),]
 archiveMaxf1 <- archiveMaxf1DF[,-c(1,4,5)]
 archiveMinf1DF <- archiveDFPareto[which.min(archiveDFPareto$f1),]
 archiveMinf1 <- archiveMinf1DF[,-c(1,4,5)]
 benchMarkData <- read.csv("./Mandl's Swiss Network/MumfordResultsParetoFront.csv")
 names(benchMarkData) <- c("f2","f1")
 benchMarkData <- benchMarkData[c("f1","f2")]
 
 attainmentList <- as.list(NULL)

for (j in 1:length(archiveList)) {
  
  attainmentList[[j]] <- removeAllDominatedSolutionsMinMin(archiveList[[j]])
  
}

combinedAtt <- as.list(NULL)
currentAtt <- as.list(NULL)
currentAtt <- archiveList[[i]]
removedAtt <- as.list(NULL)

currentAtt <- archiveList[[1]]

for (i in 1:(length(archiveList)-1)) {
  combinedAtt[[i]] <- addUndomFrontsTogether(currentAtt,archiveList[[i+1]])
  removedAtt[[i]] <- removeAllDominatedSolutionsMinMin(combinedAtt[[i]])
  currentAtt <- removedAtt[[i]]
}

all(createArchiveDF(removedAtt[[length(removedAtt)]]) == createArchiveDF(archive))


for (fi in 1:length(folderPath)) { # for loop for writing data in
  

for (i in 1:length(attainmentList)) { # create individual attainment fronts
  
  subDir <- paste(numAllowedRoutes,"routes",sep="")
  dataPath <- paste(folderPath[fi],"/",numAllowedRoutes,"routesAtt","/attainment",i,sep = "")
  
  dir.create(file.path(folderPath[fi], subDir), showWarnings = FALSE)
  create.Dat(createArchiveDF(attainmentList[[i]]) , FALSE, dataPath)
  
}
  
for (i in 1:length(combinedAtt)) { # create combined sets of attainment fronts
    
    subDir <- paste(numAllowedRoutes,"routes",sep="")
    dataPath <- paste(folderPath[fi],"/",numAllowedRoutes,"routesAtt","/combAtt",i,sep = "")
    
    dir.create(file.path(folderPath[fi], subDir), showWarnings = FALSE)
    create.Dat(createArchiveDF(combinedAtt[[i]]) , FALSE, dataPath)
    
}  
  
  for (i in 1:length(removedAtt)) { # create the intermediary undominatedfronts
    
    subDir <- paste(numAllowedRoutes,"routes",sep="")
    dataPath <- paste(folderPath[fi],"/",numAllowedRoutes,"routesAtt","/remAtt",i,sep = "")
    
    dir.create(file.path(folderPath[fi], subDir), showWarnings = FALSE)
    create.Dat(createArchiveDF(removedAtt[[i]]) , FALSE, dataPath)
    
  }  

  # Creates the benchmark and pareto fronts
  dataPath <- paste(folderPath[fi],"/",numAllowedRoutes,"routesAtt",sep = "")
  
  create.Dat(archiveDFPareto,FALSE,dataPath)
  write.dat(archiveMaxf1,dataPath)
  write.dat(archiveMinf1,dataPath)
  write.dat(benchMarkData,dataPath)
  
} # end folder path for loop


