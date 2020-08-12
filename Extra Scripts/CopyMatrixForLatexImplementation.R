# Create latex matrix:

# Create and format a distance matrix S
distMat <- read.csv("./Mandl's Swiss Network/NodesTransitTimes(min).csv")
distMat <- formatDistMatrix(distMat)

# Create and format the demand matrix
demMatrix <- read.csv("./Mandl's Swiss Network/TransitdemandMatrix.csv")
demMatrix <- formatDemandMatrix(demMatrix)

#######

distVect <- NULL

for(i in 1:nrow(distMat)){
  for(j in 1:nrow(distMat)){
    
    distVect <- c(distVect,distMat[i,j])
    
  }
}

writeClipboard(paste(distVect, sep = " ",collapse = "&"))

######

demVect <- NULL

for(i in 1:nrow(demMatrix)){
  for(j in 1:nrow(demMatrix)){
    
    demVect <- c(demVect,demMatrix[i,j])
    
  }
  
}

writeClipboard(paste(demVect, sep = " ",collapse = "&"))
