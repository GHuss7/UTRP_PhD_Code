stringD <- NULL

for (i in 1:nrow(demandMatrix)){
  for (j in 1:ncol(demandMatrix)){
    
    stringD <- paste(stringD,demandMatrix[i,j],sep = "")
    if(j != ncol(demandMatrix)){
      stringD <- paste(stringD,"&",sep = "")
    }
    
  }
  stringD <- paste(stringD,"\\",sep = "")
  
}
stringD
