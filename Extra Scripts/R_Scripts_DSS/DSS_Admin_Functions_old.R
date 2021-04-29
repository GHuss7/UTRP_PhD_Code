# A set of functions that are used to simplify saving, and retrieving of files in an automated fashion

saveResultsAsCSV = function(dataframe, nameOfFile, run_number, resultsDir){
  # a function that saves a dataframe as a cvs file under the name nameOfFile associated with a run 
  # number in the directory folder resultsDir
  if (missing(nameOfFile)) {
    write.csv(dataframe,file = paste(resultsDir,"/",deparse(quote(dataframe)),"_", run_number,".csv",sep = ""), row.names = TRUE)
    
  } else {
    write.csv(dataframe,file = paste(resultsDir,"/",nameOfFile,"_", run_number,".csv",sep = ""), row.names = TRUE)
  }
  
}

createResultsDirectory = function(resultsDir){
  # creates a new results directory and returns the name with the name incremented by 1
  
  while(!ifelse(!dir.exists(file.path(resultsDir)), dir.create(file.path(resultsDir)), FALSE)){
    
    cutLength = attr(regexpr("\\_[^\\_]*$", resultsDir),"match.length")
    
    observedChar <- substr(resultsDir,nchar(resultsDir)-cutLength+2,nchar(resultsDir))
    substr(resultsDir,nchar(resultsDir)-cutLength+2,nchar(resultsDir)) <- as.character(as.integer(observedChar)+1)
    rm(cutLength,observedChar)
  }
  
  return(resultsDir)
}

createEmptyNamedDataFrame = function(colNamesList){
  # creates an empty data frame that is named
  df <- data.frame(matrix(ncol = length(colNamesList), nrow = 0)) # only the accepted moves
  colnames(df) <- colNamesList
  return(df)
}
