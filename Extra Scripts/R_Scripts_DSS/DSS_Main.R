# DSS Main

# Load Libraries -------
list.of.packages <- c( "rstudioapi","ggplot2", "igraph","png","plotly","PythonInR","ecr","tidyverse") # list of packages to use
new_packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])] # IDs new packages
if(length(new_packages)>0){install.packages(new_packages)} #installs the new packages if required
lapply(list.of.packages, library, character.only = TRUE) # load the required packages
rm(list.of.packages,new_packages) # removes the list created

# Set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # gets and sets the directory of the current script

# Load other functions and scripts -------
source("./DSS_Functions.R")
source("./DSS_Admin_Functions.R")
source("./DSS_Visualisation_Functions.R")

# 0.) Define user specified parameters ----------

problemName <- "Mandl_Data" # NB copy this from the folders as it is used in file names

# Create the folder for the results to be stored
resultsDir = paste("./Results/Results_",substr(Sys.time(),1,10),"_",problemName,"_","Routes_0", sep = "")

resultsDir = createResultsDirectory(resultsDir) # creates a new results directory 

# Enter the number of allowed routes 
numAllowedRoutes <- 6 # (aim for > [numNodes N ]/[maxNodes in route])
minNodes <- 3 # minimum nodes in a route
maxNodes <- 10 # maximum nodes in a route

# Set if archives, workspaces should be saved or loaded
loadArchive <- FALSE
loadWorkspace <- FALSE
saveArchive <- TRUE
saveWorkspace <- TRUE
loadSpecificArchive <- FALSE
loadSpecificWorkspace <- FALSE
initialiseTemperature <- TRUE
optimiseFurther <- TRUE

# 1.) Load the appropriate files and data for the network ------------
# Create and format a distance matrix S
S <- read.csv(paste("./Input_Data/",problemName,"/Distance_Matrix.csv", sep=""))
S <- formatDistMatrix(S)

# Create and format the demand matrix
demandMatrix <- read.csv(paste("./Input_Data/",problemName,"/OD_Demand_Matrix.csv", sep=""))
demandMatrix <- formatDemandMatrix(demandMatrix)

# Collect the correct co-ordinates of the graph
# coords <- layout.auto(g) # to generate coordinates for graph automatically
# write.csv(coords,"MandlSwissNetworkCoords.csv") # use this to store the coords
coords <- read.csv(file = paste("./Input_Data/",problemName,"/Node_Coords.csv", sep=""))
coords <- as.matrix(coords)

# Create the main data frames to keep track
Main_names = c("Iteration", "f1_TRT", "f2_ATT","Temperature","C_epoch_number",
                "L_iteration_per_epoch","A_num_accepted_moves_per_epoch","eps_num_epochs_without_accepting_solution",
               "Route")
Main_UTRP = createEmptyNamedDataFrame(Main_names) # only the accepted moves

Main_UTRP_all_attempts = createEmptyNamedDataFrame(Main_names) # for keeping track of all attempts



# 3.) Determine all the shortest routes ---------
shortestRoutes <- generateAllShortestRoutes(S)

# Calculate the shortest distance matrix for the candidate routes
shortDistMatrix <- calculateRouteLengths(S,shortestRoutes)

# Create a shortened list and remove the routes longer than the specified number
N <- nrow(S) # define the number of nodes in the system

shortenedCandidateRoutes <- specifyNodesPerRoute(shortestRoutes,minNodes,maxNodes)

# 4.) Generate initial feasible solution ----------
x <- generateFeasibleSolution(shortenedCandidateRoutes,numAllowedRoutes,nrow(S),10000) # first initial solution

# 5.) Initialise the archive ----------
archive[[1]] <- list(f1_totalRouteLength(S,x), f2_averageTravelTime(S,demandMatrix,x), x)

# 6.) Simulated Annealing ----------
# Initial temp: =======

if (initialiseTemperature){

initArchive <- as.list(NULL)
energyVect <- as.vector(NULL)
MNum <- 1000
x1 <- generateFeasibleSolution(shortenedCandidateRoutes,numAllowedRoutes,nrow(S),10000)
for (i in 1:MNum){
  
  x_pert <- makeSmallChange2(x1,N,S,minNodes,maxNodes)
  if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
    
    for (j in 1:1000) {
      x_pert<-makeSmallChange2(x_pert,N,S,minNodes,maxNodes)
      if(testFeasibility2(x_pert,N,minNodes,maxNodes)){
        break
      }
    }
    
  } 
  if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
    x_pert <- makeSmallChange(x1,N,minNodes,maxNodes)
  }
  x1<-x_pert 
  initArchive[[i]] <- list(f1_totalRouteLength(S,x1), f2_averageTravelTime(S,demandMatrix,x1), x1)
}
initArcDF <- createArchiveDF(initArchive)
for(i in 1:(MNum-1)){
  energyVect[i] <- energyFunction2(initArcDF,initArchive[[i+1]][[3]],initArchive[[i]][[3]],S,demandMatrix)
  
}
P0 <- 0.999
T0 <- -(sum(abs(energyVect/MNum))/length(energyVect))/log(P0)

Num <- 1000
PNum <- 0.001
TNum <- -(sum(abs(energyVect))/length(energyVect))/log(PNum)

beta <- exp((log(TNum) - log(T0))/Num)
}else{
  T0 <- 500
}
# SA Start

Lc <- 2 # maximum allowable number length of iterations per epoch c
Amin <- 3 # minimum number of accepted moves per epoch
Cmax <- 3 # maximum number of epochs which may pass without the acceptance of any new solution
Temp <- T0  # starting temperature and a geometric cooling schedule is used on it

timeVect <- as.data.frame(NULL) # a data frame to store the time taken 
archiveList <- as.list(NULL) # a list where all the different attainment fronts can be stored
archiveTemp <- as.list(NULL) # a temporary list to store attainment fronts in

multiStarts <- 1

for (i in 1:multiStarts) { # multi-start Simulated Annealing with testing times
  t1 <- Sys.time()
  x <- generateFeasibleSolution(shortenedCandidateRoutes,numAllowedRoutes,nrow(S),10000) # first initial solution
  
  archiveTemp[[1]] <- list(f1_totalRouteLength(S,x), f2_averageTravelTime(S,demandMatrix,x), x)

  t2 <- Sys.time()
  #archiveTemp <- SimulatedAnnealing(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp)
  BIG_LiST <- SimulatedAnnealing3(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp,
                                     Main_UTRP,Main_UTRP_all_attempts)
  archiveTemp <- BIG_LiST[[1]]
  Main_UTRP <- BIG_LiST[[2]]
  Main_UTRP_all_attempts <- BIG_LiST[[3]]
  
  archiveList[[i]] <- archiveTemp
  
  t3 <- Sys.time()
  timeVect[i,1] <- t2 - t1
  timeVect[i,2] <- t3 - t2
  
  archiveTemp <- as.list(NULL)
}
rm(archiveTemp)

# 6.1) Visualise the different attainment fronts ============

pPlot <- ggplot()
colourNames <- c("red","green","blueviolet","blue","darkgreen","turquoise",
                 "pink","orange","brown","maroon","purple","magenta","lightgreen",
                 "gold","black")

if(length(colourNames) < multiStarts){
  colourNames <- colors()[sample.int(length(colors()),multiStarts)]
}

for (i in 1:multiStarts) { # generate a plot of all the solutions in each archive
  
  pPlot <- pPlot + geom_point(data = createArchiveDF(archiveList[[i]]), aes(x = f1, y = f2), color = colourNames[i])
  
}

pPlot

# 6.2) Combine all attainment fronts =============

counter <- 1
archiveComb <- as.list(NULL)
for(i in 1:multiStarts){
  for (j in 1:length(archiveList[[i]])) {
    archiveComb[[counter]] <- archiveList[[i]][[j]]
    counter <- counter + 1
  }
  
}

# 6.3) Create an undominated attainment front and visualise ============

archivePareto <- removeAllDominatedSolutionsMinMin(archiveComb)
archive <- archivePareto

paretoPlot <- ggplot()+ 
  geom_point(data = createArchiveDF(archivePareto), aes(x = f1norm, y = f2norm), color = "red") +
  xlab("f1")+
  ylab("f2")

paretoPlot # plot of normalised pareto front

paretoPlot2 <- ggplot()+ 
  geom_point(data = createArchiveDF(archivePareto), aes(x = f1, y = f2), color = "red") +
  xlab("f1")+
  ylab("f2")

paretoPlot2 # plot of actual values pareto front

if(saveArchive){
  save(list = "archive", file = paste(resultsDir,"/archive",numAllowedRoutes,"Routes",substr(Sys.time(),1,10),"_Pre-Multistart.Rdata", sep = ""))
}

if(saveWorkspace){
  save(list = ls(all.names = TRUE), file = paste(resultsDir,"/workspace",numAllowedRoutes,"Routes",substr(Sys.time(),1,10),"_Pre-Multistart.Rdata", sep = ""), envir = .GlobalEnv)
  save(list = ls(all.names = TRUE), file = ".Rdata", envir = .GlobalEnv) 
}


# 7.) SA on Existing Pareto Front --------
#Lc <- 700 # maximum allowable number length of iterations per epoch c
#Amin <- 3 # minimum number of accepted moves per epoch
#Cmax <- 4 # maximum number of epochs which may pass without the acceptance of any new solution
#Temp <- 15000  # starting temperature and a geometric cooling schedule is used on it

if(optimiseFurther){

timeVect2 <- as.data.frame(NULL)
archiveList <- as.list(NULL)
archiveTemp <- as.list(NULL)
multiStarts <- length(archivePareto)

for (i in 1:multiStarts) { # multi-start Simulated Annealing with testing times
  t1 <- Sys.time()
  x <- archivePareto[[i]][[3]]
    
  archiveTemp[[1]] <- list(f1_totalRouteLength(S,x), f2_averageTravelTime(S,demandMatrix,x), x)
  
  t2 <- Sys.time()
  #archiveTemp <- SimulatedAnnealing(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp)
  archiveTemp <- SimulatedAnnealing(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp)
  
  archiveList[[i]] <- archiveTemp
  
  t3 <- Sys.time()
  timeVect2[i,1] <- t2 - t1
  timeVect2[i,2] <- t3 - t2
  
  archiveTemp <- as.list(NULL)
  
  if(saveArchive){ # save after each iteration
    save(list = "archiveList", file = paste(resultsDir,"/archive",numAllowedRoutes,"Routes",substr(Sys.time(),1,10),"a",i,".Rdata", sep = ""))
  }
}

pPlot <- ggplot()
colourNames <- c("red","green","blueviolet","blue","darkgreen","turquoise",
                 "pink","orange","brown","maroon","purple","magenta","lightgreen",
                 "gold","black")

for (i in 1:multiStarts) { # generate a plot of all the solutions in each archive
  
  pPlot <- pPlot + geom_point(data = createArchiveDF(archiveList[[i]]), aes(x = f1, y = f2), color = colourNames[i])
  
}

pPlot

# Combine all archives
counter <- 1
archiveComb2 <- as.list(NULL)
for(i in 1:multiStarts){
  for (j in 1:length(archiveList[[i]])) {
    archiveComb2[[counter]] <- archiveList[[i]][[j]]
    counter <- counter + 1
  }
  
}

# Create an undominated attainment front of all the attainment fronts
archiveParetoComb2 <- removeAllDominatedSolutionsMinMin(archiveComb2)
#rm(archiveComb)
archive2 <- archiveParetoComb2

# Plot the undominated attainment front (normalised and not)
paretoPlot2Norm <- ggplot()+ 
  geom_point(data = createArchiveDF(archiveParetoComb2), aes(x = f1norm, y = f2norm), color = "red")
paretoPlot2Norm

paretoPlot2 <- ggplot()+ 
  geom_point(data = createArchiveDF(archiveParetoComb2), aes(x = f1, y = f2), color = "red")
paretoPlot2
}



# 8.) Save the workspace and archive ---------

# as .Rdata file in SavedRData folder
if(saveArchive){
  save(list = "archive", file = paste(resultsDir,"/archive",numAllowedRoutes,"Routes",substr(Sys.time(),1,10),"_Final.Rdata", sep = ""))
}

if(saveWorkspace){
  save(list = ls(all.names = TRUE), file = paste(resultsDir,"/workspace",numAllowedRoutes,"Routes",substr(Sys.time(),1,10),"_Final.Rdata", sep = ""), envir = .GlobalEnv)
  save(list = ls(all.names = TRUE), file = ".Rdata", envir = .GlobalEnv) 
}

saveResultsAsCSV(Main_UTRP_all_attempts,"Main_UTRP_all_attempts",0,resultsDir) # function to save the result files in the results folder
saveResultsAsCSV(Main_UTRP,"Main_UTRP",0,resultsDir)


# Compute Hypervolume
computeHV(as.matrix(createArchiveDF(archiveParetoComb2))[,4:5])

testSetA = createArchiveDF(archiveParetoComb2)[,2:3]
testSetA = arrange(testSetA,f1)

referencePoint = c(max(testSetA[,1]),max(testSetA[,2]))



