#NSGA2 Implementation test on zdt2 and zdt3
libNames <- c( "ggplot2","mco","nsga2R","reshape2","plyr")
lapply(libNames, library, character.only = TRUE) # load the required packages
rm(libNames)

BenchMarkSwitch = FALSE

# ZDT2 Benchmark (2 objectives)
if(BenchMarkSwitch){
varNo = 6

resultsZDT2 <- nsga2R(fn = zdt2, varNo, objDim = 2, lowerBounds = rep(0, varNo), upperBounds = rep(1, varNo),
                  popSize = 100, tourSize = 2, generations = 50, cprob = 0.7, XoverDistIdx = 5,
                  mprob = 0.2, MuDistIdx = 10)

plot(resultsZDT2$objectives)

plot(t(apply(resultsZDT2$parameters, 1, zdt2)))

# ZDT3 Benchmark (2 objectives)
varNo = 6

resultsZDT3 <- nsga2R(fn = zdt3, varNo, objDim = 2, lowerBounds = rep(0, varNo), upperBounds = rep(1, varNo),
                  popSize = 100, tourSize = 2, generations = 50, cprob = 0.7, XoverDistIdx = 5,
                  mprob = 0.2, MuDistIdx = 10)

plot(resultsZDT3$objectives)
}


plot2Fitness <- function(F1,F2){
  df1 = cbind(F1,0)
  df1 = rbind(df1, cbind(F2,1))
  df1 = data.frame(df1)
  colnames(df1) <- c("f1","f2","t")
  
  plotCompare <- ggplot()+ 
    geom_point(data = df1, aes(x = f1, y = f2, color = factor(t)))+
    labs(
      x = "f1",
      y = "f2",
      color = "Population t"
    )
  
  return(plotCompare)
}

plot2Pops <- function(P1,P2,fn){
  # P1 and P2 are the 2 populations and 
  # fn is the function to be applied
  F1 = t(apply(P1, 1, fn))
  F2 = t(apply(P2, 1, fn))
  
  df1 = cbind(F1,0)
  df1 = rbind(df1, cbind(F2,1))
  df1 = data.frame(df1)
  colnames(df1) <- c("f1","f2","t")
  
  plotCompare <- ggplot()+ 
    geom_point(data = df1, aes(x = f1, y = f2, color = factor(t)))+
    labs(
      x = "f1",
      y = "f2",
      color = "Population t"
    )
  
  return(plotCompare)
}

plotP_t <- function(P_t){
  # A population with added with objective functions and rank 
  plot <- ggplot()+ 
    geom_point(data = P_t, aes(x = f1, y = f2, color = factor(rank)))+
    labs(
      x = "f1",
      y = "f2",
      color = "Rank"
    )
  
  return(plot)
}

# Input parameters -----
varNo = 6
objDim = 2
fn = zdt2 # NB this should be a minimisation function
lowerBound = 0
upperBound = 1
lowerBounds = rep(0, varNo)
upperBounds = rep(1, varNo)
popSize = 100
tourSize = 2 
generations = 20 
cprob = 0.7 
XoverDistIdx = 5
mprob = 0.2 
MuDistIdx = 10

# Initialise population P_0 with fitness -----
P_0 <- matrix(ncol = varNo, nrow = popSize)
F_0 <- matrix(ncol = objDim, nrow = popSize)
t <- 0

for (i in 1:nrow(P_0)) {
  for (j in 1:ncol(P_0)) {
    P_0[i,j] <- runif(1, lowerBound, upperBound)
    F_0[i,] <- fn(P_0[i,]) 
  }
}

# Set the population in the appropriate format -----
# NB this is set up for a MINIMISATION problem
set.seed(1234)
population <- P_0
fitness <- t(apply(P_0, 1, fn))
population <- cbind(population, fitness)
ranking <- fastNonDominatedSorting(population[,(varNo+1):(varNo+objDim)])
rnkIndex <- integer(popSize)
i <- 1
while (i <= length(ranking)) {
  rnkIndex[ranking[[i]]] <- i
  i <- i + 1
} 
population <- cbind(population,rnkIndex);
objRange <- apply(population[,(varNo+1):(varNo+objDim)], 2, max) -
  apply(population[,(varNo+1):(varNo+objDim)], 2, min);
cd <- crowdingDist4frnt(population,ranking,objRange)
population <- cbind(population,apply(cd,1,sum))

# GA Generations while loop -----

# GA Generations counter
t = 0
while (t <= generations) {
  
  # Tournament selection -----
  # creates the mating pool
  matingPool <- tournamentSelection(population,popSize,tourSize)
  #matingPool
  
  # Crossover -----
  matingPool <- matingPool[,1:varNo] # get only the decision variables
  cprob <- 0.7
  XoverDistIdx <- 5
  childAfterX <- boundedSBXover(matingPool,lowerBounds,upperBounds,cprob,XoverDistIdx)
  #childAfterX
  
  # Mutation -----
  # set.seed(1234)
  matingPool <- childAfterX
  childAfterM <- boundedPolyMutation(matingPool,lowerBounds,upperBounds,mprob,MuDistIdx)
  #childAfterM
  
  # Combine parent and offspring populations
  P_t = population[,1:varNo] # previous population
  Q_t = childAfterM # offspring of previous population
  R_t = rbind(P_t, Q_t)
  fitness = t(apply(R_t, 1, fn))
  
  population = cbind(R_t, fitness) # population rewritten
  ranking = fastNonDominatedSorting(population[,(varNo+1):(varNo+objDim)]) # determine rank
  rnkIndex <- integer(popSize)
  i <- 1
  while (i <= length(ranking)) {
    rnkIndex[ranking[[i]]] <- i
    i <- i + 1
  } 
  population <- cbind(population,rnkIndex); # add rank
  
  objRange <- apply(population[,(varNo+1):(varNo+objDim)], 2, max) -
    apply(population[,(varNo+1):(varNo+objDim)], 2, min);
  cd <- crowdingDist4frnt(population,ranking,objRange)
  population <- cbind(population,apply(cd,1,sum)) # add crowding distance
  
  
  population = data.frame(population)
  colnames(population) <- c(1:varNo,"f1","f2","rank","cd")
  population = population[with(population, order(rank)),] # sort the population
  
  
  P_t_new = data.frame(matrix(ncol = length(c(1:varNo,"f1","f2","rank","cd")), nrow = 0))
  colnames(P_t_new) <- c(1:varNo,"f1","f2","rank","cd")
  
  i = 1
  while (nrow(P_t_new) + sum(population$rank == i) <= popSize) { # fill up P_t_new
    P_t_new = rbind(P_t_new, population[population$rank == i,])
    i = i + 1
  }
  
  # Fill up the remainder of R_t
  remaining_R_t = population[population$rank > max(P_t_new$rank),]
  
  P_t_new = rbind(P_t_new,
                  remaining_R_t[with(remaining_R_t,order(rank,-cd)),][1:(popSize - nrow(P_t_new)),]) # sort remaining
  
  # end first generation
  t = t + 1
  population = P_t_new
} # end overall generation while loop


# Plot the different fronts
plotP_t(P_t_new)
plotP_t(P_t_new[P_t_new$rank==1,])

# Plot different populations -----
# Evaluate fitness function 
if (BenchMarkSwitch) {
  
  F_t <- t(apply(childAfterM, 1, fn))
  plot2Fitness(F_0,F_t)
  plot2Pops(childAfterX,childAfterM,zdt2MIN)
}

