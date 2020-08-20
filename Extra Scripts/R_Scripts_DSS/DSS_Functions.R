# Skripsie 2018 functions focusing on the graph analytics and generation

# Function for testing if packages are installed, and if not, install and load

testInstallAndLoadPackages <- function(list_of_packages){
  # list_of_packages is the list of all the packages you want to include
  new.packages <- list_of_packages[!(list_of_packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)>0){install.packages(new.packages)} # installs the packages if not installed
  
  lapply(list_of_packages, library, character.only = TRUE) # load the required packages
}


# Formatting functions --------

formatDistMatrix <- function(S){
  # Set all the impossible connections between nodes to a large number.
  # Because the algorithm is looking for a minimum, very large distances will never be selected
S <- S[1:nrow(S) , 2:ncol(S)]
S <- as.matrix(S)
colnames(S) <- rownames(S) <- 1:nrow(S)
return(S)
}

formatDemandMatrix <- function(demandMatrix){
  # Format demandMatrix in correct matrix format
  demandMatrix <- demandMatrix[1:nrow(demandMatrix) , 2:ncol(demandMatrix)]
  demandMatrix <- as.matrix(demandMatrix)
  colnames(demandMatrix) <- rownames(demandMatrix) <- 1:nrow(demandMatrix)
  return(demandMatrix)
}

convertRouteListToCharString = function(routeList){
  # A function to display the routes in the standard format seperated with "\n"
  formattedRoutes <- as.vector(NULL)
  stringRoutes <- as.vector(NULL)
  
  for(i in 1:length(routeList)) {           
    
    stringRoutes <- paste(routeList[[i]],sep = "", collapse = "-")
    formattedRoutes <- paste(formattedRoutes,stringRoutes,"*",sep = "", collapse ="")
  }
  return(formattedRoutes)
}

# Dijkstra's algorithm --------------------
# adapted from https://uqkdhanj.wordpress.com/2015/02/10/dijkstras-shortest-pathway-algorithm/

dijkstra <- function(v,destination,cost){
  
  # v = beginning node
  # destination = destination node
  # cost = distance matrix
  
  # Take note that this function does not take into account various shortest paths of the same length 
  
  n <- nrow(cost)
  #create empty variables to store data
  dest = numeric(n) # wrong in code? overrides dest value
  flag = numeric(n)
  prev = numeric(n)
  
  # for every node in the network
  for(i in 1:n){
    prev[i] = -1
    dest[i] = cost[v,i] #= distance from start node v to every other node i in the network
  }
  
  #initialise counter which keeps track of number of steps through network
  count=2
  
  # until we have reached our destination node n
  while(count <= n){
    min=max(cost)
    
    # loop over each node
    for(w in 1:n){
      #if the new path is less long than the existing smallest one and flag[w] is equal to zero (aka we've not already incuded that node in route)
      if(dest[w] < min && !flag[w]){
        # overwrite the minimum with the new shortest path and update counter
        min=dest[w]
        u=w
      }
    }
    flag[u] = 1 #indicate that we go to this site
    count = count+1
    
    # loop over each node again keeping in mind where we have already been
    for(w in 1:n){
      #if the new route is shorter than the previous route
      if((dest[u]+cost[u,w] < dest[w]) && !flag[w]){
        dest[w]=dest[u]+cost[u,w] #update the distance to destination
        prev[w]=u #keep track of the node visited
      }
    }
  }
  return(prev)
}


savepath <- function(f,v,x){
# function which returns a path
  # f = output from Dijkstra function
  # v = beginning node
  # x = destination node
  
  path=x
  while(f[x] != -1){
    path=c(path,f[x])
    x=f[x]
    savepath(f,v,x)
  }
  path=c(path,v)
  return(path)
}

# Generate all the shortest routes as candidates

generateAllShortestRoutes <- function(S){

  canditateRoutes <- as.list(NULL)
  
routeCount <- 1

  for(i in 1:nrow(S)) {
  
    for(j in 1:nrow(S)) {
    
      if(i != j) {
         prev = dijkstra( i , j , S)
         canditateRoutes[[routeCount]] = rev( savepath(prev, i , j) )
         routeCount<- routeCount + 1
      
      }
    
    }
  
  }
return(canditateRoutes)
}

# Calculate the shortest distance matrix for the candidate routes-----------------------

calculateRouteLengths <- function(distMatrix , routes){
  # this function takes as input a distance matrix and routes through the matrix and 
  # gives as output the distance from the  beginning to end node using the routes
  # provided
  
  shortDistMatrix <- matrix(0,nrow(distMatrix),ncol(distMatrix))
  
  for(i in 1:length(routes)) {
    
    path <- routes[[i]]
    dist <- 0
    
    for (j in 1:(length(path)-1) ) {
      
      dist <- dist + distMatrix[path[j] , path[j+1]]
      
    }
    
    shortDistMatrix[path[1],path[length(path)]] <- shortDistMatrix[path[length(path)],path[1]] <- dist
    
  }
  # verified that the correct distances are calculated by comparing it to the iGraph shortest paths output
  return(shortDistMatrix)
}

calculateRouteLengths2 <- function(distMatrix , routes){
  # this function takes as input a distance matrix and routes through the matrix and 
  # gives as output the distance from the  beginning to end node using the routes
  # provided
  
  routeLengths <- replicate(length(routes),0)
  
  for(i in 1:length(routes)) {
    
    path <- routes[[i]]
    dist <- 0
    
    for (j in 1:(length(path)-1) ) {
      
      dist <- dist + distMatrix[path[j] , path[j+1]]
      
    }
    
    routeLengths[i] <- dist
    
  }
  # verified that the correct distances are calculated by comparing it to the iGraph shortest paths output
  return(routeLengths)
}


# Create a shortened list and remove the routes longer than the specified number ----------
specifyNodesPerRoute <- function(routes , minNodes , maxNodes){

shortenedCandidateRoutes <- routes # a copy of the original candidate routes

  for(i in length(routes):1 ) { 
  
    if(length(routes[[i]]) > maxNodes || length(routes[[i]]) < minNodes ) {
    
      shortenedCandidateRoutes[[i]] <- NULL # delete routes that do not meet specifications
    
    }
  }
return(shortenedCandidateRoutes)
}

# Generate or select Candidate routes ----------
generateCandidateRoutes <- function(routeSet, allowedRoutes){
  
  # Generate a candidate Route Set----------------------
  
  # routeSet is the route set containing all possible shortest routes from node i to j 
  #     subject to certain constraints that should have been predefined
  # allowedRoutes is the number of allowed routes
  m <- length(routeSet) # number of possible candidate routes
  x <- as.list(NULL) # create the decision variables to keep the candidate solutions
  
  k <- sample.int( m , allowedRoutes , replace = FALSE) # positions of the candidate routes
  k <- as.vector(k)
  for(i in 1:allowedRoutes) {
    
    x[[i]] <- routeSet[[k[i]]]
    
  }
  
  return(x)
  
} # end generateCandidateRoutes

generateSolution <- function(canditateRoutes, M, N , i){
  # Generate a feasible solution where all routes are connected
  # candidate routes are the set of all possible routes one can choose from
  # M is the number of routes you want generated
  # N is the number of nodes in the network
  # i are the iterations that should be performed
  
  for(j in 1:i) {
    
    x<- generateCandidateRoutes(canditateRoutes, M)
    
    if(feasibilityConnectedRoutes(x,N)) {
      
      return(x)
      
    }
    
    
    
  } # end for loop
  
  return(NULL) # if a feasible solution is not generated
  
}

generateFeasibleSolution <- function(candidateRoutes, numAllowedRoutes, N, i){
  # This code is used to generate a set of routes that are connected with each other
  # from the possible set of candidate routes presented
  # under the the constraint of the number of allowed routes specified
  # in a network consisting of N nodes
  # with i iterations allowed
  
  for(j in 1:N) {
    
    x <- generateSolution(candidateRoutes, numAllowedRoutes, N, i )
    
    if(is.null(x)) { # tests if the solution was feasible, if not, leverage the
      # constraint by adding one more allowable route
      
      numAllowedRoutes <- numAllowedRoutes + 1
      
    }
    else{
      if(feasibilityConnectedRoutes(x,N)){
        break
      }
    }
  } # end for
  return(x)
}  

generateBusNetworkDistMatrix <- function(distMatrix,routes){
  # Generate Bus Route Network ---------
  # Calculate the allowed distances in the bus network only ========
  
  busNetworkDistMatrix <- matrix(max(distMatrix),nrow(distMatrix),ncol(distMatrix))
  
  for(i in 1:length(routes)) {
    
    for(j in 1:(length(routes[[i]]) - 1) ) {
      
      busNetworkDistMatrix[routes[[i]][j] , routes[[i]][j+1]] <- busNetworkDistMatrix[routes[[i]][j+1] , routes[[i]][j]] <- distMatrix[routes[[i]][j] , routes[[i]][j+1]]
      
    }
    
  }
  
  return(busNetworkDistMatrix)
  
} # end generateBusNetworkDistMatrix

# DSS Simulated Annealing Functions ------

f1_totalRouteLength <- function(S,x){
  # Determine the total length of the current bus routes
  routeLengths <- calculateRouteLengths2(S,x)
  return(sum(routeLengths)) # divide by two to take into account the matrix symmetry
  
}

f2_averageTravelTime <- function(S,demandMatrix,x){
  # Determine the average travel time per customer
  # Generate Bus Route Network ---------
  
  # Generate Bus Network Dist Matrix ========
  busNetworkDistMatrix <- generateBusNetworkDistMatrix(S,x)
  
  # Determine all shortest routes in the Bus network from one node to another-----
  shortestBusRoutes <- generateAllShortestRoutes(busNetworkDistMatrix)
  
  # Calculate the shortest distance matrix for the candidate bus routes-----------------------
  shortBusDistMatrix <- calculateRouteLengths(busNetworkDistMatrix,shortestBusRoutes)
  N <- nrow(S)
  # Generate transfer matrix
  transferMatrix <- generateTransferMatrix(x,shortestBusRoutes,N)
  
  return(sum(demandMatrix*(shortBusDistMatrix + 5*transferMatrix)) / sum(demandMatrix))
  
} # end f2

feasibilityConnectedRoutes <- function(R,N) {
  # from Fan, Lang Mumford, Christine L. 2010
  # Function that tests the feasibility of all the nodes being included and connected
  # N is the number of nodes in network 
  # R is imported candidate route set in list format for a solution
  
  # Using Algorithm 1 from Fan and Mumford 2010 to determine whether all nodes are included --------
  # The while loop was adapted because it does not make sense to have the feasibility as the while
  # loop logic test because infeasible solutions will result in infinite loops
  
  foundNode <- c(rep(0, N)) # records nodes that have been found
  exploredNode <- c(rep(0, N)) # records nodes that have been explored
  iRoutesFound <- as.vector(NULL) # vector to keep track of the routes found containing node i
  
  iNode <- R[[1]][1] # select an arbitrary node i present in at least one route
  feas <- FALSE
  switchTest <- FALSE
  counter <- 1
  
  while(switchTest == FALSE) {
    
    
    exploredNode[iNode] <- foundNode[iNode] <- 1
    
    # Find all routes containing node i
    iRoutesFound <- NULL
    
    for(k in 1:length(R)){ # loop over the number of routes in analysis
      
      for(g in 1:length( R[[k]])) { # set flags in found node to record nodes found in the routes containing i 
        
        if ( R[[k]][g] == iNode) {
          
          iRoutesFound <- append(iRoutesFound,k)
          #break
        } 
      }
    }  
    
    # Set flags in found-node to record all the nodes found in those routes
    
    for(k in 1:length(iRoutesFound)){ # loop over the number of routes in analysis
      
      for(g in 1:length(R[[ iRoutesFound[k]]])) { # set flags in found node to record nodes found in the routes containing i 
        
        foundNode[ R[[ iRoutesFound[k]]][g]] <- 1
        
      }
    }
    
    # Select any node from found-node that is absent from explored-node
    for(i in 1:N){
      
      if(foundNode[i] != exploredNode[i])
        
        iNode <- i
      # break
      
    }
    
    
    if(sum(foundNode) == N){
      
      feas <- TRUE
      switchTest <- TRUE
      
    }
    
    if(counter > N*N & sum(foundNode==exploredNode) == N) {
      
      switchTest <- TRUE
      
    }
    
    counter <- counter + 1
    
  } # end while 
  
  return(feas)
  
} # end feasibilityConnectedRoutes

testFeasibility <- function(x,N){ # this function is used in "make-small-change" too
  # Test feasibility
  # x is the Routes
  # N is the number of nodes in the network 
  
  return(feasibilityConnectedRoutes(x,N)) # insert other feasibility tests in here
} # end testFeasibilty

testFeasibility2 <- function(x,N,minNodes,maxNodes){ # this function is used in "make-small-change" too
  # Test feasibility
  # x is the Routes
  # N is the number of nodes in the network 
  
  for (i in 1:length(x)) { # test for too short route
    if(length(x[[i]]) < minNodes){
      return(FALSE)
    }
  }

  for (i in 1:length(x)) { # test for too long route
    if(length(x[[i]]) > maxNodes){
      return(FALSE)
    }
  }
  
  return(feasibilityConnectedRoutes(x,N)) # insert other feasibility tests in here
} # end testFeasibilty2

addNodeToEndOfRoute <- function(x, S, k) {
  # This function adds a node to the route that avoids cycles and backtracking 
  # x is the route set
  # S is the distMatrix that busses can travel on
  # k is the route number
  # x[[k]][length(x[[k]])] # node at last position
  
  addNodeTest <- S[x[[k]][length(x[[k]])], ] < max(S) # exclude impossible routes
  
  addNodeTest[x[[k]]] <- FALSE # exclude current nodes already in route
  
  if(sum(addNodeTest)!=0){ # case where other nodes can be added
    nodes <- as.vector(which(addNodeTest, arr.ind = TRUE, useNames = FALSE))
    
    if (length(nodes) == 1) {
      x[[k]][length(x[[k]]) + 1] <- nodes
    }
    else{
      x[[k]][length(x[[k]]) + 1] <- sample(nodes, 1) 
    }
  }
  # case where no more nodes can be added that won't result in backtracking or cycles
  # is omitted as no changes can be made further and return x
  
  return(x)
} # end addNodeToEndOfRoute

addNodeToStartOfRoute <- function(x, S, k) {
  # This function adds a node to the route that avoids cycles and backtracking 
  # x is the route set
  # S is the distMatrix that busses can travel on
  # k is the route number
  # x[[k]][length(x[[k]])] # node at last position
  x[[k]] <- rev(x[[k]]) # reverses route so that it is added to the start
  
  addNodeTest <- S[x[[k]][length(x[[k]])], ] < max(S) # exclude impossible routes
  
  addNodeTest[x[[k]]] <- FALSE # exclude current nodes already in route
  
  if(sum(addNodeTest)!=0){ # case where other nodes can be added
    nodes <- as.vector(which(addNodeTest, arr.ind = TRUE, useNames = FALSE))
    
    if (length(nodes) == 1) {
      x[[k]][length(x[[k]]) + 1] <- nodes
    }
    else{
      x[[k]][length(x[[k]]) + 1] <- sample(nodes, 1) 
    }
  }
  # case where no more nodes can be added that won't result in backtracking or cycles
  # is omitted as no changes can be made further and return x
  
  return(x)
}

removefirstNode <- function(x,k){
  # removes a node in the first position
  x[[k]] <- x[[k]][-1]
  return(x)
}

inverseNodes <- function(x,k){
  # inverses the order of the nodes
  x[[k]] <- rev(x[[k]])
}

makeSmallChange <- function(x,N,minNodes,maxNodes){
  # Generate neigboring solution
  # x is the current solution of bus routes
  # N is the number of nodes in the network
  
  x_pert <- x # define new solution that will be the perturbation
  
  # Generate number k to determine which route to perturb
  k <- sample.int(length(x),1,replace = TRUE)
  
  n <- 1 # counter to avoid unending search
  
  repeat{
    
    # Generate random number p in (0,1) to determine the pertubation to execute
    p <- runif(1,0,1)
    
    if (length(x[[k]])>=maxNodes){
      # with max nodes, inverse or remove
      
      if(p<0.5){
        x_pert[[k]] <- x[[k]][-1] # remove first node
        
      } else{
        x_pert[[k]] <- x[[k]][-length(x[[k]])] # remove last node
      }
      
    } else{
      
      if(length(x[[k]])<=minNodes) {
        # with min nodes, add or inverse
        
        if(p<0.5){
        x_pert <- addNodeToEndOfRoute(x,S,k) # add node to end 
        
        } else{
        x_pert <- addNodeToStartOfRoute(x,S,k)  # add node to start of
        }
        
      } else{
        # inbetween case, add nodes or remove nodes
        if(p<0.25){
          x_pert <- addNodeToStartOfRoute(x,S,k)  # add node to start 
          
        } else{
          
          if(p<0.5){
            x_pert <- addNodeToEndOfRoute(x,S,k) # add node to end 
          
          } else {
            if(p<0.75){
              x_pert[[k]] <- x[[k]][-1] # remove first node
              
            } else{
              x_pert[[k]] <- x[[k]][-length(x[[k]])] # remove last node
            }
          }
        }
      }
      
    } 
    
    if(testFeasibility(x_pert,N)) { # test feasibility of pertubation
      return(x_pert)     
      break
    }else{
      n <- n+1
    }
    if(n > 1000){
      return(x)
      break
    }
    
  } # end repeat
  
} # end make small change procedure function

makeSmallChange2 <- function(x,N,S,minNodes,maxNodes){
  # Generate neigboring solution
  # x is the current solution of bus routes
  # N is the number of nodes in the network
  
  x_pert <- x # define new solution that will be the perturbation
  
  # Generate number k to determine which route to perturb
  k <- sample.int(length(x),1,replace = TRUE)
  
  # Generate random number p in (0,1) to determine the pertubation to execute
  p <- runif(1,0,1)
  
  if (length(x[[k]])>=N){
    # with max nodes, inverse or remove
    
    if(p<0.5){
      x_pert[[k]] <- x[[k]][-1] # remove first node
      
    } else{
      x_pert[[k]] <- x[[k]][-length(x[[k]])] # remove last node
    }
    
  } else{
    
    if(length(x[[k]])<=2) {
      # with min nodes, add
      
      if(p<0.5){
        x_pert <- addNodeToEndOfRoute(x,S,k) # add node to end 
        
      } else{
        x_pert <- addNodeToStartOfRoute(x,S,k)  # add node to start of
      }
      
    } else{
      # inbetween case, add nodes or remove nodes
      if(p<0.25){
        x_pert <- addNodeToStartOfRoute(x,S,k)  # add node to start 
        
      } else{
        
        if(p<0.5){
          x_pert <- addNodeToEndOfRoute(x,S,k) # add node to end 
          
        } else {
          if(p<0.75){
            x_pert[[k]] <- x[[k]][-1] # remove first node
            
          } else{
            x_pert[[k]] <- x[[k]][-length(x[[k]])] # remove last node
          }
        }
      }
    }
    
  } 
  
return(x_pert)     
  
} # end make small change procedure function

# Archive Functions -------

addSolutionToArchive <- function(x,archive,f1,f2){
  # add a solution and f1 and f2 values to the archive
  archive[[length(archive)+1]] <- list(f1, f2, x)
  return(archive)
} # end addSolutionToArchive

createArchiveDF <- function(archive){
  # Creates a archive data frame that so that f1 and f2 can be accessed
  archiveDF <- as.data.frame(NULL)
  
  for (k in 1:length(archive)) {
    archiveDF[k,1] <- k # route set number
    archiveDF[k,2] <- archive[[k]][[1]] # formatting for f1
    archiveDF[k,3] <- archive[[k]][[2]] # formatting for f2
  }
  
  f1min <- min(archiveDF[,2]) # the min value in each objective
  f2min <- min(archiveDF[,3])
  f1range <- max(archiveDF[,2]) - min(archiveDF[,2]) # the range in which to normalise
  f2range <- max(archiveDF[,3]) - min(archiveDF[,3])
  
  for (k in 1:length(archive)) {
    archiveDF[k,4] <- (archiveDF[k,2]-f1min)/f1range # normalising each value
    archiveDF[k,5] <- (archiveDF[k,3]-f2min)/f2range
    
  }
  
  names(archiveDF) <- c("SetNr","f1","f2","f1norm","f2norm")
  
  return(archiveDF)
} # end createArchiveDF

createArchiveDF4SA <- function(archive){
  # Creates a archive data frame that so that f1 and f2 can be accessed
  archiveDF <- as.data.frame(NULL)
  
  for (k in 1:length(archive)) {
    archiveDF[k,1] <- k # route set number
    archiveDF[k,2] <- archive[[k]][[1]] # formatting for f1
    archiveDF[k,3] <- archive[[k]][[2]] # formatting for f2
  }
  names(archiveDF) <- c("SetNr","f1","f2")
  return(archiveDF)
} # end createArchiveDF4SA

removeSolutionFromArchive <- function(archive,k){
  # To remove a solution from the archive -> just use one line of code
  archive[[k]] <- NULL
  return(archive)
}

countMaxMaxDominate <- function(archiveDF,f1,f2) {
  # Count the number of solutions that dominate solution x 
  # when both functions need to be maximised
  # NB: the archiveDF should be a dataframe so that it can be easily accessed
  
  if (is.null(nrow(archiveDF[archiveDF[,2] > f1 & archiveDF[,3] > f2 , ]) ) ){
    return(0)
  }
  else {
    return(nrow(archiveDF[archiveDF[,2] > f1 & archiveDF[,3] > f2 , ]))
  }
  
} # end countMaxMaxDominate

countMinMinDominate <- function(archiveDF,f1,f2) {
  # Count the number of solutions that dominate solution x 
  # when both functions need to be minimised
  # NB: the archiveDF should be a dataframe so that it can be easily accessed
  
  if (is.null(nrow(archiveDF[archiveDF[,2] < f1 & archiveDF[,3] < f2 , ]) ) ){
    return(0)
  }
  else {
    return(nrow(archiveDF[archiveDF[,2] < f1 & archiveDF[,3] < f2 , ]))
  }
  
} # end countMinMinDominate

testMinMinNonDominated <- function(archiveDF,f1,f2) {
  # Test whether the point f1 and f2 is a non-dominated solution in a min min problem
  return(sum(archiveDF[,2] < f1 & archiveDF[,3] < f2) == 0)
} # end testMinMinNonDominated

removeMinMinDominatedSolutions <- function (archive, f1, f2){
  # This function removes the points that are dominated by solution x 
  # from the archive if both functions need to be minimised 
  
  archiveDF <- createArchiveDF(archive) # create an archive in the correct format
  
  for (k in rev(archiveDF[ (archiveDF[,2] > f1 & archiveDF[,3] > f2) , 1])) {
    # need to work from bottom up so that indices aren't confused
        archive[[k]] <- NULL
  }
  
  return(archive) # remove
  
} # removeMinMinDominatedSolutions

removeAllDominatedSolutionsMinMin <- function(archive){
  
  archiveDF <- createArchiveDF(archive) # create an archive in the correct format
  
  # Test for non-dominated solutions
  nonDom <- as.vector(NULL)
  
  for(i in 1:nrow(archiveDF)){
    nonDom <- rbind(nonDom ,testMinMinNonDominated(archiveDF,archiveDF[i,2],archiveDF[i,3]))
  }

  archiveDF <- archiveDF[nonDom,]
  
  for(i in nrow(archiveDF):1) {
    
    for(j in nrow(archiveDF):1) {
      
      if(i != j){
        
        if(archiveDF[i,2] >= archiveDF[j,2] & archiveDF[i,3] >= archiveDF[j,3]) {
          
          archiveDF <- archiveDF[-i,]
          break
        }
        
      }
      
    }
    
  }  # end for
  
  # Remove the dominated solutions
  i <- 1:length(archive)
  for (k in rev(i[-archiveDF[, 1]])) { 
    # need to work from bottom up so that indices aren't confused
    archive[[k]] <- NULL
  }
  return( archive )
  
} # end removeAllDominatedSolutionsMinMin

energyFunction <- function(archiveDF,x_pert,x,S,demandMatrix){
  
  x_f1 <- f1_totalRouteLength(S,x)
  x_f2 <- f2_averageTravelTime(S,demandMatrix,x)
  x_pert_f1 <- f1_totalRouteLength(S,x_pert)
  x_pert_f2 <- f2_averageTravelTime(S,demandMatrix,x_pert)
  
  return( ( (countMinMinDominate(archiveDF,x_pert_f1,x_pert_f2)) - (countMinMinDominate(archiveDF,x_f1,x_f2)) )/ nrow(archiveDF) )
  
} # end energy function

energyFunction2 <- function(archiveDF,x_pert,x,S,demandMatrix){
  # for use with initial temperature 
  x_f1 <- f1_totalRouteLength(S,x)
  x_f2 <- f2_averageTravelTime(S,demandMatrix,x)
  x_pert_f1 <- f1_totalRouteLength(S,x_pert)
  x_pert_f2 <- f2_averageTravelTime(S,demandMatrix,x_pert)
  
  return( ( (countMinMinDominate(archiveDF,x_pert_f1,x_pert_f2)) - (countMinMinDominate(archiveDF,x_f1,x_f2)) ) )
  
} # end energy function2

probToAcceptNeighbor <- function(archiveDF,x_pert,x,Temperature,S,demandMatrix){
  # Generates a probability of measuring acceptance
  return( min(1,exp(-energyFunction(archiveDF,x_pert,x,S,demandMatrix)/Temperature)) )
  
} # end probToAcceptNeighbor function

metropolisRule <- function (f_x, f_x_pert, Temp){
  
  return ( exp(-(f_x - f_x_pert) / Temp) )
  
}

coolingSchedule <- function (c) {
  # Cooling Schedule  
  # using a geometric series to determine the cooling schedule
  return((0.9^(c-1))*9000)
}

# Simulated Annealing ---------
SimulatedAnnealing <- function(x,N,minNodes,maxNodes,S,demandMatrix,archive,Lc,Amin,Cmax,Temp){
  
  # Preliminary inputs
  # Let x be the starting solution 
  # Let archive be the associated archive
  # Let Lc be maximum allowable number length of iterations per epoch c
  # Let Amin be minimum number of accepted moves per epoch
  # Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
  # Let Temp be the starting temperature and a geometric cooling schedule is used on it
  
  # initiate algorithm
  c <- 1 # Initialise the cooling schedule epoch
  t <- 1 # Initialise the number of iterations 
  eps <- 0 # Initialise the number of epochs without an accepted solution
  archiveDF <- createArchiveDF4SA(archive) # create an archive in the correct format
  
  while (eps <= Cmax) {
    A <- 0
    t <- 0
    while (t <= Lc & A < Amin) {
      
      # generate neighboring solutions for solution x
      x_pert <- makeSmallChange(x,N,minNodes,maxNodes) # could return nulls - just check for that
      
      # Generate random number r
      r <- runif(1,0,1)
      
      # Test solution acceptance and add to archive if accepted and non-dominated
      if(r < probToAcceptNeighbor(archiveDF,x_pert,x,Temp,S,demandMatrix)) {
        x <- x_pert
        f1 <- f1_totalRouteLength(S,x)
        f2 <- f2_averageTravelTime(S,demandMatrix,x)
        
        if(countMinMinDominate(archiveDF, f1, f2) == 0){ # means the solution is undominated
          
          archive <- addSolutionToArchive(x,archive,f1,f2)
          archive <- removeMinMinDominatedSolutions(archive,f1,f2)
          archiveDF <- createArchiveDF4SA(archive) # update archive data frame
          
          A  <- A + 1
        }
        
      }
      
      t <- t + 1
      
    } # end inner while
    
    c <- c + 1  # Increase Epoch counter
    Temp <- 0.95*Temp # update cooling schedule
    
    if(A == 0){
      
      eps <- eps + 1 # update number of epochs without an accepted solution
      
    }
    
  } # end outer while
  
  return(archive)
  
}


SimulatedAnnealing2 <- function(x,N,minNodes,maxNodes,S,demandMatrix,archive,Lc,Amin,Cmax,Temp){
  
  # Preliminary inputs
  # Let x be the starting solution 
  # Let archive be the associated archive
  # Let Lc be maximum allowable number length of iterations per epoch c
  # Let Amin be minimum number of accepted moves per epoch
  # Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
  # Let Temp be the starting temperature and a geometric cooling schedule is used on it
  
  # initiate algorithm
  c <- 1 # Initialise the cooling schedule epoch
  t <- 1 # Initialise the number of iterations 
  eps <- 0 # Initialise the number of epochs without an accepted solution
  archiveDF <- createArchiveDF4SA(archive) # create an archive in the correct format
  
  while (eps <= Cmax) {
    A <- 0
    t <- 0
    while (t <= Lc & A < Amin) {
      
      # generate neighboring solutions for solution x
      
      x_pert <- makeSmallChange2(x,N,S,minNodes,maxNodes)
      if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
        
        for (i in 1:1000) {
          x_pert<-makeSmallChange2(x_pert,N,S,minNodes,maxNodes)
          if(testFeasibility2(x_pert,N,minNodes,maxNodes)){
            break
          }
        }
        
      } 
      if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
        x_pert <- makeSmallChange(x,N,minNodes,maxNodes)
      }
      
      # Generate random number r
      r <- runif(1,0,1)
      
      # Test solution acceptance and add to archive if accepted and non-dominated
      if(r < probToAcceptNeighbor(archiveDF,x_pert,x,Temp,S,demandMatrix)) {
        x <- x_pert
        f1 <- f1_totalRouteLength(S,x)
        f2 <- f2_averageTravelTime(S,demandMatrix,x)
        
        if(countMinMinDominate(archiveDF, f1, f2) == 0){ # means the solution is undominated
          
          archive <- addSolutionToArchive(x,archive,f1,f2)
          archive <- removeMinMinDominatedSolutions(archive,f1,f2)
          archiveDF <- createArchiveDF4SA(archive) # update archive data frame
          
          A  <- A + 1
        }
        
      }
      
      t <- t + 1
      
    } # end inner while
    
    c <- c + 1  # Increase Epoch counter
    Temp <- 0.95*Temp # update cooling schedule
    
    if(A == 0){
      
      eps <- eps + 1 # update number of epochs without an accepted solution
      
    }
    
  } # end outer while
  
  return(archive)
  
}


SimulatedAnnealing3 <- function(x,N,minNodes,maxNodes,S,demandMatrix,archive,Lc,Amin,Cmax,Temp,
                                Main_UTRP_df,Main_UTRP_all_df){
  
  # Preliminary inputs
  # Let x be the starting solution 
  # Let archive be the associated archive
  # Let Lc be maximum allowable number length of iterations per epoch c
  # Let Amin be minimum number of accepted moves per epoch
  # Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
  # Let Temp be the starting temperature and a geometric cooling schedule is used on it
  
  # initiate algorithm
  c <- 1 # Initialise the cooling schedule epoch
  t <- 1 # Initialise the number of iterations 
  eps <- 0 # Initialise the number of epochs without an accepted solution
  archiveDF <- createArchiveDF4SA(archive) # create an archive in the correct format
  
  # Initiate Main_UTRP
  Main_UTRP_df <- add_Main_entry(Main_UTRP_df,0,f1_totalRouteLength(S,x),f2_averageTravelTime(S,demandMatrix,x),
                                 Temp,c,0,0,0)
  Main_UTRP_all_df <- add_Main_entry(Main_UTRP_all_df,0,f1_totalRouteLength(S,x),f2_averageTravelTime(S,demandMatrix,x),
                                 Temp,c,0,0,0)
  
  while (eps <= Cmax) {
    A <- 0
    t <- 0
    while (t <= Lc & A < Amin) {
      
      # generate neighboring solutions for solution x
      
      x_pert <- makeSmallChange2(x,N,S,minNodes,maxNodes)
      if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
        
        for (i in 1:1000) {
          x_pert<-makeSmallChange2(x_pert,N,S,minNodes,maxNodes)
          if(testFeasibility2(x_pert,N,minNodes,maxNodes)){
            break
          }
        }
        
      } 
      if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
        x_pert <- makeSmallChange(x,N,minNodes,maxNodes)
      }
      
      # Generate random number r
      r <- runif(1,0,1)
      
      # Test solution acceptance and add to archive if accepted and non-dominated
      if(r < probToAcceptNeighbor(archiveDF,x_pert,x,Temp,S,demandMatrix)) {
        x <- x_pert
        f1 <- f1_totalRouteLength(S,x)
        f2 <- f2_averageTravelTime(S,demandMatrix,x)
        
        Main_UTRP_all_df <- add_Main_entry(Main_UTRP_all_df,t,f1,f2,Temp,c,t,A,eps)
        
        if(countMinMinDominate(archiveDF, f1, f2) == 0){ # means the solution is undominated
          
          archive <- addSolutionToArchive(x,archive,f1,f2)
          archive <- removeMinMinDominatedSolutions(archive,f1,f2)
          archiveDF <- createArchiveDF4SA(archive) # update archive data frame
          
          Main_UTRP_df <- add_Main_entry(Main_UTRP_df,t,f1,f2,Temp,c,t,A,eps)
          
          A  <- A + 1
          eps <- 0
        }
        
      }
      
      t <- t + 1
      
    } # end inner while
    
    c <- c + 1  # Increase Epoch counter
    Temp <- 0.95*Temp # update cooling schedule
    
    if(A == 0){
      
      eps <- eps + 1 # update number of epochs without an accepted solution
      
    }
    
  } # end outer while
  
  return(list(archive,Main_UTRP_df,Main_UTRP_all_df))
  
}


AND1 <- function (...){  
  Reduce("&", list(...))
} # Logical "and" for multiple logical vectors

OR1 <- function (...){  
  Reduce("|", list(...))
} # Logical "or" for multiple logical vectors


simplifyShortBusRoutes <- function(shortestBusRoutes,N){
  # removes the double entries in the shortest routes list
  simplifiedShortestBusRoutes <- shortestBusRoutes
  
  lookupM <- matrix(1:length(shortestBusRoutes), nrow =N-1, ncol = N)
  
  for (j in N:1) {
    for (i in (N - 1):1) {
      if (i < j) {
        simplifiedShortestBusRoutes[[lookupM[i, j]]] <- NULL
      }
    }
  }
  
  return(simplifiedShortestBusRoutes)
} # end simplifyShortBusRoutes

generateTransferMatrix <- function(x,shortestBusRoutes,N){
  # generate the transfer matrix that counts how much transfers each customer 
  # undergoes per OD pair
  # x is the paths
  # shortestBusRoutes is the shortest routes passengers can take in the network
  # N is is number of nodes in the network
  
  sSBR <- simplifyShortBusRoutes(shortestBusRoutes,N) # simplified shortest bus routes
  
  tMatrix <- matrix(3,N,N) # three transfers is set as the limit and a penalty
  diag(tMatrix)<-0
  
  for (i in 1:length(sSBR)) {
    flag <- 0
    
    for (k in 1:length(x)) {
      # test for 0 transfers
      
      if (all(sSBR[[i]] %in% x[[k]])) {
        tMatrix[sSBR[[i]][1], sSBR[[i]][length(sSBR[[i]])]] <-
         tMatrix[sSBR[[i]][length(sSBR[[i]])], sSBR[[i]][1]] <-
          0
        flag <- 1
        break
      }
    } # end test 0 transfers  
    
    
    if (!flag) {
      for (k in 1:length(x)) {
        # test for 1 transfers
        for (l in setdiff(1:length(x),k)) {
          if (all(OR1(sSBR[[i]] %in% x[[l]] , sSBR[[i]] %in% x[[k]]))) {
            tMatrix[sSBR[[i]][1], sSBR[[i]][length(sSBR[[i]])]] <-
             tMatrix[sSBR[[i]][length(sSBR[[i]])], sSBR[[i]][1]] <-
              1
            flag <- 1
            break
          }
          
        }
      }
    } # end test 1 transfers
    
    if (!flag) {
      for (k in 1:length(x)) {
        # test for 2 transfers
        for (l in setdiff(1:length(x),k)) {
          for (m in setdiff(1:length(x),c(k,l))) {
            
            
            if (all(OR1(sSBR[[i]] %in% x[[k]] , sSBR[[i]] %in% x[[l]] , sSBR[[i]] %in% x[[m]]))) {
              tMatrix[sSBR[[i]][1], sSBR[[i]][length(sSBR[[i]])]] <-
                tMatrix[sSBR[[i]][length(sSBR[[i]])], sSBR[[i]][1]] <-
                2
              flag <- 1
              break
            }
          }
        }
      }
    } # end test 2 transfers
    
  }
  return(tMatrix)
} # generateTransferMatrix

add_Main_entry <- function(Main_UTRP_df, Iteration, f1_TRT, f2_ATT,Temperature,C_epoch_number,
                           L_iteration_per_epoch,A_num_accepted_moves_per_epoch,eps_num_epochs_without_accepting_solution){
  row_entry <- nrow(Main_UTRP_df) + 1
  Main_UTRP_df[row_entry,1] <- Iteration
  Main_UTRP_df[row_entry,2] <- f1_TRT
  Main_UTRP_df[row_entry,3] <- f2_ATT
  Main_UTRP_df[row_entry,4] <- Temperature
  Main_UTRP_df[row_entry,5] <- C_epoch_number
  Main_UTRP_df[row_entry,6] <- L_iteration_per_epoch
  Main_UTRP_df[row_entry,7] <- A_num_accepted_moves_per_epoch
  Main_UTRP_df[row_entry,8] <- eps_num_epochs_without_accepting_solution

  return(Main_UTRP_df)
}

add_Main_entry2 <- function(Main_UTRP_df, Iteration, f1_TRT, f2_ATT,Temperature,C_epoch_number,
                           L_iteration_per_epoch,A_num_accepted_moves_per_epoch,eps_num_epochs_without_accepting_solution,
                           route){
  row_entry <- nrow(Main_UTRP_df) + 1
  Main_UTRP_df[row_entry,1] <- Iteration
  Main_UTRP_df[row_entry,2] <- f1_TRT
  Main_UTRP_df[row_entry,3] <- f2_ATT
  Main_UTRP_df[row_entry,4] <- Temperature
  Main_UTRP_df[row_entry,5] <- C_epoch_number
  Main_UTRP_df[row_entry,6] <- L_iteration_per_epoch
  Main_UTRP_df[row_entry,7] <- A_num_accepted_moves_per_epoch
  Main_UTRP_df[row_entry,8] <- eps_num_epochs_without_accepting_solution
  Main_UTRP_df[row_entry,9] <- route
  
  return(Main_UTRP_df)
}

createNamedDataFrame = function(...){
  string_vector_names = c(...)
  
  df_string_vector_names = data.frame(matrix(ncol = length(string_vector_names), nrow = 0))
  colnames(df_string_vector_names) = string_vector_names
  return(df_string_vector_names)
}

