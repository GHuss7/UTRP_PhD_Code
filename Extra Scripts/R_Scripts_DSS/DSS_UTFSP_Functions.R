# Functions for the UTFSP

# Dijkstra's algorithm --------------------
# adapted from https://uqkdhanj.wordpress.com/2015/02/10/dijkstras-shortest-pathway-algorithm/

dijkstra2 <- function(v,destination,cost){
  
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
  u = 0
  
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


savepath2 <- function(f,v,x){
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


generateAllShortestRoutes2 <- function(S){
  
  canditateRoutes <- as.list(NULL)
  
  routeCount <- 1
  
  for(i in 1:nrow(S)) {
    
    for(j in 1:nrow(S)) {
      
      if(i != j) {
        prev = dijkstra2(i , j , S)
        canditateRoutes[[routeCount]] = rev( savepath2(prev, i , j) )
        routeCount<- routeCount + 1
        
      }
      
    }
    
  }
  return(canditateRoutes)
}

# Calculate the shortest distance matrix for the candidate routes-----------------------

calculateRouteLengths3 <- function(distMatrix , routes){
  # this function takes as input a distance matrix and routes through the matrix and 
  # gives as output the distance from the  beginning to end node using the routes
  # provided
  # it gives a matrix of solutions
  
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

calculateRouteLengths4 <- function(distMatrix , routes){
  # this function takes as input a distance matrix and routes through the matrix and 
  # gives as output the distance from the  beginning to end node using the routes
  # provided 
  # it gives a string of solutions
  
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

calculateShortestBusRouteLengthMatrix <- function(distMatrix , routes){
  # this function takes as input a distance matrix and routes through the matrix and 
  # gives as output the distance from the  beginning to end node using the routes
  # provided
  # it gives a matrix of solutions
  
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


generateBusNetworkDistMatrix2 <- function(distMatrix,routes){
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


generateTransferMatrix2 <- function(x,shortestBusRoutes,N){
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
      
    }
    
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
    }
    
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
    }
    
  }
  return(tMatrix)
} # generateTransferMatrix


f3_averageTravelTime <- function(S,demandMatrix,x){
  # Determine the average travel time per customer
  # x in the shape of a routes list
  # Generate Bus Route Network ---------
  
  # Generate Bus Network Dist Matrix ========
  busNetworkDistMatrix <- generateBusNetworkDistMatrix2(S,x)
  
  # Determine all shortest routes in the Bus network from one node to another-----
  shortestBusRoutes <- generateAllShortestRoutes2(busNetworkDistMatrix)
  
  # Calculate the shortest distance matrix for the candidate bus routes-----------------------
  shortBusDistMatrix <- calculateShortestBusRouteLengthMatrix(busNetworkDistMatrix,shortestBusRoutes)
  N <- nrow(S)
  # Generate transfer matrix
  transferMatrix <- generateTransferMatrix2(x,shortestBusRoutes,N)
  
  return(sum(demandMatrix*(shortBusDistMatrix + 5*transferMatrix)) / sum(demandMatrix))
  
} # end f2

# Spiess (1989) optimal strategies algorithm -----

getLinkWithMinCost = function(S_list,visited_nodes){
  # take the S_list as input with format colnames ("I_i", "I_j", "c_a","f_a")
  # and a set of visited nodes with format character vector 
  # for Spiess (1989) algorithm
  test_visited_nodes = visited_nodes
  
  test_list = cbind(S_list,1:nrow(S_list))
  colnames(test_list)[5] = "index"
  
  test_list = test_list[which(!is.na(match(test_list[,2],test_visited_nodes))),]
  
  min_test = as.matrix(test_list[,c(3,5)])
  
  
  return(S_list[as.numeric(min_test[which.min(min_test[,1]),2]),]) 
  # get the link with j = destination and minimum cost
}

getLinkIndexWithMinCost = function(S_list,visited_nodes){
  # take the S_list as input with format colnames ("I_i", "I_j", "c_a","f_a")
  # and a set of visited nodes with format character vector 
  # for Spiess (1989) algorithm
  test_visited_nodes = visited_nodes
  
  test_list = cbind(S_list,1:nrow(S_list))
  colnames(test_list)[5] = "index"
  
  test_list = test_list[which(!is.na(match(test_list[,2],test_visited_nodes))),]
  
  min_test = as.matrix(test_list[,c(3,5)])
  
  
  return(as.numeric(min_test[which.min(min_test[,1]),2])) 
  # get the link with j = destination and minimum cost
}

f_i_u_i_test = function(fi, ui, alpha_val){
  if(fi == 0 && ui == Inf){
    return(alpha_val)
  }else{
    return(fi*ui)
  }
}

f3_avg_expected_travel_time = function(Distance_mx, Demand_mx, R_routes, F_frequencies){
  # Objective function to calculate the average expected travel time of passengers through a 
  # transit network based on Spiess 1989 used by John and Mumford 2014
  
  # Input other constants
  walkFactor = 3 # factor it takes longer to walk than to drive
  boardingTime = 0.1 # assume boarding and alighting time = 6 seconds
  alightingTime = 0.1 # problem when alighting time = 0 (good test 0.5)(0.1 also works)
  large_dist = max(Distance_mx) # the large number from the distance matrix
  alpha_const_inter = 0.7 # constant for interarrival times relationship (Spiess 1989)
  
  # 4.) Creating the transit network -----
  namedRoutes <- formatRoutesWithLetters(R_routes) # name each node in the routes
  
  names_of_transit_routes <- as.character() # only contains the transit route names
  
  for (i in 1:length(namedRoutes)) {
    for (j in 1:length(namedRoutes[[i]])) {
      names_of_transit_routes <- c(names_of_transit_routes,namedRoutes[[i]][j])
    }
  }
  
  n_nodes <- nrow(Distance_mx)
  names_transit_network = c(1:n_nodes,names_of_transit_routes)
  n_transit_nodes = length(names_transit_network)
  
  # make sure max(Distance_mx) is a VERY much larger number
  mx_transit_network = matrix(0, ncol = n_transit_nodes, nrow = n_transit_nodes)
  mx_C_a = matrix(large_dist, ncol = n_transit_nodes, nrow = n_transit_nodes)
  mx_f_a = matrix(0, ncol = n_transit_nodes, nrow = n_transit_nodes)
  # create all the matrices
  
  # transit network contains all the travel links
  colnames(mx_transit_network) = rownames(mx_transit_network) =
    colnames(mx_C_a) = rownames(mx_C_a) =
    colnames(mx_f_a) = rownames(mx_f_a) = names_transit_network
  # name all the links in the network
  
  
  for (i in 1:n_nodes) {
    for (j in 1:n_nodes) {
      if(Distance_mx[i,j] == large_dist){
        
        mx_C_a[i,j] = Distance_mx[i,j]
        
      }else{
        if (Distance_mx[i,j] == 0) {
          mx_transit_network[i,j] = 0
          mx_C_a[i,j] = walkFactor*Distance_mx[i,j]
          mx_f_a[i,j] = 0
        }else{
          mx_transit_network[i,j] = 1
          mx_C_a[i,j] = walkFactor*Distance_mx[i,j]
          mx_f_a[i,j] = Inf
        }
      } 
      
    }  
  } # fill in the walk links that are present in the graph
  
  #match(names_transit_network[19],names_transit_network)
  
  #as.numeric(substring(names_transit_network[19],2))
  counter = 1
  
  for (i in 1:(length(R_routes))) {
    for (j in 1:(length(R_routes[[i]])) ) {
      
      i_index = as.numeric(substring(names_of_transit_routes[counter],2)) # number of the transit node
      j_index = match(names_of_transit_routes[counter],names_transit_network) # position of the transit node in network graph
      
      mx_transit_network[i_index, j_index] = 1   
      mx_C_a[i_index, j_index] = boardingTime # sets the boarding and alighting
      mx_f_a[i_index, j_index] = F_frequencies[i] # set the frequencies per transit line
      
      counter = counter + 1  
    }
  } # fill in the boarding links characteristics
  
  counter = 1
  
  for (i in 1:(length(R_routes))) {
    for (j in 1:(length(R_routes[[i]])) ) {
      
      
      i_index = as.numeric(substring(names_of_transit_routes[counter],2)) # number of the transit node
      j_index = match(names_of_transit_routes[counter],names_transit_network) # position of the transit node in network graph
      
      mx_transit_network[j_index, i_index] = 1   
      mx_C_a[j_index, i_index] = alightingTime # sets the boarding and alighting
      mx_f_a[j_index, i_index] = Inf # set the frequencies per transit line
      
      counter = counter + 1  
    }
  } # fill in the alighting links characteristics
  
  
  for (i in 1:(length(names_of_transit_routes) - 1)) {
    if (substring(names_of_transit_routes[i],1,1) == 
        substring(names_of_transit_routes[i+1],1,1))
    {
      
      i_index = match(names_of_transit_routes[i],names_transit_network)
      j_index = match(names_of_transit_routes[i+1],names_transit_network)
      
      mx_transit_network[i_index, j_index] = 
        mx_transit_network[j_index, i_index] = 1 
      
      mx_C_a[i_index, j_index] = 
        mx_C_a[j_index, i_index] = 
        Distance_mx[as.numeric(substring(names_of_transit_routes[i],2)),
          as.numeric(substring(names_of_transit_routes[i+1],2))]
      
      mx_f_a[i_index, j_index] = 
        mx_f_a[j_index, i_index] = Inf
      
    }
    
  } # fill in the travel times using the transit lines / routes
  
  names_df_transit_links <- c("I_i", "I_j", "c_a","f_a")
  df_transit_links <- data.frame(matrix(ncol = length(names_df_transit_links), nrow = 0))
  colnames(df_transit_links) = names_df_transit_links
  
  counter = 1
  for (i in 1:n_transit_nodes) {
    for (j in 1:n_transit_nodes) {
      if(mx_transit_network[i,j]){
        
        df_transit_links[counter,1] = names_transit_network[i]
        df_transit_links[counter,2] = names_transit_network[j]
        df_transit_links[counter,3] = mx_C_a[i,j]
        df_transit_links[counter,4] = mx_f_a[i,j]
        
        counter = counter + 1
      }
      
    }
  } # put all the links in one matrix
  
  # Convert the df to a character matrix so that it is in correct format
  # unlist(df_transit_links)
  list_transit_links = as.matrix(df_transit_links, ncol = length(names_df_transit_links))
  
  names_df_transit_links <- c("I_i", "I_j", "c_a","f_a")
  links_list <- as.numeric(matrix(ncol = length(names_df_transit_links), nrow = 0))
  links_list = list_transit_links
  colnames(links_list) = names_df_transit_links
  
  names_all_nodes = names_transit_network # set the names of all the nodes in transit network
  names_I_nodes = as.character(1:n_nodes) # set the names of only the main nodes called I
  
  mx_D_a = Demand_mx # assign the demand matrix to the correct name
  colnames(mx_D_a) = rownames(mx_D_a) = names_I_nodes
  
  
  # 5.) Create objects to store data -----
  
  # Create objects to keep the node and arc volumes
  number_of_nodes = length(names_all_nodes)
  volumes_nodes = rep(0,number_of_nodes)
  
  volumes_links = matrix(0, ncol = number_of_nodes, nrow = number_of_nodes)
  colnames(volumes_links) = rownames(volumes_links) = names_all_nodes
  
  
  # 6.) Optimal strategy algorithm (Spiess, 1989) -----
  
  # Overall for loop to change the destinations
  for (i_destination in 1:length(names_I_nodes)) {
    
    #for (i_destination in 10) {
    
    # 6.1) Initialise algortihm =====
    
    # Create the data frame to keep the answers in
    df_opt_strat_alg = createNamedDataFrame("index","a=(i,","j)","f_a","u_j+c_a","a_in_A_bar")
    nodes_u_i = createNamedDataFrame(names_all_nodes)
    nodes_f_i = createNamedDataFrame(names_all_nodes)
    
    # Set values of the first row
    r_destination = names_I_nodes[i_destination]
    nodes_u_i[1,] = Inf
    nodes_u_i[1,match(r_destination,colnames(nodes_u_i))] = 0 # set the destination expected time
    nodes_f_i[1,] = 0
    S_list = links_list
    
    A_bar_strategy_lines = NULL
    
    repeat{ # repeats steps 6.2 and 6.3 until S_list is empty
      # 6.2) Get the next link=====
      
      row_counter = nrow(nodes_u_i) # sets a counter to keep track of where the algorithm is
      # TESTS
      # row_counter = match(r_destination,colnames(nodes_u_i))
      
      S_list_counter = ifelse(is.null(nrow(S_list)),1,nrow(S_list))
      
      #if(nrow(S_list) == (119-41)){
      # browser()
      #} 
      
      if(!S_list_counter == 1){
        
        for(i in 1:S_list_counter){ # loop through S_list to find the minimum u_j + c_a
          
          if (i == 1) {
            u_j = nodes_u_i[row_counter,match(S_list[i,2],colnames(nodes_u_i))]
            c_a = as.numeric(S_list[i,3])
            min_u_j_and_c_a = u_j + c_a
            min_u_j_and_c_a_index = i
            
          }else{
            u_j = nodes_u_i[row_counter,match(S_list[i,2],colnames(nodes_u_i))]
            c_a = as.numeric(S_list[i,3])
            if(u_j + c_a <= min_u_j_and_c_a){
              
              min_u_j_and_c_a = u_j + c_a
              min_u_j_and_c_a_index = i
              
            }
          }
        }
      } else {
        u_j = nodes_u_i[row_counter,match(S_list[2],colnames(nodes_u_i))]
        c_a = as.numeric(S_list[3])
        min_u_j_and_c_a = u_j + c_a
        
      }
      
      
      # 6.3) Update the node label =====
      if(!S_list_counter==1){
        current_link = S_list[min_u_j_and_c_a_index,]
      }else{
        current_link = S_list # selects the last link
      }
      
      
      row_counter = nrow(nodes_u_i) # sets a counter to keep track of where the algorithm is
      col_index_i = match(current_link[1],colnames(nodes_u_i))
      
      u_i = nodes_u_i[row_counter,col_index_i]
      f_i = nodes_f_i[row_counter,col_index_i]
      f_a = as.numeric(current_link[4])
      
      if(u_i >= min_u_j_and_c_a){
        
        if(f_a == Inf || f_i == Inf){ # for the case where the modification is needed in Spiess (1989) for no waiting time
          #if(f_a == Inf){
          nodes_u_i[row_counter,col_index_i] = min_u_j_and_c_a 
          nodes_f_i[row_counter,col_index_i] = Inf
          A_bar_strategy_lines = rbind(A_bar_strategy_lines, current_link)
          df_opt_strat_alg[nrow(links_list)-S_list_counter+1,6] = TRUE
          
        }else{ # normal case when a link is added
          nodes_u_i[row_counter,col_index_i] = 
            (f_i_u_i_test(f_i,u_i,alpha_const_inter) + f_a*(min_u_j_and_c_a))/(f_i+f_a)
          nodes_f_i[row_counter,col_index_i] = f_i + f_a
          A_bar_strategy_lines = rbind(A_bar_strategy_lines, current_link)
          df_opt_strat_alg[nrow(links_list)-S_list_counter+1,6] = TRUE
        }
      }else{
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,6] = FALSE
      }
      
      if(!S_list_counter==1){
        # Update the log of the algorithm used to store answers
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,2] = S_list[min_u_j_and_c_a_index,1]
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,3] = S_list[min_u_j_and_c_a_index,2]
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,4] = f_a
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,5] = min_u_j_and_c_a
        
        S_list = S_list[-min_u_j_and_c_a_index,]
      }else{
        # Update the log of the algorithm used to store answers
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,2] = S_list[1]
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,3] = S_list[2]
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,4] = f_a
        df_opt_strat_alg[nrow(links_list)-S_list_counter+1,5] = min_u_j_and_c_a
        
        S_list=NULL
        break
      } 
    } # end of repeat
    
    
    # 7.) Assign demand according to optimal strategy -----
    
    # 7.1) Initialise the algorithm ======
    # load the volumes of demand per node, called V_i
    
    V_i = rep(0, number_of_nodes)
    for (i in 1:ncol(mx_D_a)) {
      V_i[match(names_I_nodes[i],names_all_nodes)] = mx_D_a[names_I_nodes[i],r_destination]
    }
    V_i[match(r_destination, names_all_nodes)] = - sum(V_i)
    
    # NB this needs to hold to the conservation of flow requirements
    # colnames(V_i) = names_all_nodes
    # also the actual demand values can be input here
    
    links_volume_list = cbind(df_opt_strat_alg,rep(0,nrow(df_opt_strat_alg)))
    colnames(links_volume_list)[7] = "v_a"
    
    # 7.2) Load the links according to demand and frequencies ======
    for (i in nrow(df_opt_strat_alg):1) { # for every link in decreasing order of u_j + c_a
      if (df_opt_strat_alg[i,6]) {
        
        #index_link = which(links_volume_list[,1]==df_opt_strat_alg[i,2] & 
        #                    links_volume_list[,2]==df_opt_strat_alg[i,3])
        
        if (!links_volume_list[i, 2] == r_destination) { # this restricts the alg to assign negative demand to 
          # the outgoing nodes from the node that is being evaluated
          # also note, errors might come in when demand is wrongfully assigned out, and in.
          
          # set the indices
          node_i_index = match(links_volume_list[i, 2], names_all_nodes)
          node_j_index = match(links_volume_list[i, 3], names_all_nodes)
          
          # assign the v_a values
          if(!links_volume_list[i,4]==Inf){
            links_volume_list[i, 7] = (links_volume_list[i,4]/
                                         nodes_f_i[row_counter, node_i_index])*V_i[node_i_index]
          } else {
            links_volume_list[i, 7] = V_i[node_i_index]
          }
          
          # assign the V_j values                                                
          V_i[node_j_index] = V_i[node_j_index] + links_volume_list[i, 7]                                                                                        
        }
      }
      
    }
    
    
    # Update the volumes overall
    volumes_nodes = volumes_nodes + V_i
    
    counter_link = 1  
    while (counter_link <= nrow(links_volume_list)) {
      if(links_volume_list[counter_link,"a_in_A_bar"]){
        volumes_links[links_volume_list[counter_link,2],links_volume_list[counter_link,3]] = 
          volumes_links[links_volume_list[counter_link,2],links_volume_list[counter_link,3]] + links_volume_list[counter_link,"v_a"]
        
      }
      counter_link = counter_link + 1 
      
    } # end while
    
  } # end the overall destination change for loop spanning from 6.)
  
  # Add the volume per arc details to the list_transit_links object
  list_transit_links = cbind(list_transit_links, rep(0,nrow(list_transit_links)))
  colnames(list_transit_links)[5] = "v_a"
  
  for (i in 1:nrow(list_transit_links)) {
    list_transit_links[i,"v_a"] = volumes_links[list_transit_links[i,"I_i"],list_transit_links[i,"I_j"]]
  } # end for
  
  return(sum(volumes_links*mx_C_a)/sum(Demand_mx))
}

f4_total_buses_required = function(Distance_mx, R_routes, F_frequencies){
  # Objective function used to calculate the number of buses required based on John and Mumford 2014
  return(sum(2*F_frequencies*calculateRouteLengths4(distMatrix = Distance_mx,routes = R_routes)))
  
}

# Genetic Algorithm functions ---------------

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

GA_for_UTFSP = function(functions_to_min,decision_variables){
  
  
}

crossover_real = function(matingPool){
  # a function that performs single crossover of the matingpool uniformly
  # NB, the mating pool should be an even number
  for (i in seq(2, nrow(matingPool), by=2)) {
    
    parent_indices = sample(1:nrow(matingPool),2,replace = FALSE)
    parent_A = matingPool[parent_indices[1],]
    parent_B = matingPool[parent_indices[2],]
    matingPool = matingPool[-parent_indices,]
    
    crossover_index = sample(1:(varNo - 1),1,replace = FALSE)
    child_A = parent_A
    child_B = parent_B
    child_A[1:(crossover_index)] = parent_B[1:(crossover_index)]
    child_B[1:(crossover_index)] = parent_A[1:(crossover_index)]
    
    if (i != 2) {
      childAfterX = rbind(childAfterX, child_A, child_B)
    } else {
      childAfterX = rbind( child_A, child_B)
    }
    
  } # end for loop
  rownames(childAfterX) = NULL
  return(childAfterX)
}

mutation_real = function(childAfterX, mutation_prob, theta_set){
  # function for doing mutation on the population given
  # the mutation with probability mutation_prob
  for (i in 1:nrow(childAfterX)) {
    for (j in 1:ncol(childAfterX)) {
      
      if(runif(1,0,1) < mutation_prob){
        if(childAfterX[i,j] == 1){
          if(runif(1,0,1) < 0.5){
            childAfterX[i,j] = length(theta_set)
          } else {
            childAfterX[i,j] = 2
          }
        } else {
          if(childAfterX[i,j] == length(theta_set)){
            if(runif(1,0,1) < 0.5){
              childAfterX[i,j] = length(theta_set) - 1 
            } else {
              childAfterX[i,j] = 1
            }
          } else {
            if(runif(1,0,1) < 0.5){
              childAfterX[i,j] = childAfterX[i,j] - 1 
            } else {
              childAfterX[i,j] = childAfterX[i,j] + 1
            }
          }
        }
      }
    } # end j for loop
  } # end i for loop
  return(childAfterX)
}
