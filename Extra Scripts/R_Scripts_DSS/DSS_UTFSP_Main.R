# Decision Support System for the Urban Transit Frequency Setting Problem

# Load Libraries -------
libNames <- c( "ggplot2", "igraph","png","plotly","tikzDevice","mco","nsga2R",
               "reshape2","plyr")
lapply(libNames, library, character.only = TRUE) # load the required packages
rm(libNames)

# 0.) Define user specified parameters ----------
# Select the working directory for this project
workingDirectory <- "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/Skripsie DSS (R)/DSS"
setwd(workingDirectory)
rm(workingDirectory)

# NB copy this from the folders as it is used in file names
 problemName <- "Mandl_Data" 
#problemName <- "ORSSA_example_2019" 
optimise_with_GA <- TRUE

# Create the folder for the results to be stored
resultsDir = paste("./SavedRData/Results_",substr(Sys.time(),1,10),"_",problemName,"_","Routes_Min_a", sep = "")


# Create the folder for the results to be stored
#resultsDir = paste("./SavedRData/Results_",substr(Sys.time(),1,10),"_",problemName,"_",numAllowedRoutes,"Routes_a", sep = "")

# Load other functions and scripts -------
source("./DSS_Functions.R")
source("./DSS_UTFSP_Functions.R")
source("./DSS_Visualisation_Functions.R")


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


# Input other constants
walkFactor = 100 # factor it takes longer to walk than to drive
boardingTime = 0.1 # assume boarding and alighting time = 6 seconds
alightingTime = 0.1 # problem when alighting time = 0 (good test 0.5)(0.1 also works)
large_dist = max(S) # the large number from the distance matrix
alpha_const_inter = 0.7 # constant for interarrival times relationship (Spiess 1989)

# 2.) Import the route set to be evaluated -----
R_routes = as.list(NULL)

if(problemName == "Mandl_Data"){
load(paste("./SavedRData/Results_2019-03-13_Mandl_Data_6Routes/archive6Routes2019-03-13_Final.Rdata", sep = ""))

archiveDFPareto <- createArchiveDF(archive2)
R_min <- archive2[[which.min(archiveDFPareto$f1)]][[3]]
R_max <- archive2[[which.max(archiveDFPareto$f1)]][[3]]
R_routes = R_min

rm(archive2,archiveDFPareto)

} else {
  
  routes_imported = read.csv(paste("./Input_Data/",problemName,"/Routes.csv", sep=""))
  for (i in 1:nrow(routes_imported)) {
    R_routes[[i]] = routes_imported[i,!is.na(routes_imported[i,])] 
  }  
}

# The routes for John 2016 OVERRIDE
R_routes = as.list(NULL)
R_routes[[1]] = c(4,3,1) +1
R_routes[[2]] = c(13,12) +1
R_routes[[3]] = c(8,14) +1
R_routes[[4]] = c(9,10,12) +1
R_routes[[5]] = c(9,6,14,7,5,2,1,0) +1
R_routes[[6]] = c(10,11) +1

# 3.) Initialise the decision variables -----
  # the idea is to encode the decision variables as the argument that is to be input into the 
  # theta set so that manipulations can be easier to conduct
theta_set = c(5,6,7,8,9,10,12,14,16,18,20,25,30)

F_x_arg = sample.int(length(theta_set), length(R_routes) , replace = TRUE)
F_x = 1/theta_set[F_x_arg]
F_x = rep(1/5, length(R_routes))

#F_x_arg = P_t_new[which.min(P_t_new[,7]),1:varNo] # min expected travel time
#F_x = 1/theta_set[c(3,3,10,1,2,11)]

#F_x_arg = P_t_new[which.min(P_t_new[,8]),] # min buses required
#F_x = 1/theta_set[c(10,13,13,10,13,13)]

# 4.) Creating the transit network -----
namedRoutes <- formatRoutesWithLetters(R_routes) # name each node in the routes

names_of_transit_routes <- as.character() # only contains the transit route names

for (i in 1:length(namedRoutes)) {
  for (j in 1:length(namedRoutes[[i]])) {
    names_of_transit_routes <- c(names_of_transit_routes,namedRoutes[[i]][j])
  }
}

n_nodes <- nrow(S)
names_transit_network = c(1:n_nodes,names_of_transit_routes)
n_transit_nodes = length(names_transit_network)

# make sure max(S) is a VERY much larger number
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
    if(S[i,j] == large_dist){
      
      mx_C_a[i,j] = S[i,j]

    }else{
      if (S[i,j] == 0) {
        mx_transit_network[i,j] = 0
        mx_C_a[i,j] = walkFactor*S[i,j]
        mx_f_a[i,j] = 0
      }else{
      mx_transit_network[i,j] = 1
      mx_C_a[i,j] = walkFactor*S[i,j]
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
  mx_f_a[i_index, j_index] = F_x[i] # set the frequencies per transit line
  
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
      S[as.numeric(substring(names_of_transit_routes[i],2)),
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
  
mx_D_a = demandMatrix # assign the demand matrix to the correct name
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
  min_u_j_and_c_a_index = i
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


# 8.) Optimisation using GA ------

if(optimise_with_GA){
# Input parameters -----

# Set the variables
Dist_mx = S
Demand_mx = mx_D_a
R_routes = R_routes
F_frequencies = F_x

# Create the objective function that will be evaluated
fn_obj = function(F_frequencies){
  return(c(f3_avg_expected_travel_time(Dist_mx,Demand_mx,R_routes,F_frequencies),
           f4_total_buses_required(Dist_mx,R_routes,F_frequencies)))
}

varNo = length(R_routes)
objDim = length(fn_obj(F_x))

fn = fn_obj # NB this should be a minimisation function
lowerBound = 0
upperBound = 1
lowerBounds = rep(0, varNo)
upperBounds = rep(1, varNo)
popSize = 50 # NB, should be even number
tourSize = 2 
generations = 4 
cprob = 0.8 
XoverDistIdx = 5
mprob = 0.1 
MuDistIdx = 10

# Initialise population P_0 with fitness -----
P_0 <- matrix(ncol = varNo, nrow = popSize)
F_0 <- matrix(ncol = objDim, nrow = popSize)
t <- 0

for (i in 1:nrow(P_0)) {
    F_x_arg = sample.int(length(theta_set), length(R_routes) , replace = TRUE)
    P_0[i,] <- F_x_arg# theta_set[F_x_arg] # = F_x
    F_0[i,] <- fn(1/theta_set[P_0[i,]]) 
}

# Set the population in the appropriate format -----
# NB this is set up for a MINIMISATION problem
set.seed(1234)
population <- P_0
#fitness <- t(apply(P_0, 1, fn))
fitness <- F_0
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
  #cprob <- 0.7
  #XoverDistIdx <- 5
  #childAfterX <- boundedSBXover(matingPool,lowerBounds,upperBounds,cprob,XoverDistIdx)
  childAfterX = crossover_real(matingPool)
  
  #childAfterX
  
  # Mutation -----
  
  # set.seed(1234)
  # matingPool <- childAfterX
  # childAfterM <- boundedPolyMutation(matingPool,lowerBounds,upperBounds,mprob,MuDistIdx)
  childAfterM = mutation_real(childAfterX, 1/varNo, theta_set)
  
  # sum(childAfterM != childAfterX)/(popSize*varNo) test it

  
  
  
  # Combine parent and offspring populations
  P_t = population[,1:varNo] # previous population
  Q_t = childAfterM # offspring of previous population
  R_t = rbind(P_t, Q_t)

  #fitness = t(apply(1/theta_set[R_t], 1, fn))
  fitness = NULL
  
  for (i in 1:nrow(R_t)) {
    f_input = NULL
    for (j in 1:ncol(R_t)) {
      f_input = c(f_input,R_t[i,j])
    }
    #fitness[i,] <- fn(1/theta_set[f_input])
    fitness = rbind(fitness, fn(1/theta_set[f_input]))
  } 
  
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


plotP_t(P_t_new)
plotP_t(P_t_new[P_t_new$rank==1,])



# Save the data
save(list = ls(all.names = TRUE), file = paste(resultsDir,"/workspace_test_",substr(Sys.time(),1,10),".Rdata", sep = ""), envir = .GlobalEnv)

write.csv(P_t_new[,7:8],"UTFSP_results.csv")

} # end optimise with GA

# X.) Visualise the graph -----

if (F) { # indicate if you want visualisations

# Visualise transit network

T10 <- createGraph2(mx_transit_network)
tkp <- tkplot(T10,label.color = "darkblue", vertex.color = "gold",
              labels = TRUE, edge.labels=TRUE)

# plot_coords <- tkplot.getcoords(14, norm = FALSE)

plot(T10)
# customGraphPlot2(T10,plot_coords)
customGraphPlot3(T10)
  
# Visualise transit network
G10 <- createGraph2(mx_C_a)
tkp <- tkplot(G10,label.color = "darkblue", vertex.color = "gold",
              labels = TRUE, edge.labels=TRUE)

plot_coords <- tkplot.getcoords(tkp, norm = FALSE)
write.csv(plot_coords,"plot_coords.csv")

#plot(G10)

# Visualise a graph
g <- createGraph(S,coords)
customGraphPlot(g,"") # Plots the road network

# Adding additional edges to represent the bus routes on the road network
g2 <- addAdditionalEdges(g,R_min) # adding the bus network routes
customGraphPlot(g2,"") # Plots the road network

# Adding additional edges to represent the bus routes on the road network
g3 <- addAdditionalEdges(g,R_max) # adding the bus network routes
customGraphPlot(g3,"") # Plots the road network
}


# Create graphs

if(F){ # create new graphs
  G10 <- createGraph2(mx_C_a)
  tkp <- tkplot(G10,label.color = "darkblue", vertex.color = "gold",
                labels = TRUE, edge.labels=TRUE,
                layout = plot_coords)
  
  plot_coords <- tkplot.getcoords(tkp, norm = FALSE)
  write.csv(plot_coords,"plot_coords.csv")
  # NB --- remember to copy and paste the coords in the correct folder
 
  # plot_coords = as.matrix(read.csv("plot_coords.csv")[,-1]) # how to read it 
}


g <- graph.adjacency(mx_transit_network, weighted=TRUE, mode = "directed")

plot_coords = as.matrix(read.csv(paste("./Input_Data/",problemName,"/plot_coords.csv", sep=""))[,-1])

for(i in 1:nrow(mx_transit_network)){
  
  for(j in 1:nrow(mx_transit_network)){
    
      E(g)[get.edge.ids(g,c(i,j))]$weight <- mx_C_a[i,j]  # assign weigths to edges
      E(g)[get.edge.ids(g,c(i,j))]$volume <- volumes_links[i,j] # assign volumes to the edges
      E(g)[get.edge.ids(g,c(i,j))]$frequency <- mx_f_a[i,j] # assign frequency to the edges
    
  }
  
} # end for

E(g)$width <- 0.2 # Assign widths to the graph

E(g)$color <- "black" 

plot(g, edge.arrow.size=0.1, vertex.color="lightgrey", vertex.size=7, 
     
     vertex.frame.color="black", vertex.label.color="black", 
     edge.label.cex = 0.7,
     vertex.label.cex=0.6,
     layout = plot_coords,
     #edge.label = paste(E(g)$volume," (",E(g)$weight,",",E(g)$frequency,")",sep = ""),
     edge.label = paste(trunc(E(g)$volume),paste("(",E(g)$weight,",",round(E(g)$frequency,2),")",sep = ""),sep = "\n"),
     edge.width = E(g)$width,
     edge.curved = 0.3
     )



ptm <- proc.time()
for(i_times in 1:1){
  f3_avg_expected_travel_time(S,demandMatrix,R_routes,F_x)
}
proc.time() - ptm
print("average time: 2.8 sec in R")
