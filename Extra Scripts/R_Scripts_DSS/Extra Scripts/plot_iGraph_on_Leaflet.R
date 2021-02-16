# Mash Leaflet and iGraph

library(sp)
library(leaflet)

workingDirectory <- "C:/Users/Günther/OneDrive - Stellenbosch University/Skripsie DSS/DSS"
setwd(workingDirectory)
rm(workingDirectory)

S <- read.csv("./Input_Data (Stellies)/Distance_Matrix.csv")
S <- formatDistMatrix(S)

# Create and format the demand matrix
demandMatrix <- read.csv("./Input_Data (Stellies)/OD_Demand_Matrix.csv")
demandMatrix <- formatDemandMatrix(demandMatrix)

# Collect the correct co-ordinates of the graph
# coords <- layout.auto(g) # to generate coordinates for graph automatically
# write.csv(coords,"MandlSwissNetworkCoords.csv") # use this to store the coords
coords <- read.csv(file = "./Input_Data (Stellies)/Node_Coords.csv")
coords <- as.matrix(coords)

################################################
g <- createGraph2(S)
customGraphPlot(g,"Road network")

# 3.) Determine all the shortest routes ---------
shortestRoutes <- generateAllShortestRoutes(S)

# Calculate the shortest distance matrix for the candidate routes
shortDistMatrix <- calculateRouteLengths(S,shortestRoutes)

# Create a shortened list and remove the routes longer than the specified number
N <- nrow(S) # define the number of nodes in the system

# Generate Bus Network Dist Matrix ========
shortenedCandidateRoutes <- specifyNodesPerRoute(shortestRoutes,minNodes,maxNodes)

# 4.) Generate initial feasible solution ----------
x <- generateFeasibleSolution(shortenedCandidateRoutes,numAllowedRoutes,nrow(S),10000) # first initial solution


busNetworkDistMatrix <- generateBusNetworkDistMatrix(S,x)

# Determine all shortest routes in the Bus network from one node to another-----
shortestBusRoutes <- generateAllShortestRoutes(busNetworkDistMatrix)

# Calculate the shortest distance matrix for the candidate bus routes-----------------------
shortBusDistMatrix <- calculateRouteLengths(busNetworkDistMatrix,shortestBusRoutes)

g <- addAdditionalEdges(g,x)

customGraphPlot(g,"Road network")

############################################## Transform graph to spatial object
gg <- get.data.frame(g, "both")
vert <- gg$vertices
coordinates(vert) <- coords

edges <- gg$edges

edges <- lapply(1:nrow(edges), function(i) {
  as(rbind(vert[vert$name == edges[i, "from"], ], 
           vert[vert$name == edges[i, "to"], ]), 
     "SpatialLines")
})


for (i in seq_along(edges)) {
  edges[[i]] <- spChFIDs(edges[[i]], as.character(i))
}

edges <- do.call(rbind, edges)



leaflet(vert) %>% addTiles() %>% addCircleMarkers(data = vert,radius = 4,
                                                  label = as.character(c(1:12))) %>% 
  addPolylines(data = edges)

