# DSS Visualisation


# source("./DSS_Main.R")
source("./DSS_Visualisation_Functions.R")

# Format the routes in the correct format -------
formattedRoutes <- formatRoutes(x)


# Format a Matrix for latex
RMatrix <- demandMatrix
formatMatrix(RMatrix)

# Visualise the graph and routes --------

# Collect the correct co-ordinates of the graph =============

    # coords <- layout.auto(g) # to generate coordinates for graph automatically
    # write.csv(coords,"MandlSwissNetworkCoords.csv") # use this to store the coords
    coords <- read.csv(file = "./Input_Data/Node_Coords.csv")
    coords <- as.matrix(coords)

# Create and plot a graph of the main network ----------
    g <- createGraph(S,coords)
    customGraphPlot(g,"Road network") # Plots the road network
    saveInPlotsFolder(g,"roadNetwork") # Saves the plot into a folder

# Create and plot a graph of the bus network ----------
    gb <- createGraph(busNetworkDistMatrix,coords)
    customGraphPlot(gb,"Bus network") # Plots the bus network    
    saveInPlotsFolder(gb,"busNetwork")
    
# Adding additional edges to represent the bus routes on the road network -------
    g2 <- addAdditionalEdges(g,x) # adding the bus network routes
    customGraphPlot(g2,"Bus routes on road network") # Plots the road network
    saveInPlotsFolder(g2,"busRoutesOnRoadNetwork")
    
    gb <- addAdditionalEdges(gb,x) # adding the bus network routes
    customGraphPlot(gb,"Bus routes only") # Plots the road network
    saveInPlotsFolder(gb,"busRoutesOnly")

    
# Visualisation Example: ------
    # Generate Bus Route Network
    
    # Generate Bus Network Dist Matrix
    busNetworkDistMatrix <- generateBusNetworkDistMatrix(S,x)
    
    # Determine all shortest routes in the Bus network from one node to another
    shortestBusRoutes <- generateAllShortestRoutes(busNetworkDistMatrix)
    
    # Calculate the shortest distance matrix for the candidate bus routes
    shortBusDistMatrix <- calculateRouteLengths(busNetworkDistMatrix,shortestBusRoutes)
    
    
    # Visualise the graph and routes
    
    # Create and plot a graph of the main network
    g <- createGraph(S,coords)
    customGraphPlot(g,"") # Plots the road network
    
    # Adding additional edges to represent the bus routes on the road network
    g <- addAdditionalEdges(g,x) # adding the bus network routes
    customGraphPlot(g,"") # Plots the road network

    
# Interactive tkplot:
    # tkplot(g,600,600, edge.arrow.size=0.3, vertex.color="gold", vertex.size=15,
    #      vertex.frame.color="gray", vertex.label.color="black",
    #      vertex.label.cex=0.8,
    #      layout = coords,
    #      edge.label = E(g)$weight,
    #      edge.width = E(g)$width)

# Visualise attainment fronts------
    # Save the plot 
    fileName <- paste("paretoFrontOf",numAllowedRoutes,"Routes", sep = "")
    mypath <- file.path("C:","Users","Günther","OneDrive - Stellenbosch University",
                        "Skripsie DSS","DSS","Plots",paste(fileName, ".png", sep = ""))
    
    png(file=mypath)
    myPlot
    dev.off()
    
    myPlot <- plotParetoFront(archiveList[[1]]) 
    myPlot
    
    ggplot() + 
      geom_point(data = createArchiveDF(archiveList[[1]]), aes(x = f1, y = f2), color = "red") +
      geom_point(data = createArchiveDF(archiveList[[2]]), aes(x = f1, y = f2), color = "blue") +
      geom_point(data = createArchiveDF(archiveList[[3]]), aes(x = f1, y = f2), color = "green") +
      geom_point(data = createArchiveDF(archiveList[[4]]), aes(x = f1, y = f2), color = "orange") +
      geom_point(data = createArchiveDF(archiveList[[5]]), aes(x = f1, y = f2), color = "pink") +
      geom_point(data = createArchiveDF(archiveList[[6]]), aes(x = f1, y = f2), color = "yellow") +
      geom_point(data = createArchiveDF(archiveList[[7]]), aes(x = f1, y = f2), color = "brown") +
      geom_point(data = createArchiveDF(archiveList[[8]]), aes(x = f1, y = f2), color = "cyan") +
      geom_point(data = createArchiveDF(archiveList[[9]]), aes(x = f1, y = f2), color = "grey") +
      geom_point(data = createArchiveDF(archiveList[[10]]), aes(x = f1, y = f2), color = "darkgreen") +
      xlab('f1') +
      ylab('f2')
    
    archiveDFTest <- createArchiveDF(archive)
    plotParetoFront(archive)
    
# PLotly examples:
    set.seed(955)
    # Make some noisily increasing data
    dat <- data.frame(cond = rep(c("A", "B"), each=10),
                      xvar = 1:20 + rnorm(20,sd=3),
                      yvar = 1:20 + rnorm(20,sd=3))
    
    p <- ggplot(dat, aes(x=xvar, y=yvar)) +
      geom_point(shape=1)      # Use hollow circles
    
    p <- ggplotly(p)
    
    # Create a shareable link to your chart
    # Set up API credentials: https://plot.ly/r/getting-started
    chart_link = plotly_POST(p, filename="geom_point/scatter")
    chart_link
    
# Visualise the extreme points: ------------
    archiveDFPareto <- createArchiveDF(archive)
    archiveDFParetoSA <- createArchiveDF4SA(archive)
    archiveDFParetoSA <- archiveDFParetoSA[,-1]
    archiveMaxf1DF <- archiveDFPareto[which.max(archiveDFPareto$f1),] # max f1
    archiveMinf1DF <- archiveDFPareto[which.min(archiveDFPareto$f1),] # min f1
    
    # min ATT, max TRT
    x <- archive[[which.max(archiveDFPareto$f1)]][[3]]
    
    # Create and plot a graph of the main network ----------
    g <- createGraph(S,coords)
    customGraphPlot(g,"") # Plots the road network
    saveInPlotsFolder(g,"maxExtremePoint") # Saves the plot into a folder
    
    # Adding additional edges to represent the bus routes on the road network -------
    g <- addAdditionalEdges(g,x) # adding the bus network routes
    customGraphPlot(g,"") # Plots the road network
    saveInPlotsFolder(g,"maxExtremePointRoutes")
    
    
    paste("TRT = ",f1_totalRouteLength(S,x), sep = "")
    paste("ATT = ",f2_averageTravelTime(S,demandMatrix,x), sep = "")
    paste("Routes:")
    formatRoutes(x)
    
    paste0("Routes: \n", formatRoutes2(x))
    
    # max ATT, min TRT 
    x <- archive[[which.min(archiveDFPareto$f1)]][[3]]
    
    # Create and plot a graph of the main network ----------
    g <- createGraph(S,coords)
    customGraphPlot(g,"") # Plots the road network
    saveInPlotsFolder(g,"minExtremePoint") # Saves the plot into a folder
    
    # Adding additional edges to represent the bus routes on the road network -------
    g <- addAdditionalEdges(g,x) # adding the bus network routes
    customGraphPlot(g,"") # Plots the road network
    saveInPlotsFolder(g,"minExtremePointRoutes")
    
    
    paste("TRT = ",f1_totalRouteLength(S,x), sep = "")
    paste("ATT = ",f2_averageTravelTime(S,demandMatrix,x), sep = "")
    paste("Routes:")
    formatRoutes(x)
    
    paste0("Routes: \n", formatRoutes2(x))
    
# Visualise SA example: -------- 
    i <- 1
    g <- createGraph(S,coords)
    
    
    if(i == 1){
    x <- generateFeasibleSolution(shortenedCandidateRoutes,numAllowedRoutes,nrow(S),10000) # first initial solution
    } else {
    x <- makeSmallChange(x,N,minNodes,maxNodes)
    }
    
    # Create and plot a graph of the main network ----------
   
    #customGraphPlot(g,"") # Plots the road network
    #saveInPlotsFolder(g,paste("examplePlot",i,sep = "")) # Saves the plot into a folder
    
    # Adding additional edges to represent the bus routes on the road network -------
    gbus <- addAdditionalEdges(g,x) # adding the bus network routes
    customGraphPlot(gbus,"") # Plots the road network
    # saveInPlotsFolder(gbus,paste("examplePlot",i,sep = ""))
    
    
    paste("TRT = ",f1_totalRouteLength(S,x)," min", sep = "")
    paste("ATT = ",round(f2_averageTravelTime(S,demandMatrix,x),2)," min", sep = "")
    paste("Routes:")
    formatRoutes(x)
    
    minRoutesTRT <- archive[[which.min(archiveDFPareto$f1)]][[1]]
    minRoutesATT <- archive[[which.min(archiveDFPareto$f1)]][[2]]
    maxRoutesTRT <- archive[[which.max(archiveDFPareto$f1)]][[1]]
    maxRoutesATT <- archive[[which.max(archiveDFPareto$f1)]][[2]]
    
    paste("TRT = ",minRoutesTRT," min", sep = "")
    paste("ATT = ",round((minRoutesATT),2)," min", sep = "")
    paste("TRT = ",maxRoutesTRT," min", sep = "")
    paste("ATT = ",round((maxRoutesATT),2)," min", sep = "")
    x_routes_min <- archive[[which.min(archiveDFPareto$f1)]][[3]]
    x_routes_max <- archive[[which.max(archiveDFPareto$f1)]][[3]]
  formatRoutes3(x_routes_min,x_routes_max)
    