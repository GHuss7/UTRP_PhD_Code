# DSS Visualisation Functions

formatRoutes <- function(x){
  # A function to display the routes in the standard format
  formattedRoutes <- as.list(NULL)
  
  for(i in 1:length(x)) {           
    
    formattedRoutes[[i]] <- paste(x[[i]],sep = "", collapse = "-")
    
  }
  return(formattedRoutes)
}

formatRoutes2 <- function(x){
  # A function to display the routes in the standard format seperated with "\n"
  formattedRoutes <- as.vector(NULL)
  stringRoutes <- as.vector(NULL)
  
  for(i in 1:length(x)) {           
    
    stringRoutes <- paste(x[[i]],sep = "", collapse = "-")
    formattedRoutes <- paste(formattedRoutes,stringRoutes,"\n",sep = "", collapse ="")
  }
  return(formattedRoutes)
}

formatRoutes3 <- function(x_routes_Min, x_routes_Max){
  # A function to get two route sets of same length to be displayed side by side in a tabular format in Latex
  formattedRoutes <- as.vector(NULL)
  stringRoutes <- as.vector(NULL)
  
  for(i in 1:length(x_routes_Min)) {           
    
    stringRoutes <- paste(paste(x_routes_Min[[i]],sep = "", collapse = "-"),"&",paste(x_routes_Max[[i]],sep = "", collapse = "-"))
    formattedRoutes <- paste(formattedRoutes,stringRoutes,"\\",sep = "", collapse ="")
  }
  return(formattedRoutes)
}

formatRoutesWithLetters <- function(x){
  # A function to add letters of the alphabet to each route node to distinguish 
  # x is a list of routes
  
  for(i in 1:length(x)) {           
    
    for (j in 1:length(x[[i]])) {
      x[[i]][j] <- paste(LETTERS[i],x[[i]][j], sep = "", collapse = "")
    }
  }
  return(x)
}

formatMatrix <- function(Rmatrix){
  # A function to display the routes in the standard format
  formattedMatrix <- as.list(NULL)
  
  for(i in 1:ncol(Rmatrix)) {           
   
    formattedMatrix[[i]] <- paste(Rmatrix[i,],sep = " ", collapse = "&")
    formattedMatrix[[i]] <- paste(formattedMatrix[[i]],"\\",sep = " ", collapse = "")
    
  }
  return(formattedMatrix)
}

createGraph <- function(distMatrix , coords){
  # make sure the distMatrix is in correct nxn as.matrix(distMatrix) [1:n,1:n] format
  
  ## Set matrix in correct format - make sure of this!!!!
  infeasibleDist <- max(distMatrix)
  
  for( i in 1:nrow(distMatrix)) { # NB - Makes sure of the format of the matrix!!! add or subtract one according to right shape
    
    for(j in 1:ncol(distMatrix)) {
      
      if(distMatrix[i,j] == infeasibleDist) {
        
        distMatrix[i,j] <- 0
        
      }
      
    }
    
  }
  
  # Create the graph from the adjacency matrix
  g <- graph.adjacency(distMatrix, weighted=TRUE, mode = "undirected")
  
  # Create a weight vector from the adjacency matrix to apply as weights
  wt <- as.vector(NULL)
  
  for(i in 1:nrow(distMatrix)){
    
    for(j in 1:nrow(distMatrix)){
      
      if(i>j & g[][i,j] != 0){
       
        E(g)[get.edge.ids(g,c(i,j))]$weight <- distMatrix[i,j]  # assign weigths to edges
      }
      
    }
    
  } # end for
  
  E(g)$width <- 1 # Assign widths to the graph
  
  E(g)$color <- "grey" # assign colours to the graph
  
  return(g)
  
} # end createGraph function

createGraph2 <- function(distMatrix){
  # make sure the distMatrix is in correct nxn as.matrix(distMatrix) [1:n,1:n] format
  
  ## Set matrix in correct format - make sure of this!!!!
  infeasibleDist <- max(distMatrix)
  
  for( i in 1:nrow(distMatrix)) { # NB - Makes sure of the format of the matrix!!! add or subtract one according to right shape
    
    for(j in 1:ncol(distMatrix)) {
      
      if(distMatrix[i,j] == infeasibleDist) {
        
        distMatrix[i,j] <- 0
        
      }
      
    }
    
  }
  
  # Create the graph from the adjacency matrix
  g <- graph.adjacency(distMatrix, weighted=TRUE, mode = "undirected")
  
  # Create a weight vector from the adjacency matrix to apply as weights
  wt <- as.vector(NULL)
  
  for(i in 1:nrow(distMatrix)){
    
    for(j in 1:nrow(distMatrix)){
      
      if(i>j & g[][i,j] != 0){
        
        E(g)[get.edge.ids(g,c(i,j))]$weight <- distMatrix[i,j]  # assign weigths to edges
      }
      
    }
    
  } # end for
  
  E(g)$width <- 1 # Assign widths to the graph
  
  E(g)$color <- "grey" # assign colours to the graph
  
  return(g)
  
} # end createGraph function


createGraph3 <- function(distMatrix, routeSet){
  # make sure the distMatrix is in correct nxn as.matrix(distMatrix) [1:n,1:n] format
  # include a route set to plot
  
  ## Set matrix in correct format - make sure of this!!!!
  infeasibleDist <- max(distMatrix)
  
  for( i in 1:nrow(distMatrix)) { # NB - Makes sure of the format of the matrix!!! add or subtract one according to right shape
    
    for(j in 1:ncol(distMatrix)) {
      
      if(distMatrix[i,j] == infeasibleDist) {
        
        distMatrix[i,j] <- 0
        
      }
      
    }
    # WIP WIP WIP WIP
  }
  
  # Create the graph from the adjacency matrix
  g <- graph.adjacency(distMatrix, weighted=TRUE, mode = "undirected")
  
  # Create a weight vector from the adjacency matrix to apply as weights
  wt <- as.vector(NULL)
  
  for(i in 1:nrow(distMatrix)){
    
    for(j in 1:nrow(distMatrix)){
      
      if(i>j & g[][i,j] != 0){
        
        E(g)[get.edge.ids(g,c(i,j))]$weight <- distMatrix[i,j]  # assign weigths to edges
      }
      
    }
    
  } # end for
  
  E(g)$width <- 1 # Assign widths to the graph
  
  E(g)$color <- "grey" # assign colours to the graph
  
  return(g)
  
} # end createGraph function

addAdditionalEdges <- function(g,R){
# Adding additional edges to represent the routes -------
  colourNames <- c("red","green","orange","blue","darkgreen","turquoise",
                   "pink","blueviolet","brown","maroon","purple","magenta","lightgreen")
  
  if(length(R) > length(colourNames) ){ # tests if it is more than the predefined set
    colourNames <- sample(colors() , length(R) , replace = FALSE)
}
 
  for(i in 1:length(R)) {
  
    for(j in 1:(length(R[[i]]) - 1) ) {
    
      g <- add.edges(g , c(R[[i]][j] , R[[i]][j+1]) , attr=list(color=colourNames[i] , width = 2 ))
    
    }
  
  }
  return(g)
}

customGraphPlot <- function(g , titleName){
  
  if(missing(titleName)){
    titleName <- ""
  }
  
  plot(g, edge.arrow.size=0.3, vertex.color="lightgrey", vertex.size=13, 
       
       vertex.frame.color="black", vertex.label.color="black", 
       edge.label.cex = 1.2,
       vertex.label.cex=1.2,
       layout = coords,
       edge.label = E(g)$weight,
       edge.width = E(g)$width,
       # edge.curved = 0.5,
       main = titleName)
  
}

customGraphPlot2 <- function(g , coords, titleName){
  
  if(missing(titleName)){
    titleName <- ""
  }
  
  plot(g, edge.arrow.size=0.3, vertex.color="lightgrey", vertex.size=15, 
       
       vertex.frame.color="black", vertex.label.color="black", 
       edge.label.cex = 1.2,
       vertex.label.cex=1.2,
       layout = coords,
       edge.label = E(g)$weight,
       edge.width = E(g)$width,
       # edge.curved = 0.5,
       main = titleName)
  
}

customGraphPlot3 <- function(g , titleName){
  
  if(missing(titleName)){
    titleName <- ""
  }
  
  plot(g, edge.arrow.size=0.3, vertex.color="lightgrey", vertex.size=13, 
       
       vertex.frame.color="black", vertex.label.color="black", 
       edge.label.cex = 1.2,
       vertex.label.cex=1.2,
       edge.label = E(g)$weight,
       edge.width = E(g)$width,
       # edge.curved = 0.5,
       main = titleName)
  
}

customGraphPlotThesis <- function(g , titleName, coords){
  
  if(missing(titleName)){
    titleName <- ""
  }
  
  plot(g, 
       # === vertex
       vertex.size=13, 
       vertex.color="lightgrey",
       vertex.frame.color="black", 
       
       # === vertex label
       vertex.label.color="black", 
       vertex.label.family="Times", # Font family of the label (e.g."Times", "Helvetica")
       vertex.label.cex=1.2,
       
       # === edge
       edge.width = E(g)$width,
       edge.arrow.size=0.3,
       # edge.curved = 0.5,
       
       # === edge label
       edge.label.cex = 1.2,
       edge.label.color="black",
       edge.label.family="Times",
       edge.label = E(g)$weight,
       
       # layout
       layout = coords,
       
       main = titleName)
  
}


saveInPlotsFolder <- function(g,fileName){
 # function is used to save iGraph plots 
  mypath <- file.path("C:","Users","Günther","OneDrive - Stellenbosch University",
                      "Skripsie DSS","DSS","Plots",paste(fileName, ".png", sep = ""))
  
  png(file=mypath)
  customGraphPlot(g)
  dev.off()
  
  
}

# Visualising plots -----------

plotParetoFront <- function(archive){
  
  archiveDF <- createArchiveDF(archive)
  
  myPlot <- ggplot(data = archiveDF, mapping = aes(x = f1norm ,y = f2norm)) +
    geom_point(shape=8,color = "red")      # Use asterisks
  
  return(myPlot)
}

savePlotInFolder <- function(plot,fileName){
  # function to save a normal plot in a folder
  
  png(file=paste("./Plots",paste(fileName, ".png", sep = "")))
  plot
  dev.off()
  
} # end savePlotInFolder

savePlotInFolderPDF <- function(plot,fileName){
  # function to save a normal plot in a folder
  
  pdf(file=paste("./Plots/",paste(fileName, "pdf", sep = ".")))
  plot
  dev.off()
  
} # end savePlotInFolder
