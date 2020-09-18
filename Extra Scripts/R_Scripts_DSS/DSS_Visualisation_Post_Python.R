# Post Results Visualisation

# Load Libraries -------
list.of.packages <- c( "rstudioapi","ggplot2", "igraph","png","plotly","PythonInR","ecr","tidyverse") # list of packages to use
new_packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])] # IDs new packages
if(length(new_packages)>0){install.packages(new_packages)} #installs the new packages if required
lapply(list.of.packages, library, character.only = TRUE) # load the required packages
rm(list.of.packages,new_packages) # removes the list created

# Set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # gets and sets the directory of the current script

# source("./DSS_Main.R")
source("./DSS_Visualisation_Functions.R")

# Load other functions and scripts -------
source("./DSS_Functions.R")
source("./DSS_Admin_Functions.R")
source("./DSS_Visualisation_Functions.R")
library(extrafont)
#font_install('fontcm')
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

###### THESIS FUNCTIONS #######
customGraphPlotThesis <- function(g, coords, titleName){
  
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
       #vertex.label.family="Serif", # Font family of the label (e.g."Times", "Helvetica")
       vertex.label.cex=1.2,
       vertex.label = 0:(nrow(coords)-1),
       
       # === edge
       edge.width = E(g)$width,
       edge.arrow.size=0.3,
       #edge.curved = 0.1,
       
       # === edge label
       edge.label.cex = 1.2,
       edge.label.color="black",
       #edge.label.family="Serif",
       edge.label = E(g)$weight,
       
       # layout
       layout = coords,
       
       main = titleName)
  
} # end customGraphPlotThesis


plot_and_save_route_set_thesis <- function(routes_str, fileName, dist_mx, demand_mx, coords_mx){
  # function to save a normal plot in a folder
  g <- createGraph(dist_mx,coords_mx)
  
  R_routes = convertRouteStringToList(routes_str)
  for(i in 1:length(R_routes)){
    for (j in 1:length(R_routes[[i]])){
      R_routes[[i]][j] = R_routes[[i]][j] + 1
    }
  }
  
  g_R <- addAdditionalEdges(g,R_routes) # adding the bus network routes
  # names(pdfFonts())
  pdf(file=paste("./Thesis_plots/",paste(fileName, "pdf", sep = "."),sep = ""), height = 7, width = 7)
  customGraphPlotThesis(g_R, coords, "") # Plots the road network
  dev.off()
  
} # end plot_and_save_route_set_thesis


###### INPUT ROUTES ######

pdf(file=paste("./Thesis_plots/",paste("UTRP_TRANSIT_NETWORK_plot", "pdf", sep = "."),sep = ""), height = 7, width = 7) # fonts = "fontcm"
g <- createGraph(S,coords)
customGraphPlotThesis(g, coords, "") # Plots the road network
dev.off()

plot_and_save_route_set_thesis("0-1-4-3-5-14-6-9-13*0-1-2-5-7-9-13-12-10-11*0-1-2-5-7-14-6-9-10-12*8-14-6-9-10-11-3-1-0*0-1-2-5-14-8*2-1-4-3-5-7-9-10-12*", 
                               "UTRP_DBMOSA_ATT_MIN", 
                               S, demandMatrix, coords)

plot_and_save_route_set_thesis("12-10-9-6-14-7-5-2-1*0-1*1-3-4*8-14*11-10*13-12*", 
                               "UTRP_DBMOSA_TRT_MIN", 
                               S, demandMatrix, coords)

plot_and_save_route_set_thesis("12-13-9-6-14-5-2-1-0*0-1-3-11-10-12*11-10-9-6-14-8*0-1-4-3-5-7-14-6*10-9-7-5-3-4*0-1-2-5-7-9-10-12-13*", 
                               "UTRP_NSGAII_ATT_MIN", 
                               S, demandMatrix, coords)

plot_and_save_route_set_thesis("10-11*3-1-2-5-7-14-6-9-10-12*13-12*0-1*14-8*3-4*", 
                               "UTRP_NSGAII_TRT_MIN", 
                               S, demandMatrix, coords)

plot_and_save_route_set_thesis("4-3-1*13-12*8-14*9-10-12*9-6-14-7-5-2-1-0*10-11*", 
                               "John_2016_best_operator_obj", 
                               S, demandMatrix, coords)

  