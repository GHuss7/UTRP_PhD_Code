# Post Results Visualisation

# Load Libraries -------
list.of.packages <- c( "rstudioapi","ggplot2", "igraph","png","plotly","PythonInR","ecr","tidyverse","stringr","tkplot") # list of packages to use
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
library(extrafont)
#font_install('fontcm')
# 0.) Define user specified parameters ----------

#problemName <- "SSML_STB_DAY_SUM_0700_1700" # NB copy this from the folders as it is used in file names
problemName <- list("Mandl_UTRP",
                    "Mumford0_UTRP",
                    "Mumford1_UTRP",
                    "Mumford2_UTRP",
                    "Mumford3_UTRP")[[3]] # NB copy this from the folders as it is used in file names

print_true = FALSE

# 1.) Load the appropriate files and data for the network ------------
# Create and format a distance matrix S
S <- read.csv(paste("./../../Input_Data/",problemName,"/Distance_Matrix.csv", sep=""))
S <- formatDistMatrix(S)

# Create and format the demand matrix
demandMatrix <- read.csv(paste("./../../Input_Data/",problemName,"/OD_Demand_Matrix.csv", sep=""))
demandMatrix <- formatDemandMatrix(demandMatrix)

# Collect the correct co-ordinates of the graph
# coords <- layout.auto(g) # to generate coordinates for graph automatically
# write.csv(coords,"MandlSwissNetworkCoords.csv") # use this to store the coords
coords <- read.csv(file = paste("./../../Input_Data/",problemName,"/Node_Coords.csv", sep=""))
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

customGraphPlotLarge <- function(g, coords, titleName){
  
  if(missing(titleName)){
    titleName <- ""
  }
  
  plot(g, 
       # === vertex
       vertex.size=4, 
       vertex.color="lightgrey",
       vertex.frame.color="black", 
       
       # === vertex label
       vertex.label.color="black", 
       #vertex.label.family="Serif", # Font family of the label (e.g."Times", "Helvetica")
       vertex.label.cex=1.4,
       vertex.label = 0:(nrow(coords)-1),
       
       # === edge
       edge.width = E(g)$width,
       edge.arrow.size=0.5,
       #edge.curved = 0.1,
       
       # === edge label
       edge.label.cex = 1,
       edge.label.color="black",
       #edge.label.family="Serif",
       edge.label = E(g)$weight,
       
       # layout
       layout = coords,
       
       main = titleName)
  
} # end customGraphPlotLarge


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
  pdf(file=paste("./savedFigures/",paste(fileName, "pdf", sep = "."),sep = ""), height = 7, width = 7)
  customGraphPlotThesis(g_R, coords, "") # Plots the road network
  dev.off()
  
} # end plot_and_save_route_set_thesis

plot_and_save_route_set_folder <- function(routes_str, fileName, dist_mx, demand_mx, coords_mx, folder){
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
  pdf(file=paste("./../../Figures/",folder,"/",paste(fileName, "pdf", sep = "."),sep = ""), height = 18, width = 18)
  customGraphPlotLarge(g_R, coords, "") # Plots the road network
  dev.off()
  
} # end plot_and_save_route_set_folder


###### INPUT ROUTES ######
# Rotate coordinates
if (FALSE) {
  temp = coords[1:nrow(coords),2]
  coords[1:nrow(coords),2] = coords[1:nrow(coords),1]
  coords[1:nrow(coords),1] = temp
}

# plot_and_save_route_set_thesis("0-1-4-3-5-14-6-9-13*0-1-2-5-7-9-13-12-10-11*0-1-2-5-7-14-6-9-10-12*8-14-6-9-10-11-3-1-0*0-1-2-5-14-8*2-1-4-3-5-7-9-10-12*", 
#                                "UTRP_DBMOSA_ATT_MIN", 
#                                S, demandMatrix, coords)
# 
# plot_and_save_route_set_thesis("12-10-9-6-14-7-5-2-1*0-1*1-3-4*8-14*11-10*13-12*", 
#                                "UTRP_DBMOSA_TRT_MIN", 
#                                S, demandMatrix, coords)
# 
# plot_and_save_route_set_thesis("12-13-9-6-14-5-2-1-0*0-1-3-11-10-12*11-10-9-6-14-8*0-1-4-3-5-7-14-6*10-9-7-5-3-4*0-1-2-5-7-9-10-12-13*", 
#                                "UTRP_NSGAII_ATT_MIN", 
#                                S, demandMatrix, coords)
# 
# plot_and_save_route_set_thesis("10-11*3-1-2-5-7-14-6-9-10-12*13-12*0-1*14-8*3-4*", 
#                                "UTRP_NSGAII_TRT_MIN", 
#                                S, demandMatrix, coords)
# 
# plot_and_save_route_set_thesis("4-3-1*13-12*8-14*9-10-12*9-6-14-7-5-2-1-0*10-11*", 
#                                "John_2016_best_operator_obj", 
#                                S, demandMatrix, coords)

if (problemName == "SSML_STB_DAY_SUM_0700_1700" & print_true == TRUE) {
  
pdf(file=paste("./savedFigures/",paste("Case_study_UTRP_TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 7, width = 7) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotThesis(g, coords, "") # Plots the road network
  dev.off()

plot_and_save_route_set_thesis("9-3-8-5-0-7-2-6*5-8-6-0-4-9-3-2-1*5-7*1-0-5-8-2*5-8-4-2-1-7-6*7-4-8-3*6-0-3-7-8-5-2*", 
                               "Case_study_UTRP_ATT_MIN", 
                               S, demandMatrix, coords) # f_1=3.018588598	f_2=94

plot_and_save_route_set_thesis("0-1*0-7*7-6-8*8-5*4-8*8-3*9-3*2-8*", 
                               "Case_study_UTRP_TRT_MIN", 
                               S, demandMatrix, coords) # f_1=7.790462175	f_2=19

plot_and_save_route_set_thesis("5-7-2-8-3-9*1-7*7-6*4-8-5*7-0*7-8*1-0*2-1-0-6-8-5*", 
                               "Case_study_UTRP_MID_CHOICE", 
                               S, demandMatrix, coords) # f_1=3.381915975	f_2=40
}

if (problemName == "Mandl_UTRP") {
  
  pdf(file=paste("./savedFigures/",problemName,"/",paste("TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 18, width = 18) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotLarge(g, coords, "") # Plots the road network
  dev.off()
  
  plot_and_save_route_set_folder("3-1-2-5-7-14-6-9*", 
                                 "HighestDemandPerTime", 
                                 S, demandMatrix, coords, problemName) 
  
  plot_and_save_route_set_folder("0-1-4-3-11-10-12-9-7-5-2*", 
                                 "HighestDemand", 
                                 S, demandMatrix, coords, problemName)
  
  plot_and_save_route_set_folder("1-2-5-3-11-10-12-13-9-7-14*0-1-4-3-5-2*0-1-2-5-3-11-10-12-9-7*", 
                                 "Test", 
                                 S, demandMatrix, coords, problemName)
  
  
}

if (problemName == "Mumford0_UTRP") {
  
  pdf(file=paste("./savedFigures/",problemName,"/",paste("TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 18, width = 18) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotLarge(g, coords, "") # Plots the road network
  dev.off()
  
  plot_and_save_route_set_folder("8-12-19-22-0-13-6-5-21*", 
                                 "HighestDemand", 
                                 S, demandMatrix, coords, problemName) 
  
  plot_and_save_route_set_folder("17-19-22-0-18*", 
                                 "HighestDemandPerTime", 
                                 S, demandMatrix, coords, problemName)
  
}

if (problemName == "Mumford1_UTRP") {
  
  pdf(file=paste("./savedFigures/",problemName,"/",paste("TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 18, width = 18) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotLarge(g, coords, "") # Plots the road network
  dev.off()
  
  plot_and_save_route_set_folder("2-0-23-3-66-69-38-36-45-58-33-50*", 
                                 "HighestDemandPerTime", 
                                 S, demandMatrix, coords, problemName) 
  
  plot_and_save_route_set_folder("39-18-54-36-45-41-34-51-56-8-26-49-42*", 
                                 "HighestDemand", 
                                 S, demandMatrix, coords, problemName)
  
}

if (problemName == "Mumford2_UTRP") {
  
  pdf(file=paste("./savedFigures/",problemName,"/",paste("TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 18, width = 18) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotLarge(g, coords, "") # Plots the road network
  dev.off()
  
  plot_and_save_route_set_folder("29-43-30-26-72-79-73-106-57-10-52-100-18-9-19-33-99-69*", 
                                 "HighestDemandPerTime", 
                                 S, demandMatrix, coords, problemName)
  
  plot_and_save_route_set_folder("29-43-30-26-72-79-73-106-57-10-52-100-18-9-19-33-99-69*", 
                                 "HighestDemand", 
                                 S, demandMatrix, coords, problemName) 
  
}

if (problemName == "Mumford3_UTRP" & print_true == TRUE) {
  
  pdf(file=paste("./savedFigures/",problemName,"/",paste("TRANSIT_NETWORK", "pdf", sep = "."),sep = ""), height = 18, width = 18) # fonts = "fontcm"
  g <- createGraph(S,coords)
  customGraphPlotLarge(g, coords, "") # Plots the road network
  dev.off()
  
  plot_and_save_route_set_folder("78-96-71-100-3-115-106-14-53-23-114*", 
                                 "HighestDemandPerTime", 
                                 S, demandMatrix, coords, problemName) 
  
  plot_and_save_route_set_folder("17-43-109-86-24-5-80-66-74-114-16-55-60-32-20*", 
                                 "HighestDemand", 
                                 S, demandMatrix, coords, problemName)
  
}



if (problemName == "Mumford1_UTRP") {
  if(F){
  #"""Function to set the coordinates for the instances"""
  g <- createGraph(S)
  
  tk_plot_id <- tkplot(g,
                       vertex.size=13, 
                       vertex.color="lightgrey",
                       vertex.frame.color="black", 
                       
                       # === vertex label
                       vertex.label.color="black", 
                       #vertex.label.family="Serif", # Font family of the label (e.g."Times", "Helvetica")
                       vertex.label.cex=1.2,
                       vertex.label = 0:(nrow(S)-1),
                       
                       # === edge
                       edge.width = E(g)$width,
                       edge.arrow.size=0.3,
                       #edge.curved = 0.1,
                       
                       # === edge label
                       edge.label.cex = 1.2,
                       edge.label.color="black",
                       #edge.label.family="Serif",
                       edge.label = E(g)$weight
                       
                       # layout
                       #layout = plot_coords
  )
  
  
  plot_coords = norm_coords(tk_coords(tk_plot_id))
  
  if(FALSE){
    write.csv(plot_coords, file=paste("./../../Input_Data/",problemName,"/Node_Coords_own.csv", sep=""), row.names = F, col.names = c("V1","V2"))
  }
  } # end overall if(F)
}

if (problemName == "Mumford0_UTRP") {
  plot_and_save_route_set_thesis("4-24*23-9-3-24-7-27-15-10-6-13-0-12-8*2-15*5-6-13-18-0-22-17-11-3-1-9*10-21-6-16-7-14-23-3-1*25-28*17-19*12-8-19-18-0-25-7-20-4*4-24-14-11-17-12-8*23-20-14*9-23*2-29-27-16-28-17-22-0-26*
  ", "Mumford0_attempt", S, demandMatrix, coords) # f_1, f_2 = 17.25575754	225
} 
