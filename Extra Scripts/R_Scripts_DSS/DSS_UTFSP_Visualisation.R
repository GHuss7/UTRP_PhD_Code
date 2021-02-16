# Formatting a graph with transport links

source("./DSS_Visualisation_Functions.R")

R_literal <- formatRoutes(R_max)

R_literal

g7 <- graph_from_literal(14-13-10-11-12-4-2-3-6-8,
                         11-13-10-7-15-9,
                         1-2-3-6-8-10-7-15-9,
                         6-3-2-5-4-12-11,
                         2-5-4-6-15-8-10-14-13-11,
                         13-10-7-15-8-6-4-5-2-1)

plot(g7)

R_literal_min <- formatRoutes(R_min)

g8 <- graph_from_literal(13-14-10-8-15,
                         4-2-3-6,
                         13-11-12,
                         7-15-9,
                         5-4-6-15,
                         3-2-1)

plot(g8)

g9 <- graph_from_literal(A13-A14-A10-A8-A15,
                         B4-B2-B3-B6,
                         C13-C11-C12,
                         D7-D15-D9,
                         E5-E4-E6-E15,
                         F3-F2-F1)

plot(g9)

coordinatesX <- c(13,14,10,8,15,
                  4,2,3,6,
                  13,11,12,
                  7,15,9,
                  5,4,6,15,
                  3,2,1) 

coordinatesY <- -1*c(rep(1,5),
                  rep(2,4),
                  rep(3,3),
                  rep(4,3),
                  rep(5,4),
                  rep(6,3))

coords <- cbind(coordinatesX,coordinatesY)

tkplot(g9,layout = coords)

# Visualisation for ORSSA 2019

g <- graph.adjacency(mx_transit_network, weighted=TRUE, mode = "directed")

plot_coords = as.matrix(read.csv(paste("./Input_Data/",problemName,"/plot_coords.csv", sep=""))[,-1])

for(i in 1:nrow(mx_transit_network)){
  
  for(j in 1:nrow(mx_transit_network)){
    
    E(g)[get.edge.ids(g,c(i,j))]$weight <- mx_C_a[i,j]  # assign weigths to edges
    E(g)[get.edge.ids(g,c(i,j))]$volume <- volumes_links[i,j] # assign volumes to the edges
    E(g)[get.edge.ids(g,c(i,j))]$frequency <- mx_f_a[i,j] # assign frequency to the edges
    
  }
  
} # end for

E(g)$width <- 0.2 # Assign widths to the graph (in pixels...)

E(g)$color <- "black" 

plot(g, edge.arrow.size=0.1, vertex.color="lightgrey", vertex.size=10, 
     
     vertex.frame.color="black", vertex.label.color="black", 
     edge.label.cex = 1,
     vertex.label.cex=1,
     layout = plot_coords,
     #edge.label = paste(E(g)$volume," (",E(g)$weight,",",E(g)$frequency,")",sep = ""),
     edge.label = paste(trunc(E(g)$volume),paste("(",E(g)$weight,",",round(E(g)$frequency,2),")",sep = ""),sep = "\n"),
     edge.width = E(g)$width,
     edge.curved = 0.3
)


##### Visualisation of extemal points of UTFSP


