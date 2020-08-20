# DSS Validation etc

# 9.) Validation of results with Mumford -------------
mumfordResults6Routes <- read.csv("./Mandl's Swiss Network/MumfordResultsParetoFront.csv")
names(mumfordResults6Routes) <- c("f2","f1")

validationCompare <- ggplot()+ 
  geom_point(data = createArchiveDF(archive2), aes(x = f1, y = f2), color = "red") +
  geom_point(data = mumfordResults6Routes, aes(x = f1, y = f2), color = "blue")

validationCompare

# Validation of results with Mandl ------

archive <- removeAllDominatedSolutionsMinMin(archive)
archiveDF <- createArchiveDF(archive)

x_Mandl <- as.list(NULL)
x_Mandl[[1]] <- c(0,1,2,5,7,9,10,12) +1
x_Mandl[[2]] <- c(4,3,5,7,14,6) +1
x_Mandl[[3]] <- c(11,3,5,14,8) +1
x_Mandl[[4]] <- c(12,13,9) +1

f1_totalRouteLength(S,x_Mandl)
f2_averageTravelTime(S,demandMatrix,x_Mandl)
feasibilityConnectedRoutes(x_Mandl,15)

x_Mandl <- as.list(NULL)
x_Mandl[[1]] <- c(1,2,3,6,8,10,11,13)
x_Mandl[[2]] <- c(9,15,6,4,12,11,13,14)
x_Mandl[[3]] <- c(14,10,7,15,6,4,2,1)
x_Mandl[[4]] <- c(12,11,10,8,6,4,5,2)

f1_totalRouteLength(S,x_Mandl)
f2_averageTravelTime(S,demandMatrix,x_Mandl)
feasibilityConnectedRoutes(x_Mandl,15)

x_Mandl <- as.list(NULL)
x_Mandl[[1]] <- c(1,2,3,6,15,7,10,11)
x_Mandl[[2]] <- c(12,11,13,14,10,7,15,9)
x_Mandl[[3]] <- c(1,2,5,4,6,8,10,11)
x_Mandl[[4]] <- c(1,2,3,6,8,10,13,11)
x_Mandl[[5]] <- c(1,2,4,12,11,10,14,13)
x_Mandl[[6]] <- c(1,2,5,4,6,8,15,7)

# Mumford's test that she sent me
# x_Mandl <- as.list(NULL)
# x_Mandl[[1]] <- c(12, 10, 9, 7, 5, 2, 1, 0) +1
# x_Mandl[[2]] <- c(12, 13, 9, 7, 5, 3, 11) +1
# x_Mandl[[3]] <- c(6, 14, 7, 5, 2, 1, 4) +1
# x_Mandl[[4]] <- c(8, 14, 5, 3, 11, 10, 12, 13) +1
# x_Mandl[[5]] <- c(11, 10, 9, 7, 5, 3, 1, 2) +1
# x_Mandl[[6]] <- c(12, 10, 9, 6, 14, 5, 3, 4) +1

f1_totalRouteLength(S,x_Mandl)
f2_averageTravelTime(S,demandMatrix,x_Mandl)
feasibilityConnectedRoutes(x_Mandl,15)

# Generate Bus Route Network

# Generate Bus Network Dist Matrix
busNetworkDistMatrix2 <- generateBusNetworkDistMatrix(S,x_Mandl)

# Determine all shortest routes in the Bus network from one node to another
shortestBusRoutes_Mandl <- generateAllShortestRoutes(busNetworkDistMatrix2)

tM <- generateTransferMatrix(x_Mandl,shortestBusRoutes_Mandl,N)

# Determine other parameters to test the network
sum((tM %in% 0)*demandMatrix) / sum(demandMatrix)

sum((tM %in% 1)*demandMatrix) / sum(demandMatrix)

sum((tM %in% 2)*demandMatrix) / sum(demandMatrix)

# Extreme points
xTest <- archive[[16]][[3]]
N <- 15
# Generate Bus Network Dist Matrix
busNetworkDistMatrix2 <- generateBusNetworkDistMatrix(S,xTest)

# Determine all shortest routes in the Bus network from one node to another
shortestBusRoutes_Mandl <- generateAllShortestRoutes(busNetworkDistMatrix2)

tM <- generateTransferMatrix(xTest,shortestBusRoutes,N)

# Determine other parameters to test the network
sum((tM %in% 0)*demandMatrix) / sum(demandMatrix)

sum((tM %in% 1)*demandMatrix) / sum(demandMatrix)

sum((tM %in% 2)*demandMatrix) / sum(demandMatrix)

sum((tM %in% 3)*demandMatrix) / sum(demandMatrix)

f2_averageTravelTime(S,demandMatrix,xTest)
