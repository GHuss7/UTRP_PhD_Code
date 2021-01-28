# Bus Routes Generator App

# 0.) Load preliminary packages and custom functions ------------
libNames <- c( "shiny","ggplot2","igraph","png","plotly","DT")
lapply(libNames, library, character.only = TRUE) # load the required packages
rm(libNames)

source("DSS_Functions.R")
source("DSS_Visualisation_Functions.R")

# 1.) Global variables


# 2.) UI ------
ui <- fluidPage(
   
   # Application title
   titlePanel("Bus route generator"),
   tabsetPanel(type = "tabs",
               tabPanel("Main", 
                        fluidRow(
                          column(3,
                                 fileInput("distMatrixFile", "Choose graph distance matrix CSV file",
                                           accept = c(
                                             "text/csv",
                                             "text/comma-separated-values,text/plain",
                                             ".csv")
                                 ),
                                 
                                 fileInput("coordsFile", "Choose graph coordinates CSV file",
                                           accept = c(
                                             "text/csv",
                                             "text/comma-separated-values,text/plain",
                                             ".csv")
                                 ),
                                 
                                 fileInput("ODMatrixFile", "Choose OD demand matrix CSV file",
                                           accept = c(
                                             "text/csv",
                                             "text/comma-separated-values,text/plain",
                                             ".csv")
                                 ),
                                 
                                 sliderInput("numAllowedRoutes",
                                             "Enter the number of allowed routes:",
                                             min = 1,
                                             max = 20,
                                             value = 6),
                                 
                    
                                 sliderInput("range", "Range of minimum and maximum stops in a route:",
                                             min = 2, max = 20,
                                             value = c(3,10)),
                                 
                                 actionButton("generateInitialSolution",
                                              "Generate a solution"),
                                 actionButton("perturb",
                                              "Perturb"),
                                 # actionButton("runButton",
                                 #              "Run Random Solutions"),
                                 actionButton("runButton2",
                                              "Run"),
                                 actionButton("optimiseButton",
                                              "Optimise further"),
                                 actionButton("viewButton",
                                              "view")
                                 
                          ),
                          column(5, offset = 0,
                                 tableOutput("values"),
                                 plotOutput("routesPlot"),
                                 verbatimTextOutput("routeDisplay")
                                 
                          ),
                          column(4,
                                 plotlyOutput("paretoPlotly"),
                                 numericInput("viewSetNumber",
                                              "Select the set number to view from the archive:",
                                              min = 1,
                                              max = 20,
                                              value = 1),
                                 actionButton("viewSolution",
                                              "View solution")
                          )
                        )
                        ),
               tabPanel("Simulated Annealing Parameters", 
                        numericInput("initialTemperature",
                                     "Initial Temperature:",
                                     value = 10000),
                        numericInput("iterationLc",
                                     "Iterarions per epoch:",
                                     value = 10),
                        numericInput("minAcceptedAmin",
                                     "Minimum number of accepted moves per epoch:",
                                     value = 1),
                        numericInput("maxPassCmax",
                                     "Maximum number of epochs which may pass without accepting a new solution:",
                                     value = 2),
                        numericInput("multiStart",
                                     "Number of multi starts:",
                                     value = 10),
                        numericInput("maxRuntime",
                                     "Maximum run time in minutes:",
                                     value = 2)
                        
                        ),
               tabPanel("Archive",
                        DT::dataTableOutput("mytable"),
                        h4("Click the download button to download the routes and their objective functions"),
                        downloadButton("downloadData", "Download")
                        )
   )
   
) # end of UI 

# 3.) Server ------
server <- function(input, output, session) {
  
  # Databases ======
  archive4Display <- as.list(NULL)
  dispCounter <- 1
  
  # Reactive values ========
  reacVals <- reactiveValues() # creates a object to store all reactive values in
  
  rv <- reactiveValues(i = 0) # for forcing plots
  maxIter <- 100
  
  coordsReac <- reactive({
    # Collect the correct co-ordinates of the graph
    infile <- input$coordsFile
    if (is.null(infile)) {
      # User has not uploaded a file yet
      return(NULL)
    }
    
    as.matrix(read.csv(infile$datapath))
    
  })
  
  SReac <- reactive({
    # Create and format a distance matrix S
    infile <- input$distMatrixFile
    if (is.null(infile)) {
      # User has not uploaded a file yet
      return(NULL)
    }
    
    formatDistMatrix(read.csv(infile$datapath))
    
  })
  
  demandMatrixReac <- reactive({
    # Create and format the demand matrix
    infile <- input$ODMatrixFile
    if (is.null(infile)) {
      # User has not uploaded a file yet
      return(NULL)
    }

    formatDemandMatrix(read.csv(infile$datapath))
    
  })
  
  gReac <- reactive({
    if (is.null(SReac())) {
      # User has not uploaded a file yet
      return(NULL)
    }
    S <- SReac()
    #g <- createGraph(S,coords)
    createGraph2(S)
  })
  
  set_S_dist_matrix <- observe({
    reacVals$S <- SReac()
    reacVals$N <- nrow(reacVals$S) # define the number of nodes in the system
  })
  
  set_coords <- observe({
    reacVals$coords <- coordsReac()
  })
  
  set_demand_matrix <- observe({
    reacVals$demandMatrix <- demandMatrixReac()
  })
  
  set_graph_g <- observe({
    reacVals$g <- gReac()
  })
  
  # Shortest Routes Graph generation functions:======
  
  set_shortestRoutes <- observe({
    # Generate all the shortest routes
    if(is.null(reacVals$S)){ 
      return()}else{
    isolate({reacVals$shortestRoutes <- generateAllShortestRoutes(reacVals$S)
    
    # Calculate the shortest distance matrix for the candidate routes
    reacVals$shortDistMatrix <- calculateRouteLengths(reacVals$S,reacVals$shortestRoutes)
    })
      }
    })

  set_candidate_routes <- observe({
    
    input$range[1]
    input$range[2]
    # Create a shortened list and remove the routes longer than the specified number
    if(is.null(reacVals$shortestRoutes)){ 
      return()}else{
    reacVals$shortenedCandidateRoutes <- specifyNodesPerRoute(reacVals$shortestRoutes,input$range[1],input$range[2])
}
  })
  
  
  # Reactive expression to create data frame of all input values
  performanceValues <- reactive({
    # Generate Bus Network Dist Matrix
    if(!is.null(reacVals$xRoutes)){
    busNetworkDistMatrix <- generateBusNetworkDistMatrix(reacVals$S,reacVals$xRoutes)
    shortestBusRoutes <- generateAllShortestRoutes(busNetworkDistMatrix)
    tM <- generateTransferMatrix(reacVals$xRoutes,shortestBusRoutes,reacVals$N)
    demandMatrix <- reacVals$demandMatrix
    }
    
    data.frame(
      
      TRT = if(is.null(reacVals$xRoutes)){
        return("-")
      }else{
        paste(f1_totalRouteLength(reacVals$S,reacVals$xRoutes)," min", sep = "")
      },
      ATT = if(is.null(reacVals$xRoutes)){
        return("-")
      }else{
        paste(round(f2_averageTravelTime(reacVals$S,reacVals$demandMatrix,reacVals$xRoutes),2)," min", sep = "")
      } ,       
      d_0 = sum((tM %in% 0)*demandMatrix) / sum(demandMatrix),
      
      d_1 = sum((tM %in% 1)*demandMatrix) / sum(demandMatrix),
      
      d_2 = sum((tM %in% 2)*demandMatrix) / sum(demandMatrix),
      
      d_un = sum((tM %in% 3)*demandMatrix) / sum(demandMatrix),

      stringsAsFactors = FALSE)
    
  })
  
  # Results table ----
  output$values <- renderTable({
    performanceValues()
  })

  # Routes graphical plot ====
  output$routesPlot <- renderPlot({
    S <- SReac()
    coords <- coordsReac()
    g <- gReac()

    if(is.null(reacVals$S)){
      return()
    }else{
    
    if(is.null(reacVals$xRoutes)){
      customGraphPlot2(g,coords,"") # Plots the road network
    }
    else{
      gbus <- addAdditionalEdges(g,reacVals$xRoutes) # adding the bus network routes
      customGraphPlot2(gbus,coords,"") # Plots the road network
      
    }
    }
  })
  
  # Observe Events------
  observe({
    output$routesPlot <- renderPlot({
      S <- SReac()
      coords <- coordsReac()
      g <- gReac()
      
      if(is.null(reacVals$S)){
        return()
      }else{
        
        if(is.null(reacVals$xRoutes)){
          customGraphPlot2(g,coords,"") # Plots the road network
        }
        else{
          gbus <- addAdditionalEdges(g,reacVals$xRoutes) # adding the bus network routes
          customGraphPlot2(gbus,coords,"") # Plots the road network
          
        }
      }
    })
    
  }, priority = 20000)
  
  
  # Create initial solution ======
  observeEvent(input$generateInitialSolution, {
    reacVals$xRoutes <- generateFeasibleSolution(reacVals$shortenedCandidateRoutes,
                                                 isolate(input$numAllowedRoutes),
                                                 nrow(reacVals$S),100000) # first initial solution
  }) # end initial candidate solution
  
  # observeEvent(input$runButton, {
  #   archiveTemp <- as.list(NULL) 
  #   for (i in 1:3) {
  #   x <- generateFeasibleSolution(reacVals$shortenedCandidateRoutes,
  #                                                isolate(input$numAllowedRoutes),
  #                                                nrow(reacVals$S),100000) # first initial solution
  #   reacVals$xRoutes <- x
  #   archiveTemp[[i]] <- list(f1_totalRouteLength(reacVals$S,x), f2_averageTravelTime(reacVals$S,reacVals$demandMatrix,x), x)
  #   }
  #   reacVals$archive <- archiveTemp
  #   })
  
  # Visualise a solution in archive =======
  observeEvent(input$viewSolution, {
    archive <- reacVals$archive
    reacVals$xRoutes <- archive[[input$viewSetNumber]][[3]]
    
  })
 
   output$routeDisplay <- renderText({
     if(is.null(reacVals$xRoutes)){
       return()
     }else{
     paste0("Routes: \n", formatRoutes2(reacVals$xRoutes))
     }
   })
   
   # Pareto plot  =======
   output$paretoPlotly <- renderPlotly({
     if(is.null(reacVals$archive)){
       return()
     }else{
     archive <- reacVals$archive
     arcDF <- createArchiveDF(archive)
     plot_ly(arcDF, x = ~f1, y = ~f2, mode = "markers", type="scatter",
             text = arcDF$SetNr,
             hoverinfo = 'text'
            )%>%
       layout(
         title = "Undominated attainment front",
           xaxis = list(title = "Total route time (min)"),
           yaxis = list(title = "Average travel time (min)")
         )
     }
   })
   
   output$mytable = DT::renderDataTable({
     archive <- reacVals$archive
     createArchiveDF4SA(archive)
   })
   
   # Update numeric set size input
   #updateNumericInput(session, viewSetNumber, label = NULL, value = 1,
    #                  min = 1, max = length(archive), step = NULL)
   
# Plot force chunk ==========
   observeEvent(input$viewButton, {
     rv$i <- 0
     observe({
       isolate({
         rv$i <- rv$i + 1
         reacVals$xRoutes <- archive4Display[[rv$i]][[3]]
       })
       
       if (isolate(rv$i) < maxIter){
         invalidateLater(2000, session)
       }
     })
   })
   
   
   
# Simulated Annealing Implementation============
   
   observeEvent(input$runButton2, {
  
      archive <- as.list(NULL) 
  
       
       
      x <- generateFeasibleSolution(reacVals$shortenedCandidateRoutes,
                                     isolate(input$numAllowedRoutes),
                                     nrow(reacVals$S),100000) # first initial starting solution
       
       isolate(reacVals$xRoutes <- x)
       
       archive[[1]] <- list(f1_totalRouteLength(reacVals$S,x), f2_averageTravelTime(reacVals$S,reacVals$demandMatrix,x), x)
       archive4Display[[1]] <- archive[[1]]
       dispCounter <- dispCounter + 1
       
     isolate(reacVals$archive <- archive)
     
     # SA Parameters input
     N <- reacVals$N
     minNodes <- input$range[1]
     maxNodes <- input$range[2]
     S <- reacVals$S
     demandMatrix <- reacVals$demandMatrix
     archive  <- reacVals$archive  # Let archive be the associated archive
     Lc <- input$iterationLc # Let Lc be maximum allowable number length of iterations per epoch c
     Amin <- input$minAcceptedAmin # Let Amin be minimum number of accepted moves per epoch
     Cmax <- input$maxPassCmax # Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
     Temp <- input$initialTemperature # Let Temp be the starting temperature and a geometric cooling schedule is used on it
     multiStart <- input$multiStart  # Let multistart represent the number of multistarts that will occur
     
     #SimulatedAnnealing2(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp)
     
     
       # initiate algorithm
       c <- 1 # Initialise the cooling schedule epoch
       t <- 1 # Initialise the number of iterations 
       eps <- 0 # Initialise the number of epochs without an accepted solution
       archiveDF <- createArchiveDF4SA(archive) # create an archiveDF in the correct format
        

         
  
       while (eps <= Cmax) {
         A <- 0
         t <- 0
         while (t <= Lc & A < Amin) {

           # generate neighboring solutions for solution x

           x_pert <- makeSmallChange2(x,N,S,minNodes,maxNodes)
           #archive4Display[[dispCounter]] <- list(f1_totalRouteLength(S,x_pert), f2_averageTravelTime(S,demandMatrix,x_pert), x_pert)
           #dispCounter <- dispCounter + 1
           
            if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){

             for (i in 1:1000) {
               x_pert<-makeSmallChange2(x_pert,N,S,minNodes,maxNodes)
               #archive4Display[[dispCounter]] <- list(f1_totalRouteLength(S,x_pert), f2_averageTravelTime(S,demandMatrix,x_pert), x_pert)
               #dispCounter <- dispCounter + 1
               
               if(testFeasibility2(x_pert,N,minNodes,maxNodes)){
                 break
               }
             }

            }

           if(!testFeasibility2(x_pert,N,minNodes,maxNodes)){
             x_pert <- makeSmallChange(x,N,minNodes,maxNodes)
             #archive4Display[[dispCounter]] <- list(f1_totalRouteLength(S,x_pert), f2_averageTravelTime(S,demandMatrix,x_pert), x_pert)             
             #dispCounter <- dispCounter + 1
           }

           # Generate random number r
           r <- runif(1,0,1)

           # Test solution acceptance and add to archive if accepted and non-dominated
           if(r < probToAcceptNeighbor(archiveDF,x_pert,x,Temp,S,demandMatrix)) {
             isolate(reacVals$xRoutes <- x_pert)
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

       isolate(reacVals$archive <- archive)
   })
   
   # Simulated Annealing further optimisation========
   
   observeEvent(input$optimiseButton, {
     
     archive <- reacVals$archive 
     
     
     
     x <- generateFeasibleSolution(reacVals$shortenedCandidateRoutes,
                                   isolate(input$numAllowedRoutes),
                                   nrow(reacVals$S),100000) # first initial starting solution
     
     reacVals$xRoutes <- x
     
     archive[[1]] <- list(f1_totalRouteLength(reacVals$S,x), f2_averageTravelTime(reacVals$S,reacVals$demandMatrix,x), x)
     
     reacVals$archive <- archive
     
     # SA Parameters input
     N <- reacVals$N
     minNodes <- input$range[1]
     maxNodes <- input$range[2]
     S <- reacVals$S
     demandMatrix <- reacVals$demandMatrix
     archive  <- reacVals$archive  # Let archive be the associated archive
     Lc <- input$iterationLc # Let Lc be maximum allowable number length of iterations per epoch c
     Amin <- input$minAcceptedAmin # Let Amin be minimum number of accepted moves per epoch
     Cmax <- input$maxPassCmax # Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
     Temp <- input$initialTemperature # Let Temp be the starting temperature and a geometric cooling schedule is used on it
     multiStart <- input$multiStart  # Let multistart represent the number of multistarts that will occur
     
     #SimulatedAnnealing2(x,N,minNodes,maxNodes,S,demandMatrix,archiveTemp,Lc,Amin,Cmax,Temp)
     
     
     # initiate algorithm
     c <- 1 # Initialise the cooling schedule epoch
     t <- 1 # Initialise the number of iterations 
     eps <- 0 # Initialise the number of epochs without an accepted solution
     archiveDF <- createArchiveDF4SA(archive) # create an archiveDF in the correct format
     
     
     
     
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
           reacVals$xRoutes <- x_pert
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
     
     reacVals$archive <- archive
   })
   
   
   # Create one perturbation =======
   observeEvent(input$perturb, {
     # Make 1 perturbation
     
     x <- reacVals$xRoutes

     N <- reacVals$N
     minNodes <- input$range[1]
     maxNodes <- input$range[2]
     S <- reacVals$S
     demandMatrix <- reacVals$demandMatrix
     
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
         reacVals$xRoutes <- x_pert
   })
   
   
   # Download Dataset Results
   output$downloadData <- downloadHandler(
     filename = function() {
       fileNameText <- substr(Sys.time(),1,10)
       paste("archive", fileNameText, ".text", sep = "")
     },
     content = function(file) {
       archive <- reacVals$archive
       #archive <- createArchiveDF4SA(archive)
       write.table(archive, file)
     }
   )
   
} # end server

# Run the application ----
shinyApp(ui = ui, server = server)

