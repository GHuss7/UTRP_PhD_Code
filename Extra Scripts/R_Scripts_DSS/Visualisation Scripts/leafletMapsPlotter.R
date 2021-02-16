# Script to help plot markers on leaflet

library(shiny)
library(leaflet)

r_colors <- rgb(t(col2rgb(colors()) / 255))
names(r_colors) <- colors()

ui <- fluidPage(
  leafletOutput("mymap",width = 1300, height = 900),
  p(),
  actionButton("recalc", "New points")
)

server <- function(input, output, session) {
  
  pointNames <- c(
  "Bus Stop on Cluver",
  "Food Science ",
  "Academia",
  "Bus Stop",
  "Majuba",
  "Endler",
  "Coetzenburg Sport Centre",
  "Neelsie Parking",
  "Admin A building",
  "Nooitgedacht",
  "Stellenbosch Town Hall",
  "Theology Faculty")
  
  pointNames2 <- as.character(c(1:12))
  
  points <- eventReactive(input$recalc, {
    #delf <- cbind(rnorm(40) * 2 + 13, rnorm(40) + 48)

    coordsC <- c(-33.929381,	18.874639,
    -33.925075,	18.87079,
    -33.927575,	18.866976,
    -33.931109,	18.864616,
    -33.934174,	18.871354,
    -33.934085,	18.8657,
    -33.940022,	18.8707,
    -33.93218,	18.866687,
    -33.933206,	18.862343,
    -33.925835,	18.857158,
    -33.936748,	18.861334,
    -33.938467,	18.863858,
    -33.941982,	18.866581
    )

    latLonCoords <- t(matrix(coordsC,nrow =2))
    latLonCoords <- cbind(latLonCoords[,2],latLonCoords[,1])
    latLonCoords
    
  }, ignoreNULL = FALSE)
  
  output$mymap <- renderLeaflet({
    leaflet() %>% #addTiles() %>% setView(	18.874639, -33.929381, 13) %>% 
      addProviderTiles(providers$Stamen.TonerLite,
                       options = providerTileOptions(noWrap = FALSE)
      ) %>%
      addCircleMarkers(data = points(), label =pointNames,
                 labelOptions = labelOptions(noHide = T,textOnly = FALSE,
                                             textsize = "30px",
                                             fillOpacity = 0.0,
                                             stroke = FALSE))
  })
}

shinyApp(ui, server)
