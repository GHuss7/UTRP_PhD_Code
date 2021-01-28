library("tikzDevice")
# Example 1 -----
tikz('normal.tex', standAlone = TRUE, width=5, height=5)

# Normal distribution curve
x <- seq(-4.5,4.5,length.out=100)
y <- dnorm(x)

# Integration points
xi <- seq(-2,2,length.out=30)
yi <- dnorm(xi)

# plot the curve
plot(x,y,type='l',col='blue',ylab='$p(x)$',xlab='$x$')
# plot the panels
lines(xi,yi,type='s')
lines(range(xi),c(0,0))
lines(xi,yi,type='h')

#Add some equations as labels
title(main="$p(x)=\\frac{1}{\\sqrt{2\\pi}}e^{-\\frac{x^2}{2}}$")
int <- integrate(dnorm,min(xi),max(xi),subdivisions=length(xi))
text(2.8, 0.3, paste("\\small$\\displaystyle\\int_{", min(xi),
                     "}^{", max(xi), "}p(x)dx\\approx", round(int[['value']],3),
                     '$', sep=''))

#Close the device
dev.off()

# Compile the tex file
tools::texi2dvi('normal.tex',pdf=T)

# optionally view it:
# system(paste(getOption('pdfviewer'),'normal.pdf'))


# Example 2 ------
tikz('./simpleEx.tex',width=3.5,height=3.5)
plot(1,main='Hello World!')
dev.off()


### Search Space ---------

tikzFileName <- paste('searchSpace',numAllowedRoutes,'.tex',sep="")
tikz(tikzFileName, standAlone = TRUE, width=5, height=5)

pPlot <- ggplot()
colourNames <- c("red","green","blueviolet","blue","darkgreen","turquoise",
                 "pink","orange","brown","maroon","purple","magenta","lightgreen",
                 "gold","black")

for (i in 1:epoch) { # generate a plot of all the solutions in each archive
  
  pPlot <- pPlot + geom_point(data = createArchiveDF(archiveList[[i]]), aes(x = f1, y = f2), color = colourNames[i])
  
}

pPlot

#Close the device
dev.off()

# Compile the tex file
tools::texi2dvi('normal.tex',pdf=T)

# optionally view it:
# system(paste(getOption('pdfviewer'),'normal.pdf'))

#### Pareto fronts ------

numRoutes <- numAllowedRoutes
tikzFileName <- paste('paretoFront',numRoutes,"routes",'.tex',sep="")
tikz(file = tikzFileName, width = 7,
     height = 7, onefile = TRUE, bg = "transparent", fg = "black",
     pointsize = 10, lwdUnit = getOption("tikzLwdUnit"), standAlone = FALSE,
     bareBones = FALSE, console = FALSE, sanitize = FALSE,
     engine = getOption("tikzDefaultEngine"),
     documentDeclaration = getOption("tikzDocumentDeclaration"), packages,
     footer = getOption("tikzFooter"),
     symbolicColors = getOption("tikzSymbolicColors"),
     colorFileName = "%s_colors.tex",
     maxSymbolicColors = getOption("tikzMaxSymbolicColors"),
     timestamp = TRUE, verbose = interactive())

# options(
#   tikzSanitizeCharacters = c('%','$','}','{','^','_','#','&','~'),
#   tikzReplacementCharacters = c('\\%','\\$','\\}','\\{','\\^{}','\\_{}',
#                                 '\\#','\\&','\\char`\\~')
# )


# Normal distribution curve
x <- archiveDF$f1
y <- archiveDF$f2


# plot the curve
plot(x,y,col='red',ylab='Average travel time (min)',xlab='Total route length (min)',type = "p")

#Add some equations as labels
title(main=paste("Attainment front for ",numRoutes," routes",sep=""))

text(2.8, 0.3, paste("\\small$\\displaystyle\\int_{", min(x),
                     "}^{", max(x), "}p(x)dx\\approx", 0.95,
                     '$', sep=''))

#Close the device
dev.off()

# Compile the tex file
tools::texi2dvi(tikzFileName,pdf=T)

# optionally view it:
# system(paste(getOption('pdfviewer'),'normal.pdf'))
