## The ringbuffer package installation is required
install.packages("https://cran.r-project.org/src/contrib/Archive/ringbuffer/ringbuffer_1.1.tar.gz", repos=NULL)

##-------------------------------------------##
# Sample usage for simple Quadratic function  #
# 10 Dimmensions, budget = 100000 Evaluations,#
# constraints: lower= -100, upper= 100        #
##-------------------------------------------##
DES(rep(0,10),lower=-100,upper=100,fn=function(x){sum(x^2)},control=list("budget"=100000))