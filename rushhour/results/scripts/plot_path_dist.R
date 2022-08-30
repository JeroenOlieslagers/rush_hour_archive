library(ggplot2)
library(grid)
library(gridExtra)
setwd("~/gdrivezb9/rushhour/results/pilot")
logl = read.csv('aspen_ll_es.csv', header = TRUE)

logl = read.csv('fit_A.csv', header = TRUE)


#model 0: 2.639057329615259 ((0.1815603171698146) (14 trials)
#model 1: 0.6931471805599453 ( 0.08750753,  0.19998829) (2 trials)
#model 3: 2.3978952727983707 (3.0770971577603987/27.859505401044196)  (11 trials)

ggplot(logl, aes(x=logl$model)) + geom_bar(stat="identity", aes(y=logl$ll))


#one instance, 3 models - 
# 1) Astar + lapse
# 2) Astar + search lapse 
# 3) Astar + stop criterion 
# 4) Astar + reconstruction lapse 

# sample, step, state, instance, alg
#dist<-read.csv('raw_path_dist1.csv',header = TRUE, sep = ' ')
#dist<-read.csv('raw_dist_algs.1.csv',header = TRUE, sep = ' ')
#dist<-read.csv('raw_path_dist_2.csv',header = TRUE, sep = ' ')
#dist<-read.csv('all_astar.csv',header = TRUE, sep = ' ')
dist<-read.csv('raw_dist_astars_easy1',header = TRUE, sep = ' ')

dists=split(dist,dist[,c('instance')])

# plot 1: distribution of 2-3 different models and real data 
# plot 2: logliklihood of 2-3 different models with different sampling sizes
# plot 3: Summary statistics - ratio of optimal vs. length of optimal


solution_lengths<-function(d){
  steps=aggregate(d[,c('sample','step')],by = list(d$sample), FUN = max)
  return(paste(unique(d$instance),unique(steps$step),sep = ':'))
}

#A*h:magsizef:1*(g+1)+1his_stop:<lambda>lapse_rate:0.184072757689,search_lapse:0 ->  0:1000 1:478 2:111 3:47 4:15 5:38 6:68
# 16.1520990724
#Checking A*h:magsizef:1*(g+1)+1his_stop:<lambda>lapse_rate:0.100012528878,search_lapse:0.277182451791 ->  0:1000 1:464 2:126 3:52 4:19 5:45 6:99
# 15.1729001771
#Checking A*h:magsizef:1*(g+1)+1his_stop:stop_Xlapse_rate:0.075196556899,search_lapse:0 ->  0:1000 1:464 2:165 3:65 4:27 5:9 6:33
# 17.0367453731

solution_length_plot<-function(d){
  alg_splits=split(d, d$alg)
  all_steps=list()
  for (x in alg_splits){
    steps=aggregate(x[,c('sample','step')],by = list(x$sample), FUN = max)
    steps$alg=unique(x$alg)
    all_steps=rbind(all_steps,steps)
  }
  g<-ggplot(all_steps,aes(x=all_steps$step,fill=all_steps$alg)) + xlab('length') + geom_bar(position = "dodge")+theme(legend.position="top", legend.direction="vertical", axis.title.x=element_blank(), axis.title.y=element_blank())
  return(g);
}

path_dist_plot<-function(d){
  g<-ggplot(d,aes(x=factor(d$state),fill=d$alg)) + geom_bar(position = "dodge")+theme(legend.position="none", axis.title.x=element_blank(),axis.title.y=element_blank())
  return(g);
}


slplots=lapply(dists, solution_length_plot)
do.call('grid.arrange',c(slplots, ncol = 5, top = "Solution Lengths"))


optimal_alg=subset(dist,dist$alg=='A*_h:zeroh_f:1*(g+1)+1h_is_stop:<lambda>_lapse_rate:0')

dist1=subset(dist,dist$alg=='A*_h:zeroh_f:1*(g+1)+1h_is_stop:<lambda>_lapse_rate:0' | dist$alg=='A*_h:zeroh_f:1*(g+1)+1h_is_stop:<lambda>_lapse_rate:0.01')

dists=split(dist1,dist1[,c('instance')])

slplots=lapply(dists, path_dist_plot)
do.call('grid.arrange',c(slplots, ncol = 5, top = "Path Distributions"))

