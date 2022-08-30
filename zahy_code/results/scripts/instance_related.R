library(ggplot2)
library(plyr)
library(gridExtra)
library(Cairo)
library(reshape2)

######################################################################
# P40: Multiple histograms for instance stats for selection  
#####################################################################
setwd('~/gdrivezb9/rushhour/results/pilot/')
i=read.csv('instance_stats.csv')
p=read.csv('paths.csv')
p=subset(p, complete=='True')
names(i)[names(i) == 'jsonfile'] <- 'instance'
d=merge(p,i, by = 'instance', all=FALSE)
p1<-ggplot(d, aes(x = d$v_size)) + geom_histogram(binwidth = 1) + xlab('#vertical cars') 
p3<-ggplot(d, aes(x = d$mag_nodes)) + geom_histogram(binwidth = 1)+ xlab('#nodes in mag')
p4<-ggplot(d, aes(x = d$path_length)) + geom_histogram(binwidth = 1)+ xlab('path length')
p5<-ggplot(d, aes(x = d$mag_edges)) + geom_histogram(binwidth = 1)+ xlab('#edges in mag')
p6<-ggplot(d, aes(x = d$num_sccs)) + geom_histogram(binwidth = 1)+ xlab('#SCC')
p7<-ggplot(d, aes(x = d$max_scc_size)) + geom_histogram(binwidth = 1)+ xlab('max SCC size')
grid.arrange(p4,p1,p5,p3,p6,p7, ncol=2, top='Pilot instances')


##########################################################################
# P41: Heatmap of correlation of difficulty and instance charecteristics
#########################################################################
setwd('~/gdrivezb9/rushhour/results/pilot/')
p=read.csv('paths.csv')
i=read.csv('instance_stats.csv')
names(i)[names(i) == 'jsonfile'] <- 'instance'
d=merge(p,i, by = 'instance')
dn=d[,c(3,4,6,11,12,13,14,15,17,18)]
dn$err_norm <- (dn$human_length-dn$optimal_length)/dn$optimal_length
dn$err <- (dn$human_length-dn$optimal_length)
cormat<-round(cor(dn, method = 'spearman'),2)
library(reshape2)
melted_cormat <- melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + geom_text(aes(label = melted_cormat$value))

