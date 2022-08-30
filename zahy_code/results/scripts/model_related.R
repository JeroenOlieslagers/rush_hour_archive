library(ggplot2)
library(plyr)
library(gridExtra)
library(Cairo)
library(reshape2)


######################################################## 
# Plot Error ratio of models.
#######################################################
setwd("~/gdrivezb9/rushhour/results/lrta")
lrta=read.csv('lrta_phs.csv', header = TRUE)
alg_lvls=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','A*h:min_manhattan_distancef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0','A*h:min_manhattan_distancef:0*(g+1)+1.5histop:<lambda>lapse:0slapse:0reca:100.0','A*h:magsizef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0')
alg_lbl=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','manhat','1.5manhat','magsize')
lrta$instance <- factor(lrta$instance, levels = lvls_e, labels = 1:length(lvls_e))
lrta$alg <- factor(lrta$alg, levels = alg_lvls , labels = alg_lvls)

lrta=subset(lrta, step==1)
lrta$error_ratio = lrta$alg_path/lrta$optimal_path
lrta$learning_iters = lrta$learning_iters*lrta$h_weight
lrta$learning_iters = factor(lrta$learning_iters)
ggplot(lrta, aes(x=instance, y=error_ratio)) + geom_bar(stat = 'identity',position = 'dodge',aes(fill=alg),size=2) +
  ggtitle('Effect of learning on errors') 


######################################################## 
# Summary stat analysis of the instances
# Heatmap of spearman correlations.
########################################################
setwd('~/gdrivezb9/rushhour/results/instances')
d=read.csv('instances.csv')
res=round(cor(d[-1], method = 'spearman'),3)
melted_cormat<- melt(res, na.rm = TRUE)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + geom_text(aes(label=melted_cormat$value))

######################################################## 
# LRTA distributions 
#######################################################
setwd("~/gdrivezb9/rushhour/results/pilot/lrta")
d=read.csv('lrta_fitting_exp.csv', header = T, sep = ',')

######################################################## 
# LRTA distributions 
#######################################################

setwd("~/gdrivezb9/rushhour/results/pilot/lrta")
no_exp=read.csv('dist_lrta_best', header= FALSE, sep = ',')
exp3=read.csv('dist_lrta_exp3', header= FALSE, sep = ',')
exp5=read.csv('dist_lrta_exp5', header= F, sep = ',')
exp50=read.csv('dist_lrta_exp50', header= F, sep = ',')
p1<-ggplot(no_exp, aes(x=V1)) + geom_histogram(binwidth = 1) + ggtitle('max node')
p2<-ggplot(exp3, aes(x=V1)) + geom_histogram(binwidth = 1)+ ggtitle('k=3')
p3<-ggplot(exp5, aes(x=V1)) + geom_histogram(binwidth = 1)+ ggtitle('k=5')
p4<-ggplot(exp50, aes(x=V1)) + geom_histogram(binwidth = 1)+ ggtitle('k=50')
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=subset(paths, paths$complete=='True' & paths$instance=='Jam-1')
p5<-ggplot(d, aes(x=d$human_length))+ geom_histogram(binwidth = 1)+ ggtitle('Subject Data')
grid.arrange(p5,p1,p4,p3,p2, ncol=3, top='Random Selection of nodes')


######################################################## 
# LRTA fitting MCMC
#######################################################

setwd("~/gdrivezb9/rushhour/results/pilot/lrta/")
d=read.csv('lrta_fitting_exp.csv', header= TRUE, sep = ',')
d=read.csv('lrta_subs.csv', header= TRUE, sep = ',')

#Percentage of 21 trials for data_point,subject
ratio=ddply(d, .(subject, h_epsilon, exp,learning_iter), function(x){return(length(which(x$trials=='21'))/nrow(x))})
ggplot(ratio, aes(x=ratio$V1)) + geom_histogram(binwidth = 0.05)
ggplot(d, aes(x=trials)) + geom_histogram(binwidth=1)

#s=split(d, list(d$subject,d$h_epsilon,d$learning_iter))
#x=s$andra.log.0.000686180040586.1
#ggplot(x, aes(x=trials)) + geom_histogram(binwidth=1)



######################################################## 
# Plot 33:   Error ratio of models.
#######################################################
setwd("~/gdrivezb9/rushhour/results/lrta")
lrta=read.csv('lrta_phs.csv', header = TRUE)
alg_lvls=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','A*h:min_manhattan_distancef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0','A*h:min_manhattan_distancef:0*(g+1)+1.5histop:<lambda>lapse:0slapse:0reca:100.0','A*h:magsizef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0')
alg_lbl=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','manhat','1.5manhat','magsize')
lrta$instance <- factor(lrta$instance, levels = lvls_e, labels = 1:length(lvls_e))
lrta$alg <- factor(lrta$alg, levels = alg_lvls , labels = alg_lvls)

lrta=subset(lrta, step==1)
lrta$error_ratio = lrta$alg_path/lrta$optimal_path
lrta$learning_iters = lrta$learning_iters*lrta$h_weight
lrta$learning_iters = factor(lrta$learning_iters)
ggplot(lrta, aes(x=instance, y=error_ratio)) + geom_bar(stat = 'identity',position = 'dodge',aes(fill=learning_iters),size=2) +
  scale_fill_grey(start = .8, end = .0) + theme_bw() + 
  ggtitle('Effect of learning on errors') 




########################################################### 
# Plot 34:   Scatter plot of average errors and algorithms
###########################################################
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True')
paths$human_error_ratio = paths$human_length/paths$optimal_length
t1=aggregate(paths, by = list('instance'=paths$instance), FUN='sd')$human_error_ratio
t2=aggregate(paths, by = list('instance'=paths$instance), FUN='length')$human_error_ratio
sem=t1/sqrt(t2)
paths_ag=aggregate(paths, by = list('instance'=paths$instance), FUN='mean')
paths_ag=paths_ag[,c('instance','human_error_ratio')]
paths_ag$sem=sem
setwd("~/gdrivezb9/rushhour/results/lrta")
lrta=read.csv('lrta_phs.csv', header = TRUE)
alg_lvls=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','A*h:min_manhattan_distancef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0','A*h:min_manhattan_distancef:0*(g+1)+1.5histop:<lambda>lapse:0slapse:0reca:100.0','A*h:magsizef:0*(g+1)+1histop:<lambda>lapse:0slapse:0reca:100.0')
alg_lbl=c('lrta0','lrta1','lrta2','lrta3','lrta4','lrta5','manhat','1.5manhat','magsize')
lrta$instance <- factor(lrta$instance, levels = lvls_e, labels = 1:length(lvls_e))
lrta$alg <- factor(lrta$alg, levels = alg_lvls , labels = alg_lbl)
lrta=subset(lrta, step==1)
lrta$alg_error_ratio = lrta$alg_path/lrta$optimal_path
lrta$learning_iters = lrta$learning_iters*lrta$h_weight
lrta$learning_iters = factor(lrta$learning_iters)
d=merge(lrta,paths_ag, by = 'instance')
ggplot(d, aes(x=human_error_ratio, y=alg_error_ratio)) + geom_point(aes(color=alg),size=2) + 
  + geom_text(cor(d$human_error_ratio,d$alg_error_ratio, method = 'spearman'))+
  facet_grid(. ~ alg) + geom_abline(slope = 1) + scale_fill_grey(start = .8, end = .0) + 
  geom_errorbar(aes(ymin=alg_error_ratio-sem, ymax=alg_error_ratio+sem))+
  theme_bw() + ggtitle('Error ratio of alg/humans') + ylim(1,5)


########################################################### 
# Plot 34.1:   Scatter plot of average errors and algorithms (LRTA5)
###########################################################
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True')
paths$human_error_ratio = paths$human_length/paths$optimal_length
paths_ag=aggregate(paths, by = list('instance'=paths$instance), FUN='mean')
paths_ag=paths_ag[,c('instance','human_error_ratio')]
setwd("~/gdrivezb9/rushhour/results/lrta")
lrta=read.csv('lrta.csv', header = TRUE)
lrta$instance <- factor(lrta$instance, levels = lvls_e, labels = 1:length(lvls_e))
lrta=subset(lrta, step==1)
lrta=subset(lrta, lrta$alg=='lrta5')
lrta$alg_error_ratio = lrta$alg_path/lrta$optimal_path
lrta$learning_iters = lrta$learning_iters*lrta$h_weight
lrta$learning_iters = factor(lrta$learning_iters)
d=merge(lrta,paths_ag, by = 'instance')

ggplot(d, aes(x=human_error_ratio, y=alg_error_ratio)) + geom_point(aes(color=learning_iters),size=2) +
  geom_abline(slope = 1) + scale_fill_grey(start = .8, end = .0) + theme_bw() + 
  ggtitle('Error ratio of alg/humans') 



####
#  Plot 29: Bar plot of which model explains difficulty better? (Pure H, manhattan, 0 heuristics.) 
#### 
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True' & !(paths$instance %in% c('22','23','24')))

p1<-ggplot(paths, aes(x=paths$instance, y=paths$human_length)) + stat_summary(geom='bar',fun.y = 'mean') +
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2) +
  stat_summary(geom='bar', aes(y=paths$optimal_length+1), fun.y = 'mean', fill='black')
p3<-ggplot(paths, aes(x=paths$instance, y=paths$rt)) + stat_summary(geom='bar',fun.y = 'mean') +
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)

d=read.csv('rid.csv', header = TRUE, sep=',',stringsAsFactors=F)
d$instance=factor(d$puzzle, levels = lvls_e, labels = 1:length(lvls_e))

p2<-ggplot(d, aes(x=instance)) + stat_summary(aes(y=expanded), geom='bar', fun.y = 'mean') + 
  facet_grid(alg ~ . ,scales = 'free')+
  stat_summary(aes(y=expanded), geom='errorbar', fun.data = mean_sem, width=0.2) 
grid.arrange(p1,p2,ncol = 1)

########
## P30 : Scatter plots of Spearman corellations between expanded nodes and error_ratios
##########
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True' & !(paths$instance %in% c('22','23','24')))
paths$optimal_length<-paths$optimal_length+1
rid=read.csv('rid.csv', header = TRUE, sep=',',stringsAsFactors=F)
rid$instance=factor(rid$puzzle, levels = lvls_e, labels = 1:length(lvls_e))
d=ddply(rid, .(instance,alg), function(x){return(mean(x$expanded))})
names(d)<-c('instance','alg','expanded')
d1=merge(d,paths,by = c('instance'))
d1$error_ratio=d1$human_length/d1$optimal_length
d1$error_n=d1$human_length-d1$optimal_length
s<-ddply(d1, .(subject, alg), function(x){return(cor(x$expanded, x$error_ratio, method = 'spearman'))} )
names(s)<-c('subject','alg','spearman')
dc<-dcast(s, subject ~ alg)
names(dc)<-c('subject', 'H','H+G','G')
p1<-ggplot(dc, aes(x=dc$H, y=dc$`H+G`)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ 
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' heuristics + distance')
p2<-ggplot(dc, aes(x=dc$H, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' distance only')
p3<-ggplot(dc, aes(x=dc$`H+G`, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics + distance') + ylab(' distance only')
Cairo(file=paste(figpath,"/p30.png",sep=''), 
      type="png",
      units="px", 
      width=1124, 
      height=320,
      pointsize=12, 
      dpi=72*2)
grid.arrange(p1,p2,p3, ncol=3, top='Spearman coefficients of #expaned and error ratio')
dev.off()


########
## P31 : Scatter plots of Spearman corellations between expanded nodes and RT
##########
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True' & !(paths$instance %in% c('22','23','24')))
paths$optimal_length<-paths$optimal_length+1
rid=read.csv('rid.csv', header = TRUE, sep=',',stringsAsFactors=F)
rid$instance=factor(rid$puzzle, levels = lvls_e, labels = 1:length(lvls_e))
d=ddply(rid, .(instance,alg), function(x){return(mean(x$expanded))})
names(d)<-c('instance','alg','expanded')
d1=merge(d,paths,by = c('instance'))
d1$error_ratio=d1$human_length/d1$optimal_length
d1$error_n=d1$human_length-d1$optimal_length
s<-ddply(d1, .(subject, alg), function(x){return(cor(x$expanded, x$rt, method = 'spearman'))} )
names(s)<-c('subject','alg','spearman')
dc<-dcast(s, subject ~ alg)
names(dc)<-c('subject', 'H','H+G','G')
p1<-ggplot(dc, aes(x=dc$H, y=dc$`H+G`)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ 
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' heuristics + distance')
p2<-ggplot(dc, aes(x=dc$H, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' distance only')
p3<-ggplot(dc, aes(x=dc$`H+G`, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics + distance') + ylab(' distance only')
Cairo(file=paste(figpath,"/p31.png",sep=''), 
      type="png",
      units="px", 
      width=1124, 
      height=320,
      pointsize=12, 
      dpi=72*2)
grid.arrange(p1,p2,p3, ncol=3, top='Spearman coefficients of #expaned and Response times')
dev.off()



########
## P32 : Scatter plots of Spearman corellations between Solution lengths
##########
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
paths=subset(paths,paths$complete=='True' & !(paths$instance %in% c('22','23','24')))
paths$optimal_length<-paths$optimal_length+1
rid=read.csv('rid.csv', header = TRUE, sep=',',stringsAsFactors=F)
rid$instance=factor(rid$puzzle, levels = lvls_e, labels = 1:length(lvls_e))
d=ddply(rid, .(instance,alg), function(x){return(mean(x$path))})
names(d)<-c('instance','alg','path_length')
d1=merge(d,paths,by = c('instance'))
d1$error_ratio=d1$human_length/d1$optimal_length
d1$error_n=d1$human_length-d1$optimal_length
s<-ddply(d1, .(subject, alg), function(x){return(cor(x$path_length, x$human_length, method = 'spearman'))} )
names(s)<-c('subject','alg','spearman')
dc<-dcast(s, subject ~ alg)
names(dc)<-c('subject', 'H','H+G','G')
p1<-ggplot(dc, aes(x=dc$H, y=dc$`H+G`)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ 
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' heuristics + distance')
p2<-ggplot(dc, aes(x=dc$H, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics only') + ylab(' distance only')
p3<-ggplot(dc, aes(x=dc$`H+G`, y=dc$G)) + geom_point() + xlim(-1,1) + ylim(-1,1)+ geom_abline(slope = 1, intercept = 0)+
  geom_abline(slope = 1, intercept = 0) + xlab('heuristics + distance') + ylab(' distance only')
Cairo(file=paste(figpath,"/p31.png",sep=''), 
      type="png",
      units="px", 
      width=1124, 
      height=320,
      pointsize=12, 
      dpi=72*2)
grid.arrange(p1,p2,p3, ncol=3, top='Spearman coefficients of Solution lengths')
dev.off()



####
# Plot 26.X: bar plot of expaned nodes 
#####
setwd("~/gdrivezb9/rushhour/src/")
d=read.csv('data_456', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(d, aes(x=puzzle, y=expanded)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=generated)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=close)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=open)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)

########
## Plot 27.X: bar plot of expaned nodes (pure heuristics) 
###########
d=read.csv('data_456_pureh', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(d, aes(x=puzzle, y=expanded)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=generated)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=close)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=open)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
#solution length
ggplot(d, aes(x=puzzle, y=length)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)


###########
## No heuristics. 
########
## Plot 28.X: bar plot of expaned nodes (no heuristics) 
###########

###########
d=read.csv('data_456_0', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(d, aes(x=puzzle, y=expanded)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=generated)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=close)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)
ggplot(d, aes(x=puzzle, y=open)) + stat_summary(geom='bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2)




########
# TMP plot
#########
setwd("~/gdrivezb9/rushhour/results/pilot")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, complete=='True' & paths$instance %in% c('Jam-4','Jam-5','Jam-6'))
paths$instance=factor(paths$instance, levels = lvls_e, labels = 1:length(lvls_e))
g<-ggplot(paths, aes(x = paths$instance)) + 
  stat_summary(geom='bar', aes(y=paths$human_length), fun.y = 'mean', fill='wheat4')+
  stat_summary(geom='bar', aes(y=paths$optimal_length+1), fun.y = 'mean', fill='black')+
  stat_summary(geom='errorbar', aes(y=paths$human_length), fun.data = mean_sem, width=0.4)+
  xlab('puzzle') + ylab('#moves') + theme(text = element_text(size=20))
g
