library(ggplot2)
library(plyr)
library(gridExtra)
library(Cairo)
library(reshape2)


#####################################
# response: distribution of errors across steps in the plan 
#           #consecutive errors
#           distribution of negative/positive/neutral
#           surrenders/restarts (when)
#           RT
#           Backtracks (of size)
# stimuli: 
#         v_size
#         num_sccs
#         max_scc_size
#         mag nodes
#
# CCN:   1) RT per move
#        2) real-distance on surrender/restart
#        3) Burst behavior
# Others: 
#        1) Model for error(rate) in relation to v_size,num_sccs,max_scc_size,mag_nodes,mag_edges
#        2) number of unsafe-moves made, and/or.
#####################################



######################################################## 
# Data analysis
# - General setup
#######################################################
setwd("~/gdrivezb9/rushhour/results/all_stages/")
figpath<-'../../paper/figures'
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
#sort by experiment design
lvls_e  = unique(paths[order(nchar(paths$instance),paths$instance),c('instance')])
#sort by solution length
lvls_sl = unique(paths[order(paths$optimal_length),c('instance')])
lvls_how_many =  names(sort(table(paths$instance)))
mean_sem <- function(x) {
  meanx=mean(x)
  sem=sd(x)/sqrt(length(x))
  return(data.frame("y"=meanx, "ymin"=meanx-sem/2, "ymax"=meanx+sem/2))
}

mean_sem_ <- function(x) {
  meanx=mean(x)
  sem=sd(x)/sqrt(length(x))
  return(data.frame("mean"=meanx, "sem"=sem/2))
}


##############################
# Describing collected data
##############################

###############################
## Plot 1: Bar Plot of instances moves by humans vs optimal.
#################################
setwd("~/gdrivezb9/rushhour/results/all_stages")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, complete=='True')
#paths=subset(paths, subject=='A191V7PT3DQKDP:3PMBY0YE28B43VMUKKJ6EDF85RV9C8')
paths$instance=factor(paths$instance, levels = lvls_sl, labels = 1:length(lvls_e))
g<-ggplot(paths, aes(x = paths$instance)) + 
  stat_summary(geom='bar', aes(y=paths$human_length), fun.y = 'mean', fill='wheat4')+
  stat_summary(geom='bar', aes(y=paths$optimal_length+1), fun.y = 'mean', fill='black')+
  stat_summary(geom='errorbar', aes(y=paths$human_length), fun.data = mean_sem, width=0.4)+
  xlab('puzzle') + ylab('#moves') + theme(text = element_text(size=20))
g
Cairo(file=paste(figpath,"/p1.png",sep=''), 
      type="png",
      units="px", 
      width=1124, 
      height=320,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()

##########
# Plot 2: Bar plot of restrats/skipped/solved instance Status
##########
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$status<-factor(with(paths, ifelse(paths$skipped,'Surrender',ifelse(paths$complete,ifelse(paths$trial_number=='0','Solved','Restart'),'NA')))
                     ,levels = rev(c('Solved','Restart','Surrender')))
paths=subset(paths,status!= 'NA' & instance !='Jam-25')
paths$instance=factor(paths$instance, levels = lvls_sl,labels = 1:length(lvls_sl))
g<-ggplot(paths, aes(x=paths$instance, na.rm=T)) + geom_bar(aes(fill=status)) + xlab('puzzle') + ylab('#subejcts')+
  scale_y_continuous(breaks=1:length(unique(paths$subject)))+ 
  scale_fill_manual(values = c('darkred','darkorange','darkgreen'))+
  theme(legend.position = c(0.8,0.7), text = element_text(size=20))+
  theme(legend.text = element_text(size=14), legend.key.height = unit(0.4,'cm'),legend.key.width = unit(0.4,'cm') )
g
Cairo(file=paste(figpath,"/p2.png",sep=''), 
      type="png",
      units="px", 
      width=1124, 
      height=420,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()


####################
# Plot 3: Scatter plot of response-time as move number, by subject
# per path
####################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, moves$move_number<150)
ggplot(moves, aes(x=moves$move_number)) + geom_point(aes(y=moves$rt, color=subject)) +
  scale_color_manual(values=1:length(unique(moves$subject)), guide=FALSE)+
  ggtitle("Response Times") + xlab('move number') + ylab('Seconds') + ylim(0,90)



###################
# Plot 3.1: scatter plot of response-time as move number, by subject
###################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, moves$move_number<150)
g<-ggplot(moves, aes(x=moves$move_number)) + geom_point(aes(y=moves$rt)) +
   xlab('move number') + ylab('seconds') + ylim(0,90) +
  theme(text = element_text(size=18))
g
Cairo(file=paste(figpath,"/p3_1.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()

#######################################################################################
#  Plot 3.2: Bar plot showing the average Spearman correlations of
#  RT with and move# and distance to goal
#   !!!Requires running distance to goal!!
######################################################################################
require(plyr)

cor_move_rt <- function(xx)
{
  return(data.frame(COR = cor(xx$move_number, xx$rt, method = "spearman")))
}
cor_dist_rt <- function(xx)
{
  return(data.frame(COR = cor(xx$distance_to_goal, xx$rt, method = "spearman")))
}
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=ddply(moves, .(subject), cor_dist_rt)
d$type='distance to goal'
d1=ddply(moves, .(subject), cor_move_rt)
d1$type='move#'
d2=rbind(d,d1)
cors=ddply(d2,.(type), function(x){return(mean_sem(x$COR))})
cors$sem=(cors$ymax-cors$ymin)/2
ggplot(d2, aes(x=type, y=COR)) + stat_summary(geom = 'bar', fun.data = mean_sem) +
  stat_summary(geom = "errorbar", fun.data = mean_sem, width=0.22) 



##########################
# Plot 3.3: Line plot of mean Response time vs move number per subject
#########################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=aggregate(moves,by = list("subject"=moves$subject,"move_num"=moves$move_number), FUN = mean)
d=subset(d,d$move_num<=150)
g<-ggplot(d, aes(x=d$move_number,y=d$rt)) + stat_summary(fun.y = "mean", geom = "point", size=1) +
  stat_summary(fun.y = "mean", geom = "line" ) + 
  stat_summary(fun.data = mean_sem, geom = "errorbar" , color="blue", size=1, width=2) + 
  ggtitle('mean response time across subjects') + xlab('move#') + ylab('rt')
g



##################################################
# Plot 4: Scatter plot of Human solution length, vs Optimal solution length 
# aggregated by instance.
######################################
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=subset(paths, paths$complete=='True')
dm=aggregate(d, by = list('instance'=d$instance), FUN = mean)
ds=aggregate(d, by = list(d$instance), FUN = sd)
dl=aggregate(d, by = list(d$instance), FUN = length)
d=dm[,c('instance','optimal_length','human_length')]
d$sem=ds$human_length/sqrt(dl$human_length)
d$instance = factor(d$instance, levels = lvls_sl)
g<-ggplot(d, aes(x=d$optimal_length , y=d$human_length))+ geom_point(stroke=2) + 
  geom_errorbar(aes(ymin=d$human_length-(d$sem/2),ymax=d$human_length+(d$sem/2))) + 
  xlab('Optimal solution') + ylab('Human solution')
g


######################
##  Burst analysis 
######################

do_burst<-function(x){
  return(burst(as.character(x$level)))
}

burst<-function(x){
  return(diff(which(c('H',x,'H')=='H')))
}


#sampling
do_rand_burst<-function(x,f,sample_size){
  res=c()
  l=as.character(x$level)
  for(i in 1:sample_size){
    s=sample(l)
    res=c(res,f(burst(s)))
  }
  return(res)
}

burst_analysis<-function(sp,f,sample_size){
  moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
  moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
  d=ddply(moves, .(path), function(x){return(sp(x$rt))})
  names(d)<-c('path','rt_med')
  moves=merge(moves,d)
  moves$level = with(moves,ifelse(rt>rt_med,'H','L'))
  ab=ddply(moves, .(path), function(x){return(f(do_burst(x)))})
  names(ab)<-c('path','ab') #actual variance in burst size
  rb=ddply(moves, .(path), function(x){return(mean(do_rand_burst(x,f,sample_size)))}) #random variance in burst size
  names(rb)<-c('path','rb')
  k=merge(ab,rb)
  k$diff=k$rb-k$ab
  k$ratio=k$ab/k$rb
  return(k)
}

spearman_corr<-function(x){
  return(cor(x,1:length(x), method = 'spearman'))
}
##############################################################################
# Plot 3.4: line plot of response-time as move number, by subject per path
##############################################################################
do_plot<-function(moves) {
  g<-ggplot(moves, aes(x=moves$move_number)) + geom_line(aes(y=moves$rt)) +
  scale_color_manual(values=1:11, guide=FALSE) + 
    theme(axis.line=element_blank(),
          axis.text.x=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks=element_blank(),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          legend.position="none",
          #panel.background=element_blank(),
          #panel.border=element_blank(),
          #panel.grid.major=element_blank(),
          panel.grid.minor=element_blank()
          #plot.background=element_blank()
          )
  return(g)
}
k<-burst_analysis(function(x){return(mean(x))},spearman_corr,1000)
hbursts=head(k[order(k$diff,decreasing = T),c('path')], 100)
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
#moves=subset(moves, moves$path in hbursts)
moves_sp=split(moves, moves$path)
moves_sp=tail(moves_sp, n = 100)
slplots<-lapply(moves_sp,do_plot)
do.call('grid.arrange',c(slplots, ncol = 5))

Cairo(file=paste(figpath,"/p3_4.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*2)
do.call('grid.arrange',c(slplots, ncol = 4))
dev.off()


#############################################################
# Plot 5: Bar plot of number of (good, neutral, wrong) moves 
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
# scale_fill_manual(values = c('darkred','darkorange','darkgreen'))
##################################################################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'neutral',ifelse(progress>0,'positive','negative')))
moves<-subset(moves, category!='NA')
g<-ggplot(moves,aes(x=moves$category)) + geom_bar(aes(fill=moves$category)) + xlab('move')  +guides(fill=FALSE)
g
Cairo(file=paste(figpath,"/p5.png",sep=''), 
      type="png",
      units="px", 
      width=500, 
      height=350,
      pointsize=12*4, 
      dpi=72*3.7)
g
dev.off()

#########################################################
# Plot 6: Line plot of progress of every solution path
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
#########################################################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
plot6<-function(moves){
  title=strsplit(moves$instance[[2]],'-')[[1]][2]
  g<-ggplot(moves,aes(x=moves$move_number)) + geom_line(aes(color=paste(moves$subject, moves$trial_number),y=moves$distance_to_goal), size=0.5) +
    ggtitle(paste('puzzle ',title)) +xlab('move number') + ylab('distance from goal') +
    guides(color=guide_legend(title="path"))+theme(legend.position="none", axis.title.x=element_blank(),axis.title.y=element_blank())
  return(g)
}
moves_ins = split(moves, moves$instance)
slplots=lapply(moves_ins, plot6)
#slplots<-slplots[order(nchar(names(slplots)),names(slplots))]
slplots<-slplots[order(match(names(slplots),lvls_sl), names(slplots))]
#do.call('grid.arrange',c(slplots, ncol = 4, top = "Solution Progress"))
do.call('grid.arrange',c(slplots, ncol = 4))


####
# Plot 6.1: Line plot of progress of some solution path
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, moves$instance %in% c('Jam-8', 'Jam-1', 'Jam-10','Jam-11','Jam-13','Jam-16'))
plot6<-function(moves){
  title=strsplit(moves$instance[[1]],'-')[[1]][2]
  g<-ggplot(moves,aes(x=moves$move_number)) + geom_line(aes(color=paste(moves$subject, moves$trial_number),y=moves$distance_to_goal), size=0.5) +
    ggtitle(paste('Puzzle-',title,sep = '')) +xlab('move#') + ylab('distance from goal') +
    guides(color=guide_legend(title="path"))+theme(legend.position="none", text = element_text(size = 12))+
    theme(axis.title = element_blank(), title = element_text(size=10))
  return(g)
}
moves_ins = split(moves, moves$instance)
moves_ins=moves_ins[c(1,6,3,2,4,5)]
slplots=lapply(moves_ins, plot6)
Cairo(file=paste(figpath,"/p6_1.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*2)
do.call('grid.arrange',c(slplots, ncol = 3))
dev.off()




######################################################
# Plot 6.2: Line plot of progress of restarts
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
########################################################
plot6<-function(moves){
  g<-ggplot(moves,aes(x=moves$move_number)) + geom_line(aes(color=paste(moves$subject, moves$trial_number, moves$instance),y=moves$distance_to_goal), size=1) +
    xlab('move#') + ylab('distance from goal') +
    guides(color=guide_legend(title="path"))+theme(legend.position="none", text = element_text(size = 12))
  return(g)
}
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$success<-with(paths, ifelse(paths$skipped,'skipped',ifelse(paths$complete,ifelse(paths$trial_number=='0','solved','restarted'),'NA')))
#restart
p=subset(paths,success=='restarted')
ps=paste(p$subject,p$instance,p$trial_number,sep='_')
moves=subset(moves, moves$path %in% ps)
moves_ins = split(moves, moves$instance)
d=data.frame('puzzle'=names(moves_ins), 'L'=sapply(moves_ins, nrow))
puz=subset(d, L>40)$puzzle
moves=subset(moves, moves$instance %in% puz)
moves_ins = split(moves, moves$instance)
slplots=lapply(moves_ins, plot6)
do.call('grid.arrange',c(slplots, ncol = 2))
plot6(moves)


####
# Plot 6.3: Line plot of progress of surrenders
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
####
plot6<-function(moves){
  g<-ggplot(moves,aes(x=moves$move_number)) + 
    geom_line(aes(color=paste(moves$subject, moves$trial_number, moves$instance),y=moves$distance_to_goal), size=1) +
    xlab('move#') + ylab('distance from goal') +
    guides(color=guide_legend(title="path"))+theme(legend.position="none", text = element_text(size = 12))
  return(g)
}
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$success<-with(paths, ifelse(paths$skipped,'skipped',ifelse(paths$complete,ifelse(paths$trial_number=='0','solved','restarted'),'NA')))
#restart
p=subset(paths,success=='skipped')
ps=paste(p$subject,p$instance,p$trial_number,sep='_')
moves=subset(moves, moves$path %in% ps)
plot6(moves)



###########################  
# Plot 8: Bar plot of #Moves Category Across Subjects
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
########################  
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'neutral',ifelse(progress>0,'positive','negative')))
d=aggregate(moves,by = list("category"=moves$category,"subject"=moves$subject), FUN = "length")
g<-ggplot(d,aes(x=d$category, y=d$rt)) + stat_summary( fun.y="mean",aes(fill=category), geom="bar") + xlab('move')+
  ylab("Count") + stat_summary(geom = "errorbar", position = "dodge", width=0.2, size=1,fun.data = mean_sem)+guides(fill=F)
g
Cairo(file=paste(figpath,"/p8.png",sep=''), 
      type="png",
      units="px", 
      width=500, 
      height=350,
      pointsize=12*4, 
      dpi=72*3.2)
g
dev.off()


#######################################################################  
# Plot 8.1: Bar plot of mean rt Category Across Subjects
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
#########################################################################  
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
d=moves
#d=aggregate(moves,by = list("category"=moves$category,"subject"=moves$subject), FUN = "mean")
ggplot(d,aes(x=d$category, y=d$rt)) + stat_summary( fun.y="mean", geom="bar") + xlab('Move category')+
  ylab("Seconds") + stat_summary(geom = "errorbar", position = "dodge", width=0.2, size=1,fun.data = mean_sem)+
  theme(text = element_text(size=18))


###########################################  
# Plot 8.2: Bar plot of Average Mistakes per category, per instance  Across Subjects
# !!!!!!!!!UNAVIALABLE BECAUSE OF THE TRUE DISTANCE ANALYSIS!!!!!!!
########################################  
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Same',ifelse(progress>0,'Closer','Further')))
d=aggregate(moves,by = list("category"=moves$category,"subject"=moves$subject), FUN = "length")
ggplot(d,aes(x=d$instance, y=d$rt)) + stat_summary( fun.y="mean", geom="bar") + xlab('Moving closer/same/further from solution')+
  ylab("Count") + stat_summary(geom = "errorbar", position = "dodge", width=0.2, size=1,fun.data = mean_sem)




# What are the states that people took more time?
############################################################
# Plot 3.4: Line plot of mean response-time by move number
############################################################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves,moves$move_number<=100)
#msplit=split(moves,f = moves$instance)

ggplot(moves, aes(x=moves$move_number, y=moves$rt)) +
  stat_summary(fun.y = 'mean', geom='line', size=2)+
  stat_summary(fun.y = 'mean', geom='point', size=3)+
  scale_alpha_discrete(aggregate(moves, by = list('move_number'=moves$move_number), FUN=length)$rt)+
  stat_summary(fun.data = mean_sem, geom='errorbar')+
  stat_summary(fun.data = function(x){data.frame(y=mean(x), label=length(x))} , geom='text' ,vjust=-1.5, hjust=1 , color='red', size=4)+
  ggtitle("response time bursts ") + xlab('move number') + ylab('Seconds') +  facet_grid(~ instance)

####################
# Plot 3.5:  histogram of rt 
########################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, rt<60)
g<-ggplot(moves, aes(x=moves$rt)) + geom_histogram(bins = 100)  + xlab("Seconds")+
theme(axis.line=element_blank(),
      #axis.text.x=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks=element_blank(),
      #axis.title.x=element_blank(),
      axis.title.y=element_blank(),
      legend.position="none",
      #panel.background=element_blank(),
      #panel.border=element_blank(),
      #panel.grid.major=element_blank(),
      panel.grid.minor=element_blank()
      #plot.background=element_blank()
)
g
Cairo(file=paste(figpath,"/p3.5.png",sep=''), 
      type="png",
      units="px", 
      width=470*1.2, 
      height=300,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()




##############
# Plot 3.6:  histogram of rt of subjects 
#############
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
srt=ddply(moves, .(subject), function(x){return(mean(x$rt))})
moves$subject = factor(moves$subject, levels = srt[order(srt$V1),c('subject')])
g<-ggplot(moves, aes(x=moves$subject, y=moves$rt)) + stat_summary(geom = 'bar', fun.y = 'mean') + 
  stat_summary(geom='errorbar', fun.data = mean_sem, width=0.2) + xlab('subject') + ylab('msec')+
theme(axis.line=element_blank(),
      axis.text.x=element_blank(),
      #axis.text.y=element_blank(),
      axis.ticks=element_blank(),
      #axis.title.x=element_blank(),
      #axis.title.y=element_blank(),
      legend.position="none",
      #panel.background=element_blank(),
      #panel.border=element_blank(),
      #panel.grid.major=element_blank(),
      panel.grid.minor=element_blank()
      #plot.background=element_blank()
)
g
Cairo(file=paste(figpath,"/p13_1.png",sep=''), 
      type="png",
      units="px", 
      width=470*1.2, 
      height=300,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()


#####
# Plot 3.7:  histogram of rt  (median normalized)
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
med_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=median)[,c('subject','rt')]
sd_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=sd)[,c('subject','rt')]
names(med_rt)<-c('subject','med_rt')
names(sd_rt)<-c('subject','sd_rt')
d=merge(moves,med_rt)
d=merge(d,sd_rt)
d$norm_rt = (d$rt - d$med_rt)/d$sd_rt
ggplot(d, aes(x=d$norm_rt)) + geom_histogram(binwidth = 0.01) + ggtitle("Response times (normalized by median)") + xlab("Seconds")


#####
# Plot 15:  histogram of rt  (median normalized - zoom [-1 - 1])
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves= subset(moves, moves$rt < 90)
med_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=median)[,c('subject','rt')]
sd_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=sd)[,c('subject','rt')]
names(med_rt)<-c('subject','med_rt')
names(sd_rt)<-c('subject','sd_rt')
d=merge(moves,med_rt)
d=merge(d,sd_rt)
d$norm_rt = (d$rt - d$med_rt)/d$sd_rt
d=subset(d, d$norm_rt < 1 )
ggplot(d, aes(x=d$norm_rt)) + geom_histogram(binwidth = 0.01) + ggtitle("Response times (normalized by median zoom [-1,1]) ") + xlab("Seconds")



#############
# Plot 16: RT histograms per move category (good, neutral, wrong), trucncated (rt<15), with medians.
#  !!!! Requires true distance !!!!!!!!!!
#############
moves=read.csv('moves.1.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
moves= subset(moves, moves$rt < 15)
my_colors= c('blue','green','red')
ggplot() + 
  geom_density(data = subset(moves,category=='Negative'), aes(x=rt,fill='Negative'), alpha=0.3) + 
  geom_density(data = subset(moves,category=='Neutral'), aes(x=rt,fill='Neutral'), alpha=0.4) + 
  geom_density(data = subset(moves,category=='Positive'), aes(x=rt,fill="Positive"), alpha=0.2) + 
  scale_fill_manual(name="Move category", values=my_colors)+
  geom_density(data = subset(moves,category=='Negative'), aes(x=rt,fill='Negative'), alpha=0.0, size=0.5, color=my_colors[1], show.legend = F) + 
  geom_density(data = subset(moves,category=='Neutral'), aes(x=rt,fill='Neutral'), alpha=0.0, size=0.5, color=my_colors[2], show.legend = F) + 
  geom_density(data = subset(moves,category=='Positive'), aes(x=rt,fill="Positive"), alpha=0.0, size=0.5, color=my_colors[3], show.legend = F) + 
  
  #add means
  geom_vline(xintercept=mean(subset(moves,category=='Negative')$rt), color=my_colors[1], size=1, linetype=2)+
  geom_vline(xintercept=mean(subset(moves,category=='Neutral')$rt), color=my_colors[2], size=1, linetype=2)+
  geom_vline(xintercept=mean(subset(moves,category=='Positive')$rt), color=my_colors[3], size=1, linetype=2)+
    theme(legend.position=c(.8, .6), legend.title = element_text(size=12),legend.text = element_text(size=10), text = element_text(size=18))+
  guides(color='blue')+
  xlab("Seconds")


#####################################
# Plot 16.1: RT ECDF per move category (good, neutral, wrong) 
# !!! Requires True Distance !!!! 
####################################
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
sub='andra.log'
moves= subset(moves, moves$progress !='NA' & subject==sub)
moves=arrange(moves, subject, category, rt)
d=ddply(moves, .(subject, category), transform, ecdf=ecdf(rt)(rt))
mr=d[which(d$rt == max(d$rt)),]
d<-rbind(d,mr)
d[nrow(d),12]<-'Positive'
d<-rbind(d,mr)
d[nrow(d),12]<-'Negative'
d<-rbind(d,mr)
d[nrow(d),12]<-'Neutral'

g<-ggplot(d, aes(y=ecdf, x=rt)) + geom_line(aes(color=category), size=1) +
  xlab('response time') + ylab('accum. probability')+
  theme(legend.position=c(.6, .5),
        legend.title = element_text(size=12),
        legend.text = element_text(size=10), 
        text = element_text(size=18), 
        legend.key.height = unit(0.2,'cm'))
g
Cairo(file=paste(figpath,"/p16_1.png",sep=''), 
      type="png",
      units="px", 
      width=470*1.2, 
      height=400,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()

####
# Plot 16.2: RT histograms per move category (good, neutral, wrong), trucncated (rt<15), with medians. 
# !!! Requires True Distance !!!! 
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
moves= subset(moves, category!='NA')
moves= subset(moves, moves$rt < 15)
bw=0.5
alp=1
sz=1
g<-ggplot() + 
  geom_freqpoly(data = moves, aes(x=rt,group=interaction(subject,category), color=category), binwidth=bw, size=sz, alpha=alp) +
  theme(legend.position=c(.8, .6), text = element_text(size=18))+
  theme(legend.title = element_text(size=12), legend.text = element_text(size=12),legend.key.height = unit(0.5,'cm'))+
  xlab("Seconds")
g
Cairo(file=paste(figpath,"/p16_2.png",sep=''), 
      type="png",
      units="px", 
      width=570, 
      height=300,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()


####
# Plot 16.3: bar plot of RT per move category (good, neutral, wrong) 
# !!! Requires True Distance !!!!
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
moves= subset(moves, category!='NA')
d=ddply(moves, .(subject, category), function(x){return(mean_sem(x$rt))})
g<-ggplot(d, aes(x=d$category,y=d$y)) +  stat_summary(geom='bar', fun.y = 'mean')  + 
  stat_summary(geom='errorbar', fun.data = mean_sem, size=2, width=0.4)+  
  theme(text = element_text(size=18))+ xlab("category") + ylab("Rt(sec)") 
g
Cairo(file=paste(figpath,"/p16_3.png",sep=''), 
      type="png",
      units="px", 
      width=470*1.4, 
      height=400,
      pointsize=12*2, 
      dpi=72*2.5)
g
dev.off()


####
# Plot 17: Scatter plot of RT average across subjects, per instance, in relation to distance from goal.
# !!! Requires True Distance !!!! 

moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, moves$rt<90)
ggplot(moves, aes(x=moves$distance_to_goal,y= moves$rt, group=instance)) + 
  stat_summary(fun.y = "mean", geom="point") + stat_summary(fun.data = mean_sem, geom="errorbar")


####
# Plot 18: Scatter plot of RT average across subjects, per instance, in relation to distance from goal.
# Normalized by subject median
# !!! Requires True Distance !!!! 
#
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves= subset(moves, moves$rt < 90)
med_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=median)[,c('subject','rt')]
sd_rt = aggregate(moves, by = list('subject'=moves$subject), FUN=sd)[,c('subject','rt')]
names(med_rt)<-c('subject','med_rt')
names(sd_rt)<-c('subject','sd_rt')
d=merge(moves,med_rt)
d=merge(d,sd_rt)
d$norm_rt = (d$rt - d$med_rt)/d$sd_rt
ggplot(d, aes(x=d$distance_to_goal,y= d$norm_rt, group=instance)) + 
  stat_summary(fun.y = "mean", geom="point") + stat_summary(fun.data = mean_sem, geom="errorbar") + 
  ggtitle("RT vs. distance from goal") + xlab("Distance from goal") + ylab("Seconds")

############
# Plot 19: Boxplot with Kruskal Wallis test on rt per category
# !!! Requires True Distance !!!! 
############
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=as.factor(with(moves, ifelse(progress==0, 'Same',ifelse(progress>0,'Closer','Further'))))
t=kruskal.test(rt ~ category, data = moves)
ggplot(data = moves, aes(x=category, y=rt)) + geom_boxplot() + 
  ggtitle('Kruskal-Wallis rank sum test - rt by category') + 
  annotate(geom = 'text',label='p-val= ~5.7e-10', x=2, y=95, size=9)

#####
# Plot 20: linear regression of log(rt) and distance from goal 
# !!! Requires True Distance !!!! 
#######
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(moves, aes(x=distance_to_goal, y=log(rt)))  + geom_point() + 
  geom_smooth(method = 'lm', se=FALSE) + 
  annotate(geom = "text", label="intercept=0.82144, coefficient=0.01508(***)", x = 10, y=4.5) +
  ggtitle('linear regression log(rt)~distance_to_goal')

summary(lm(formula = log(moves$rt) ~ moves$distance_to_goal))


#####
# Plot 21:   linear regression of rt and distance from goal
# !!! Requires True Distance !!!! 
####
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(moves, aes(x=distance_to_goal, y=rt))  + geom_point() + 
  geom_smooth(method = 'lm', se=FALSE) + 
  annotate(geom = "text", label="intercept=3.4022, coefficient=0.1142(***)", x = 10, y=90)+
  ggtitle('linear regression rt~distance_to_goal')

summary(lm(formula = moves$rt ~ moves$distance_to_goal))



######
# Plot 22:  linear regression of rt and move_number 
# !!! Requires True Distance !!!! 
#######
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
ggplot(moves, aes(x=move_number, y=rt))  + geom_point() + 
  geom_smooth(method = 'lm', se=FALSE) + 
  annotate(geom = "text", label="intercept=5.9195, coefficient=-0.0984(***)", x = 15, y=90)+
  ggtitle('linear regression rt~move_num')
summary(lm(formula = moves$rt ~ moves$move_number))

###
# Plot 23:  linear regression of rt and move_number (without the first move)
# !!! Requires True Distance !!!! 
######
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, move_number>0)
ggplot(moves, aes(x=move_number, y=rt))  + geom_point() + 
  geom_smooth(method = 'lm', se=FALSE) + 
  annotate(geom = "text", label="intercept=5.9195, coefficient=0.01", x = 15, y=90, color='red') +
  ggtitle('linear regression rt~move_num (first move excluded)')
summary(lm(formula = moves$rt ~ moves$move_number))



###
# Plot 24 : Scatter plot of response time, vs Optimal solution length aggregated by instance.
# !!! Requires True Distance !!!! 
###
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=subset(paths, paths$complete=='True')
dm=aggregate(d, by = list('instance'=d$instance), FUN = mean)
ds=aggregate(d, by = list(d$instance), FUN = sd)
dl=aggregate(d, by = list(d$instance), FUN = length)
d=dm[,c('instance','optimal_length','rt')]
d$sem=ds$rt/sqrt(dl$rt)
d$instance = factor(d$instance, levels = lvls_sl)
ggplot(d, aes(x=d$optimal_length , y=d$rt))+ geom_point(stroke=2,aes(color=instance)) + 
  geom_errorbar(aes(ymin=d$rt-(d$sem/2),ymax=d$rt+(d$sem/2))) + 
  ggtitle('RT per instance') + xlab('Optimal solution') + ylab('Seconds')


###
# Plot 25  : Scatter plot of response time, vs Distance from goal.
# !!! Requires True Distance !!!! 
###
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
g<-ggplot(moves, aes(x=distance_to_goal , y=rt))+ geom_point() + 
   xlab('Distance from goal') + ylab('Seconds') + theme(text = element_text(size=18))

Cairo(file=paste(figpath,"/p25.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*2)
g
dev.off()



# Stat 1: Wilcox.test per subject (non-standard)
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
moves=subset(moves, category!='NA')
func<-function(x){
  neg=subset(x,category=='Negative')$rt
  pos=subset(x,category=='Positive')$rt
  neu=subset(x,category=='Neutral')$rt
  return(wilcox.test(pos, neg, alternative = 'less',paired = FALSE)$p.value)
}
posneg=ddply(moves, .(subject), func)
names(posneg)<-c('subject','posneg')
func<-function(x){
  neg=subset(x,category=='Negative')$rt
  pos=subset(x,category=='Positive')$rt
  neu=subset(x,category=='Neutral')$rt
  return(wilcox.test(pos, neu, alternative = 'less',paired = FALSE)$p.value)
}
posneu=ddply(moves, .(subject), func)
names(posneu)<-c('subject','posneu')
func<-function(x){
  neg=subset(x,category=='Negative')$rt
  pos=subset(x,category=='Positive')$rt
  neu=subset(x,category=='Neutral')$rt
  return(wilcox.test(neg, neu, alternative = 'less',paired = FALSE)$p.value)
}
negneu=ddply(moves, .(subject), func)
names(negneu)<-c('subject','negneu')
d=merge(posneg,posneu,by='subject')
d=merge(d,negneu, by='subject')
  
###########
# Stat 3: Wilcox.test on every subject
###############
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'Neutral',ifelse(progress>0,'Positive','Negative')))
d=ddply(moves, .(subject, category), function(x){return(median(x$rt))})
sp=split(d,d$category)
wilcox.test(sp$Positive$V1, sp$Negative$V1, alternative = 'two.sided',paired = TRUE)$p.value
wilcox.test(sp$Positive$V1, sp$Neutral$V1, alternative = 'two.sided',paired = TRUE)$p.value
wilcox.test(sp$Negative$V1, sp$Neutral$V1, alternative = 'two.sided',paired = TRUE)$p.value




##########
#  Stat 4:  spearman correlation per subject rt~move#
########## 
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
d=ddply(moves, .(subject), function(x){return(cor(x$rt, x$move_number,method = 'spearman'))})
hist(d$V1,breaks = 30,main = 'spearman correlations \nof rt~move# (per subject)')
mean_sem_(d$V1)

##########
#  Stat 5:  spearman correlation per subject rt~distance#
# !!! Requires true distance !!! 
########## 
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves,distance_to_goal !='NA' & progress !='NA' )
d=ddply(moves, .(subject), function(x){return(cor(x$rt, x$distance_to_goal,method = 'spearman'))})
mean_sem_(d$V1)


##########
#  Stat 5.1:  Quadratic relationship of rt~distance#
# !!! Requires true distance !!! 
########## 
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves,distance_to_goal !='NA' & progress !='NA')

d=ddply(moves,.(subject), function(x){return(summary(lm(x$rt ~ poly(x$distance_to_goal, 2, raw = T)))$adj.r.squared)})
mean_sem_(d$V1)


##########
#  Stat 6:  mean completed puzzles. 
#  Subjects completed on average X ± x puzzles (minimum of X, maximum of Y).
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, complete=="True")
d=ddply(paths, .(subject), function(x){return(nrow(x))})
mean_sem_(d$V1)

##########
#  Stat 7:  Human vs optimal 
#  On average, subjects performed X\% more moves than the minimal solution.
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, complete=='True')
d=ddply(paths, .(subject,instance), function(x){return(x$human_length/(x$optimal_length+1))})
d1=ddply(d, .(subject), function(x){return(mean(x$V1))})
mean_sem_(d$V1)



##########
#  Stat 8:  Optimal human solutions 
#  In total there were XX optimal solutions out of YY solution paths that our subjects tried.
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$path<-paste(paths$subject,paths$instance,paths$trial_number,sep='_')
length(unique(paths$path))
nrow(subset(paths, complete=='True' & human_length==optimal_length+1))

paths=subset(paths, complete=='True')
d=ddply(paths, .(subject,instance), function(x){return(x$human_length/x$optimal_length)})
d1=ddply(d, .(subject), function(x){return(mean(x$V1))})
mean_sem_(d$V1)

##########
#  Stat 9:  Surrender/Restart
#  Out of the 11 subjects, X surrendered and only two surrendered more than once. 
#  On average subjects surrendered after three trials. 
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$success<-with(paths, ifelse(paths$skipped,'skipped',ifelse(paths$complete,ifelse(paths$trial_number=='0','solved','restarted'),'NA')))
# Total subjects that skipped 
length(unique(subset(paths, success == 'skipped')$subject))
length(unique(subset(paths, success == 'restarted')$subject))


##########
#  Stat 9.1:  Surrender decisions.
#  When are they surrendering 
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$success<-with(paths, ifelse(paths$skipped,'skipped',ifelse(paths$complete,ifelse(paths$trial_number=='0','solved','restarted'),'NA')))
paths$path<-paste(paths$subject,paths$instance,paths$trial_number,sep='_')
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
s =subset(paths, success == 'skipped')
sm=subset(moves, moves$path %in% s$path) 
#check the trial_number in moves. Something is weird there. 
mx=ddply(moves, .(path), function(x){return(max(x$move_number))})
names(mx)<-c('path','move_number')
mn=ddply(moves, .(path), function(x){return(min(x$distance_to_goal))})
names(mn)<-c('path','min_dist')
k<-merge(mx,sm)
k<-merge(k,mn)
k$t<-k$distance_to_goal-k$min_dist
mean_sem_(k$t)



##########
#  Stat 9.2:  Restart decisions.
#  When are they restarting 
########## 
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths$success<-with(paths, ifelse(paths$skipped,'skipped',ifelse(paths$complete,ifelse(paths$trial_number=='0','solved','restarted'),'NA')))
paths$path<-paste(paths$subject,paths$instance,paths$trial_number,sep='_')
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
s =subset(paths, success == 'restarted')
sm=subset(moves, moves$path %in% s$path) 
#check the trial_number in moves. Something is weird there. 
mx=ddply(moves, .(path), function(x){return(max(x$move_number))})
names(mx)<-c('path','move_number')
mn=ddply(moves, .(path), function(x){return(min(x$distance_to_goal))})
names(mn)<-c('path','min_dist')
k<-merge(mx,sm)
k<-merge(k,mn)
k$t<-k$distance_to_goal-k$min_dist
mean_sem_(k$t)




##########
#  Stat 9:  First move RT
########## 
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
d=ddply(moves, .(path), function(x){
  first=x[x$move_number==0,c('rt')]
  m = mean(x[x$move_number!=0,c('rt')])
  return(first/m)
})
x=d$V1
x=x[!is.na(x)]
mean_sem_(x)

############################
#  Stat 10: Move categories
############################ 
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves$category=with(moves, ifelse(progress==0, 'neutral',ifelse(progress>0,'positive','negative')))
table(moves$category)
d=ddply(moves, .(subject, category), function(x){return (nrow((x)))})
ddply(d, .(category), function(x){return (mean_sem_(x$V1))})

d=ddply(moves, .(subject), function(x){return (data.frame(table(x$category)))})
d1=ddply(d, .(Var1), function(x){return (mean_sem_(x$Freq))})



#Stat 11.1, Ratio of median split of SD
k=burst_analysis(function(x){return(median(x))},function(x){return(sd(x))},10000)
mean_sem(k$diff[!is.na(k$diff)])
mean_sem_(k$ratio[!is.na(k$ratio)])






#Stat 11.2, Spearman correlation
burst_analysis<-function(sp,f,sample_size){
  moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
  moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
  d=ddply(moves, .(path), function(x){return(sp(x$rt))})
  names(d)<-c('path','rt_med')
  moves=merge(moves,d)
  moves$level = with(moves,ifelse(rt>rt_med,'H','L'))
  ab=ddply(moves, .(path), function(x){return(f(do_burst(x)))})
  names(ab)<-c('path','ab') #actual variance in burst size
  rb=ddply(moves, .(path), function(x){return(mean(do_rand_burst(x,f,sample_size)))}) #random variance in burst size
  names(rb)<-c('path','rb')
  k=merge(ab,rb)
  k$diff=k$rb-k$ab
  k$ratio=k$ab/k$rb
  return(k)
}

k=burst_analysis(function(x){return(median(x))},spearman_corr,1000)
mean_sem(k$rb[!is.na(k$rb)])
mean_sem_(k$ab[!is.na(k$ab)])

g<-ggplot(k,aes(k$ab)) + geom_histogram(binwidth = .1) + xlab('spearman') + 
    #geom_vline(xintercept = 0, linetype=4, size=1.3)+
  theme(#axis.line=element_blank(),
    #axis.text.x=element_blank(),
    axis.text.y=element_blank(),
    axis.ticks=element_blank(),
    #axis.title.x=element_blank(),
    axis.title.y=element_blank(),
    legend.position="none",
    #panel.background=element_blank(),
    #panel.border=element_blank(),
    panel.grid.major=element_blank(),
    panel.grid.minor=element_blank()
    #plot.background=element_blank()
  )
g
Cairo(file=paste(figpath,"/stat11_2.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*4)
g
dev.off()

#stat 11.3 sd CDF histogram 
burst_analysis<-function(sp,f,sample_size){
  moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
  moves$path<-paste(moves$subject,moves$instance,moves$trial_number,sep='_')
  d=ddply(moves, .(path), function(x){return(sp(x$rt))})
  names(d)<-c('path','rt_med')
  moves=merge(moves,d)
  moves$level = with(moves,ifelse(rt>rt_med,'H','L'))
  ab=ddply(moves, .(path), function(x){return(f(do_burst(x)))})
  names(ab)<-c('path','ab') #actual variance in burst size
  rb=ddply(moves, .(path), function(x){
    actual=f(do_burst(x))
    rnds=do_rand_burst(x,f,sample_size)
    return(length(which(rnds<=actual))/sample_size)}) #random
  names(rb)<-c('path','rb')
  k=merge(ab,rb)
  k$diff=k$rb-k$ab
  k$ratio=k$ab/k$rb
  return(k)
}

k=burst_analysis(function(x){return(median(x))},sd,1000)
t=ks.test(k$rb, 'punif',0,1)
g<-ggplot(k,aes(k$rb)) + geom_histogram(binwidth = .05) + xlab('accumulated probability of sd') + 
  #geom_vline(xintercept = 0.5, linetype=4, size=1.3)+
  theme(#axis.line=element_blank(),
      #axis.text.x=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks=element_blank(),
      #axis.title.x=element_blank(),
      axis.title.y=element_blank(),
      legend.position="none",
      #panel.background=element_blank(),
      #panel.border=element_blank(),
      panel.grid.major=element_blank(),
      panel.grid.minor=element_blank()
      #plot.background=element_blank()
)
g
Cairo(file=paste(figpath,"/stat11.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=350,
      pointsize=12*2, 
      dpi=72*4)
g
dev.off()

# time_dist plot per subject
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves,moves$rt<10)
g<-ggplot(moves, aes(x=moves$rt)) + geom_density(aes(color=moves$subject)) + guides(color=FALSE) +
  theme(#axis.line=element_blank(),
  #axis.text.x=element_blank(),
  #axis.text.y=element_blank(),
  #axis.ticks=element_blank(),
  axis.title.x=element_blank(),
  axis.title.y=element_blank(),
  legend.position="none"
  #panel.background=element_blank(),
  #panel.border=element_blank()
  #panel.grid.major=element_blank(),
  #panel.grid.minor=element_blank()
  #plot.background=element_blank()
)
g
Cairo(file=paste(figpath,"/time_dist.png",sep=''), 
      type="png",
      units="px", 
      width=770, 
      height=150,
      pointsize=12*2, 
      dpi=72*3)
g
dev.off()

#bk=boot(k$diff, statistic = function(x, indices){return(mean(x[indices]))}, R=100000)
#plot(bk)
#ggplot(k, aes(x=diff)) + geom_histogram()



#library(combinat)
#exshustive (slow)
#rand_burst<-function(x){
#  l=as.character(x$level)
#  pp=unique(permn(l))
#  for (p in pp){
#    h=which(c('H',p,'H')=='H')
#    res=c(res,mean(diff(h)))
#  }
#  return(mean(res))
#}


####
# Trying to say something about the distribution of RT
####
library(mclust)
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
#m=Mclust(moves$rt)
m=densityMclust(moves[,c('rt','move_number')])
summary(m)
plot(m) 


#Do pearson
# Also, (as always), I would suggest to compute the correlation by subject, then report mean and SEM across subjects, since if different subjects are in different regions of the plot, this could easily dilute the correlation.

# Burst detection - "Suprise" measurement - Burst in FR is an anomaly in time-series. 
# This should be done by  window-based time-series analysis methods: 
# [10] V. Ganti, J. Gehrke, and R. Ramakrishnan. Demon: Data evolution and monitoring. In Proceedings of the 16th International Conference on Data Engineering, San Diego, California, 2000.
#[11] J. Gehrke, F. Korn, and D. Srivastava. On computing correlated aggregates over continual data streams. In Proc. ACM SIGMOD International Conf. on Management of Data, 2001.
#[12] A. C. Gilbert, Y. Kotidis, S. Muthukrishnan, and M. Strauss. Surfing wavelets on streams: One-pass summaries for approximate aggregate queries. In VLDB 2001, pages 79–88. Morgan Kaufmann, 2001.
# We want to detect a reoccuring pattern in the subjects rt which is conditioned on the stimuli.
# Dynamic time warping measures the match between two time-series.
# Clustering is also possible.
library(dtw)

moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, instance=='Jam-6')
ggplot(moves, aes(x=moves$move_number)) + geom_line(aes(y=moves$rt, color=paste(subject,instance,trial_number)), show_guide=F) +
  ggtitle("response time bursts - raw") + xlab('move number') + ylab('Seconds') + ylim(0,90)
query=subset(moves, subject=='gianni.log')$rt
reference=subset(moves, subject=='weiji.log')$rt
d=dtw(query,reference, step=asymmetric,keep.internals = TRUE)

plot(d, type = "twoway")
plot(d, type = "threeway")

#split by solution-paths, heatmap of distances.
moves=read.csv('moves.csv', header = TRUE, sep=',',stringsAsFactors=F)
moves=subset(moves, instance=='Jam-6')
moves$path=paste(moves$subject,moves$instance,moves$trial_number,sep = '')
dtw_dist<-function(path1,path2){
  query=subset(moves, path==path1)$rt
  reference=subset(moves, path==path2)$rt
  print(path1)
  cat(query)
  print(path2)
  cat(reference)
  d=dtw(query,reference, distance.only = TRUE)
  return (d$distance)
}
z=outer(moves$path, moves$path, FUN = dtw_dist)

ggplot(moves, aes(x=path, y=path)) + geom_tile(aes(fill= rt)) + scale_fill_gradient(low = "white",high = "steelblue")

lm <- matrix(nrow = 6, ncol = 6, byrow = TRUE, c(
  1,1,2,2,3,3,
  1,1,1,2,2,2,
  3,1,2,2,3,3,
  3,1,2,1,1,2,
  3,2,1,2,1,2,
  3,3,3,2,1,2 
))
alignment <- dtw(lm, step = asymmetric, keep = TRUE)
lcm <- alignment$localCostMatrix
image(x = 1:nrow(lcm), y = 1:ncol(lcm), lcm)
text(row(lcm), col(lcm), label = lcm)
lines(alignment$index1, alignment$index2)
ccm <- alignment$costMatrix
image(x = 1:nrow(ccm), y = 1:ncol(ccm), ccm)
text(row(ccm), col(ccm), label = ccm)
lines(alignment$index1, alignment$index2)



###################################################
##  Checking the linear relationship
################################################# 
i=read.csv('~/gdrivezb9/rushhour/results/instances/instances_selected_set4.csv', header = TRUE, sep=',',stringsAsFactors=F)
p=read.csv('~/gdrivezb9/rushhour/results/instances/instances_selected_set4_plan.csv', header = TRUE, sep=',',stringsAsFactors=F)
p=p[,c('instance','number_of_unsafe_moves')]
i=merge(i,p,by = 'instance')
setwd("~/gdrivezb9/rushhour/results/all_stages")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, paths$complete == 'True')
d=merge(paths,i,by = 'instance')
d$err <- (d$human_length-d$optimal_length)
d$err_norm <- (d$err)/d$optimal_length
d$err_ratio <- (d$human_length/d$optimal_length)

f<-function(d,name){
  g1<-ggplot(d, aes_string(x=name, y=d$err)) + geom_point() + stat_smooth(method = 'lm') + stat_summary(color='red', fun.data = mean_sem) + ggtitle('err') + ylab('')
  g2<-ggplot(d, aes_string(x=name, y=d$err_norm)) + geom_point() + stat_smooth(method = 'lm') + stat_summary(color='red', fun.data = mean_sem)+ ggtitle('err_norm')+ ylab('')
  g3<-ggplot(d, aes_string(x=name, y=d$err_ratio)) + geom_point() + stat_smooth(method = 'lm') + stat_summary(color='red', fun.data = mean_sem)+ ggtitle('err_ratio')+ ylab('')
  g4<-ggplot(d, aes_string(x=name, y=d$rt)) + geom_point() + stat_smooth(method = 'lm') + stat_summary(color='red', fun.data = mean_sem)+ ggtitle('rt')+ ylab('')
  grid.arrange(g1,g2,g3,g4)
}

summary(lm(d$rt ~ d$avg_location_size))
summary(lm(d$rt ~ d$mag_nodes + d$avg_bf + d$mag_edges + d$avg_location_size + d$num_sccs + d$max_scc_size ))
summary(lm(d$err ~ d$mag_nodes + d$avg_bf + d$mag_edges + d$avg_location_size + d$num_sccs + d$max_scc_size ))
summary(lm(d$err_norm ~ d$mag_nodes + d$avg_bf + d$mag_edges + d$avg_location_size + d$num_sccs + d$max_scc_size ))

summary(lm(d$err ~ d$number_of_unsafe_moves + 
             d$mag_nodes + d$mag_edges + d$avg_location_size + 
             d$avg_bf + d$max_scc_size + d$num_sccs ))


summary(lm(d$rt ~ d$number_of_unsafe_moves))
summary(lm(d$err_norm ~ d$number_of_unsafe_moves))
summary(lm(d$err_ratio ~ d$number_of_unsafe_moves))


d$unsafe<-ifelse(d$number_of_unsafe_moves > 0,1,0)
summary(lm(d$err ~ d$unsafe))
summary(lm(d$rt ~ d$unsafe))
summary(lm(d$err_norm ~ d$unsafe))
summary(lm(d$err_ratio ~ d$unsafe))


sl=split(d,as.factor(d$optimal_length))
f(sl$`7`,'avg_bf')
f(sl$`11`,'avg_bf')
f(sl$`14`,'avg_bf')
f(sl$`16`,'avg_bf')

f(sl$`7`,'number_of_unsafe_moves')
f(sl$`11`,'number_of_unsafe_moves')
f(sl$`14`,'number_of_unsafe_moves')
f(sl$`16`,'number_of_unsafe_moves')

f(sl$`7`,'mag_nodes')
f(sl$`11`,'mag_nodes')
f(sl$`14`,'mag_nodes')
f(sl$`16`,'mag_nodes')


f(sl$`7`,'mag_edges')
f(sl$`11`,'mag_edges')
f(sl$`14`,'mag_edges')
f(sl$`16`,'mag_edges')

f(sl$`7`,'avg_location_size')
f(sl$`11`,'avg_location_size')
f(sl$`14`,'avg_location_size')
f(sl$`16`,'avg_location_size')

f(sl$`7`,'max_scc_size')
f(sl$`11`,'max_scc_size')
f(sl$`14`,'max_scc_size')
f(sl$`16`,'max_scc_size')

f(sl$`7`,'num_sccs')
f(sl$`11`,'num_sccs')
f(sl$`14`,'num_sccs')
f(sl$`16`,'num_sccs')


#######################################
##  Plot 4: Correlation matrix. Show relationship between instances and perofrmance.
###################################### 
i=read.csv('~/gdrivezb9/rushhour/results/instances/instances_selected_set4.csv', header = TRUE, sep=',',stringsAsFactors=F)
setwd("~/gdrivezb9/rushhour/results/all_stages")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, paths$complete == 'True')
d=merge(paths,i,by = 'instance')
d$err_norm <- (d$human_length-d$optimal_length)/d$optimal_length
d$err <- (d$human_length-d$optimal_length)
d=d[,c('optimal_length','human_length','rt','err','err_norm','v_size', 'mag_nodes',
       'mag_edges','path_length','num_sccs','max_scc_size','avg_bf','avg_location_size')]
cormat=round(cor(d,method = "spearman"),3)
library('reshape2')
melted=melt(cormat)
melted=subset(melted, melted$Var2 %in% c('err','err_norm','rt'))
ggplot(melted, aes(x=melted$Var1, y=melted$Var2, fill=melted$value)) + geom_tile()+ geom_text(label = melted$value)


#################################################
##  stat 5: p.values and corelation coefficients.
#################################################
i=read.csv('~/gdrivezb9/rushhour/results/instances/instances_selected_set4.csv', header = TRUE, sep=',',stringsAsFactors=F)
setwd("~/gdrivezb9/rushhour/results/all_stages")
paths=read.csv('paths.csv', header = TRUE, sep=',',stringsAsFactors=F)
paths=subset(paths, paths$complete == 'True')
d=merge(paths,i,by = 'instance')
d$err_norm <- (d$human_length-d$optimal_length)/d$optimal_length
d$err <- (d$human_length-d$optimal_length)
factors<-c('v_size', 'mag_nodes','mag_edges','num_sccs','max_scc_size','avg_bf','avg_location_size')
responses<-c('optimal_length','human_length','rt','err','err_norm')
k=expand.grid(factors,responses)
k$pval=apply(k,1,function(x){cor.test(d[,x[1]],d[,x[2]], method = "spearman", alternative = "two.sided")$p.value})
k$estimate=apply(k,1,function(x){cor.test(d[,x[1]],d[,x[2]], method = "spearman", alternative = "two.sided")$estimate})



