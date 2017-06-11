# santander customer satisfaction
library(mlbench)
library(caret)
library(acepack)
library(homals)
library(abodOutlier)
library(cluster)
library(fpc)
library(clValid)
library(xgboost)
library(plyr)
library(fastAdaboost)
library(caretEnsemble)
library(randomForest)
library(pROC)
library(RRF)

main <- function()
{
  trainData <- read.csv(file="C:/Users/Dhrumel/Desktop/custSat_train.csv",header=TRUE) #(76020 * 370)
  
  TARGET <- trainData$TARGET
  trainData$TARGET <- NULL
  trainData <- clean(trainData)
  trainData$TARGET <- TARGET
  
  trainData <- cfs(trainData) # 279 remain
  #trainData <- pca(trainData) # 250 remain
  #trainData <- nlpca(trainData)
  trainData <- clarans_abod(trainData)
  
  testData <- read.csv(file="C:/Users/Dhrumel/Desktop/custSat_test.csv",header=TRUE) #(75818 * 370)
  id <- testData$Id
  
  testData <- clean(testData)
  
  testData <- testData[colnames(testData) %in% colnames(trainData)]       #ensure same columns in train and test
  
  TARGET <- trainData$TARGET
  trainData <- trainData[colnames(trainData) %in% colnames(testData)]
  trainData <-cbind(trainData,TARGET)
  
  results <- train_test(trainData,testData)
  
  results$ID <- id
  write.csv(results,file="C:/Users/Dhrumel/Desktop/sanResult_e1.csv")
}

train_test <- function(training,testing)
{
  training$TARGET[training$TARGET==1] <- "dis"
  training$TARGET[training$TARGET==0] <- "sat"
  training$TARGET <- as.factor(training$TARGET)
  
  set.seed(12579)
  control1 <- trainControl(method="cv",number=2,savePredictions="final",classProbs=TRUE, summaryFunction = twoClassSummary)
  modelList <- caretList(TARGET~.,data=training,trControl=control1,tuneList=list(xgb1=caretModelSpec(method="xgbTree",
                          tuneGrid=expand.grid(eta=0.02,max_depth=5,nrounds=560,gamma=1,min_child_weight=5,subsample=0.7,colsample_bytree=0.8)),
                          rf1=caretModelSpec(method="RRF",maxnodes=1000,tuneGrid=expand.grid(mtry=20,coefReg=0.5,coefImp=0.8))))
                        
  control2 <- trainControl(method="cv",number=2,savePredictions="final",classProbs=TRUE, summaryFunction = twoClassSummary)
  ensemble <- caretEnsemble(modelList,metric="ROC",trControl=control2)
  summary(ensemble)

  p <- data.frame(predict(ensemble,newdata=testing,type="prob"))
}

clarans_abod <- function(x) # remove outliers using clarans and abod ()
{
  print(nrow(x))
  k<-35
  results <- clara(x,k,samples=10,sampsize = 1000)
  x$cluster <- results$clustering
  cluster <- data.frame(index=1:k,size=results$clusinfo[,"size"], med=results$i.med)
  x$index = 1:nrow(x)
  print(cluster)
  
  #remove clusters with less than 1% of observations (23 of 35 remain)
  outlier <- cluster[cluster[,2]<(0.01*nrow(x)),"index"]
  cluster <- cluster[cluster[,2]>=(0.01*nrow(x)),]
  x <- x[!(x$cluster %in% outlier),]
  
  print(nrow(x))
  print(cluster)
  
  #eliminate clusters with small abod factors (22 of 23 remain)
  
  cluster$abodFactor <- abod(x[x$index %in% cluster$med,(1:(ncol(x)-2))])
  
  outlier<-cluster[cluster$abodFactor<((mean(cluster$abodFactor))-2*sd(cluster$abodFactor)),"index"]
  
  x <- x[!(x$cluster %in% outlier),]
  cluster <- cluster[cluster$abodFactor>=((mean(cluster$abodFactor))-2*sd(cluster$abodFactor)),]
  
  print(nrow(x))
  print(cluster)

  #eliminate observations within clusters far from the median
  for(i in 1:nrow(cluster))
  {
    dist <- numeric()
    outlier <- integer()
    clust <- x[x["cluster"]==cluster[i,"index"],]
    median <- clust[clust$index==cluster[i,"med"],1:(ncol(clust)-1)]
    for(j in 1:nrow(clust))
      dist =  c(dist,sqrt(sum((clust[j,1:(ncol(clust)-1)]-median)^2)))
    for(j in 1:nrow(clust))
     {
      if(dist[j]>((mean(dist))+2*sd(dist))&&dist[j]!=0)
        outlier=c(outlier,clust[j,"index"])
    }
    cluster[i,"size"]<-cluster[i,"size"]-length(outlier)
    x <- x[!(x$index %in% outlier),]
  }
  x <- x[,-c(ncol(x)-1,ncol(x))]
  
  print(nrow(x))
  x
}


nlpca <- function(x)  # not used
{
  x <- homals(x,ndim=2,level="numerical")
  x <- data.frame(x$eigenvalues)
}

pca <- function(data)   # not used
{
  TARGET <- data$TARGET
  data$TARGET <- NULL
  prin_comp <- prcomp(data)
  variance <- (prin_comp$sdev)^2
  prop_var <- variance/sum(variance)
  plot(cumsum(prop_var))
  data <- data.frame(prin_comp$x)
  data <- data[,1:250]
  data <- cbind(data,TARGET)
  
  testData<-as.data.frame(predict(prin_comp, newdata=testData))
  testData<-testData[1:250]
}

#remove highly correlated attributes using maximal correlation to capture non-linear associations
#Betweeen the 2 correlated attributes, the one with smaller correlation with TARGET is removed
cfs<-function(data)
{
  n <- ncol(data)
  i<-1
  while(i < (n-1))
  {
    j<-i+1
    while(j < n)
    {
      argmax =ace(data[,i],data[,j])
      c <- cor(argmax$tx, argmax$ty)
      if(c>0.999)
      {
        argmax =ace(data[,i],data[,"TARGET"])
        corTarget1 <- cor(argmax$tx, argmax$ty)
        argmax =ace(data[,j],data[,"TARGET"])
        corTarget2 <- cor(argmax$tx, argmax$ty)
        n<-n-1
        if(corTarget1 <= corTarget2)
        {
          data[,i]<-NULL
          i<-i-1
          break
        }
        else
        {
          data[,j] <-NULL
          j<-j-1
        }
      }
      j<-j+1
    }
    i<-i+1
  }
  print(ncol(data))
  data
}

clean<-function(data)
{
	
	data$Id<-NULL     #remove id attribute
	data$var3[data$var3 < 0] <-0  	#convert -99999 from var3 to 0

	#convert noise values > 999,999,999 to mode
	for(j in (1:ncol(data)))
	{
	  ux <- unique(data[,j])
	  data[(data[,j]>999999999),j] <- ux[which.max(tabulate(match(data[,j],ux)))]
	}
	  
	#create new attributes for those columns with min < 0 and max > 1 
	
	new<-data[,(lapply(data,max)>1 & lapply(data,min)<0)] 
	new[new<0] <- 0
	new[new>0] <- 1
	data <- cbind(data,new)
	
	#add min to attributes with min < 0 and max > 1 to allow log transformation
	
	#data[,(lapply(data,max)>1 & lapply(data,min)<0)]<-as.data.frame(lapply(data[,(lapply(data,max)>1 & lapply(data,min)<0)],addmin))
	
	#add 1 and log attributes with max > 1
	
	#data[,(lapply(data,max)>1)] <- data[,(lapply(data,max)>1)] + 1
	#data[,(lapply(data,max)>1)] <- as.data.frame(lapply(data[,(lapply(data,max)>1)],log10))
	
	data<-data[,lapply(data,var)!=0]   #remove attributes with variance 0
  soz<-integer()
	for(i in 1: nrow(data))         #add sum of zeros attribute
	  soz[i] <- sum(data[i,]==0)
	data <- cbind(data,soz)
	
	print(ncol(data))
	
	data<-as.data.frame(lapply(data,normalize))    #min-max normalization

}

normalize <- function(x)
{
  x <- ((x-min(x))/(max(x)-min(x)))
}
addmin <- function(x)
{
  x <- x - min(x)
}