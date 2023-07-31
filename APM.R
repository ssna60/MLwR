#ch3
apropos("confusion")
RSiteSearch("confusion", restrict = "functions")

library(AppliedPredictiveModeling)
data(segmentationOriginal)
segData <- subset(segmentationOriginal, Case == "Train")
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
# Now remove the columns
segData <- segData[, -(1:3)]
statusColNum <- grep("Status", names(segData))
statusColNum
segData <- segData[, -statusColNum]

#Transformations
library(e1071)
# For one predictor:
skewness(segData$AngleCh1)
# Since all the predictors are numeric columns, the apply function can
# be used to compute the skewness across columns.
skewValues <- apply(segData, 2, skewness)
head(skewValues)
library(caret)
Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)
Ch1AreaTrans
# The original data
head(segData$AreaCh1)
# After transformation
predict(Ch1AreaTrans, head(segData$AreaCh1))
(819^(-.9) - 1)/(-.9)
pcaObject <- prcomp(segData,
                    center = TRUE, scale. = TRUE)
# Calculate the cumulative percentage of variance which each component
# accounts for.
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
percentVariance[1:3]
head(pcaObject$x[, 1:5])
head(pcaObject$rotation[, 1:3])

#spatialSign(segData)
#library(preProcess)
#impute.knn

trans <- preProcess(segData,
                    method = c("BoxCox", "center", "scale", "pca"))
trans
# Apply the transformations:
transformed <- predict(trans, segData)
# These values are different than the previous PCA components since
# they were transformed prior to PCA
head(transformed[, 1:5])

#Filtering
nearZeroVar(segData)
# When predictors should be removed, a vector of integers is
# returned that indicates which columns should be removed.
correlations <- cor(segData)
dim(correlations)
correlations[1:4, 1:4]
library(corrplot)
corrplot(correlations, order = "hclust")
highCorr <- findCorrelation(correlations, cutoff = .75)
length(highCorr)
head(highCorr)
filteredSegData <- segData[, -highCorr]

#Creating Dummy Variables
library(caret)#
data(cars)#
head(cars)#

head(carSubset)
levels(carSubset$Type)
simpleMod <- dummyVars(~Mileage + Type,
                       data = carSubset,
                       ## Remove the variable name from the
                       ## column name
                       levelsOnly = TRUE)
simpleMod

predict(simpleMod, head(carSubset))


withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                             data = carSubset,
                             levelsOnly = TRUE)
withInteraction
predict(withInteraction, head(carSubset))

#

#
library(earth)
data(etitanic)
head(model.matrix(survived ~ ., data = etitanic))
dummies <- dummyVars(survived ~ ., data = etitanic)
head(predict(dummies, newdata = etitanic))

dummies <- dummyVars(~pclass + sex + age, data = etitanic)
head(predict(dummies, newdata = etitanic))

dummies <- dummyVars(~., data = etitanic)
head(predict(dummies, newdata = etitanic))

dummies <- dummyVars(survived ~., data = etitanic)
head(predict(dummies, newdata = etitanic))
#


# NOT RUN {
when <- data.frame(time = c("afternoon", "night", "afternoon",
                            "morning", "morning", "morning",
                            "morning", "afternoon", "afternoon"),
                   day = c("Mon", "Mon", "Mon",
                           "Wed", "Wed", "Fri",
                           "Sat", "Sat", "Fri"),
                   stringsAsFactors = TRUE)

levels(when$time) <- list(morning="morning",
                          afternoon="afternoon",
                          night="night")
levels(when$day) <- list(Mon="Mon", Tue="Tue", Wed="Wed", Thu="Thu",
                         Fri="Fri", Sat="Sat", Sun="Sun")

## Default behavior:
model.matrix(~day, when)

mainEffects <- dummyVars(~ day + time, data = when)
mainEffects
predict(mainEffects, when[1:3,])

when2 <- when
when2[1, 1] <- NA
predict(mainEffects, when2[1:3,])
predict(mainEffects, when2[1:3,], na.action = na.omit)


interactionModel <- dummyVars(~ day + time + day:time,
                              data = when,
                              sep = ".")
predict(interactionModel, when[1:3,])

noNames <- dummyVars(~ day + time + day:time,
                     data = when,
                     levelsOnly = TRUE)
predict(noNames, when)

head(class2ind(iris$Species))

two_levels <- factor(rep(letters[1:2], each = 5))
class2ind(two_levels)
class2ind(two_levels, drop2nd = TRUE)
# }

#















#ch4
library(AppliedPredictiveModeling)
data(twoClassData)
str(predictors)
str(classes)
# Set the random number seed so we can reproduce the results
  set.seed(1)
# By default, the numbers are returned as a list. Using
  # list = FALSE, a matrix of row numbers is generated.
  # These samples are allocated to the training set.
trainingRows <- createDataPartition(classes,
                                    p = .80,
                                    list= FALSE)
head(trainingRows)

# Subset the data into objects for training using
  # integer sub-setting.
trainPredictors <- predictors[trainingRows, ]
trainClasses <- classes[trainingRows]
# Do the same for the test set using negative integers.
  testPredictors <- predictors[-trainingRows, ]
testClasses <- classes[-trainingRows]
str(trainPredictors)

str(testPredictors)

#Resampling
set.seed(1)
# For illustration, generate the information needed for three
  # resampled versions of the training set.
repeatedSplits <- createDataPartition(trainClasses, p = .80,
                                      times = 3)
str(repeatedSplits)

set.seed(1)
cvSplits <- createFolds(trainClasses, k = 10,
                        returnTrain = TRUE)
str(cvSplits)


# Get the first set of row numbers from the list.
fold1 <- cvSplits[[1]]

cvPredictors1 <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]
nrow(trainPredictors)
nrow(cvPredictors1)

#Basic Model Building in R

#modelFunction(price ~ numBedrooms + numBaths + acres,
#                + data = housingData)
#modelFunction(x = housePredictors, y = price)

trainPredictors <- as.matrix(trainPredictors)
knnFit <- knn3(x = trainPredictors, y = trainClasses, k = 5)
knnFit

testPredictions <- predict(knnFit, newdata = testPredictors,
                           type = "class")
head(testPredictions)

str(testPredictions)

#Determination of Tuning Parameters
library(caret)
data(GermanCredit)

#
trainingRows <- createDataPartition(GermanCredit$Class,
                                    p = .80,
                                    list= FALSE)
head(trainingRows)
trainPredictors <- GermanCredit[trainingRows, ]
trainClasses <- class[trainingRows]
testPredictors <- predictors[-trainingRows, ]
testClasses <- class[-trainingRows]
str(trainPredictors)
str(testPredictors)
#


set.seed(1056)

svmFit <- train(Class ~ .,
                  data = GermanCredit[trainingRows, ], #GermanCreditTrain,
                  # The "method" argument indicates the model type.
                    # See ?train for a list of available models.
                    method = "svmRadial")

set.seed(1056)
svmFit <- train(Class ~ .,
                  data = GermanCredit[trainingRows, ], #GermanCreditTrain,
                  method = "svmRadial",
                  preProc = c("center", "scale"))


set.seed(1056)
svmFit <- train(Class ~ .,
                  data = GermanCredit[trainingRows, ], #GermanCreditTrain,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 10)


set.seed(1056)
svmFit <- train(Class ~ .,
                  data = GermanCredit[trainingRows, ], #GermanCreditTrain,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 10,
                  trControl = trainControl(method = "repeatedcv",
                                             repeats = 5))
svmFit

# A line plot of the average performance
plot(svmFit, scales = list(x = list(log = 2)))

predictedClasses <- predict(svmFit, GermanCredit[trainingRows, ]) #GermanCreditTrain)
str(predictedClasses)

# Use the "type" option to get class probabilities
predictedProbs <- predict(svmFit, newdata = GermanCredit[-trainingRows, ], #GermanCreditTest,
                            type = "prob")
head(predictedProbs)

#Between-Model Comparisons
set.seed(1056)
logisticReg <- train(Class ~ .,
                     data = GermanCredit[trainingRows, ], #GermanCreditTrain,
                     method = "glm",
                     trControl = trainControl(method = "repeatedcv",
                                              repeats = 5))
logisticReg

resamp <- resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)

modelDifferences <- diff(resamp)
summary(modelDifferences)


#ch5

# Use the 'c' function to combine numbers into a vector
observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4,
              0.62, 0.99, -0.18, 0.32, 0.34, -0.30, 0.04, -0.87,
              0.55, -1.30, -1.15, 0.20)

predicted <- c(0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43,
               0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42,
               -0.25, -0.64, -1.26, -0.07)
residualValues <- observed - predicted
summary(residualValues)


# Observed values versus predicted values
  # It is a good idea to plot the values on a common scale.
  axisRange <- extendrange(c(observed, predicted))
plot(observed, predicted,
     ylim = axisRange,
     xlim = axisRange)
# Add a 45 degree reference line
abline(0, 1, col = "darkgrey", lty = 2)
# Predicted values versus residuals
plot(predicted, residualValues, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

R2(predicted, observed)
RMSE(predicted, observed)

# Simple correlation
cor(predicted, observed)

# Rank correlation
cor(predicted, observed, method = "spearman")


#ch6
library(AppliedPredictiveModeling)
data(solubility)
## The data objects begin with "sol":
ls(pattern = "^solT")
set.seed(2)
sample(names(solTrainX), 8)

#Ordinary Linear Regression

trainingData <- solTrainXtrans
## Add the solubility outcome
trainingData$Solubility <- solTrainY
lmFitAllPredictors <- lm(Solubility ~ ., data = trainingData)
summary(lmFitAllPredictors)

lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues1)

rlmFitAllPredictors <- rlm(Solubility ~ ., data = trainingData)

ctrl <- trainControl(method = "cv", number = 10)

set.seed(100)
lmFit1 <- train(x = solTrainXtrans, y = solTrainY,
                method = "lm", trControl = ctrl)
lmFit1


xyplot(solTrainY ~ predict(lmFit1),
       ## plot the points (type = 'p') and a background grid ('g')
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")
xyplot(resid(lmFit1) ~ predict(lmFit1),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")

corThresh <- .9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
corrPred <- names(solTrainXtrans)[tooHigh]
trainXfiltered <- solTrainXtrans[, -tooHigh]

testXfiltered <- solTestXtrans[, -tooHigh]
set.seed(100)
lmFiltered <- train(solTrainXtrans, solTrainY, method = "lm",
                    trControl = ctrl)
lmFiltered

set.seed(100)
rlmPCA <- train(solTrainXtrans, solTrainY,
                method = "rlm",
                preProcess = "pca",
                trControl = ctrl)
rlmPCA                  
                  

#Partial Least Squares
plsFit <- plsr(Solubility ~ ., data = trainingData)
predict(plsFit, solTestXtrans[1:5,], ncomp = 1:2)

set.seed(100)
plsTune <- train(solTrainXtrans, solTrainY,
                 method = "pls",
                 ## The default tuning grid evaluates
                 ## components 1... tuneLength
                 tuneLength = 20,
                 trControl = ctrl,
                 preProc = c("center", "scale"))

#Penalized Regression Models
ridgeModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                   lambda = 0.001)

ridgePred <- predict(ridgeModel, newx = as.matrix(solTestXtrans),
                     s = 1, mode = "fraction",
                     type = "fit")
head(ridgePred$fit)

## Define the candidate set of values
 ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
set.seed(100)
ridgeRegFit <- train(solTrainXtrans, solTrainY,
                     method = "ridge",
                     ## Fir the model over many penalty values
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     ## put the predictors on the same scale
                     preProc = c("center", "scale"))
ridgeRegFit

enetModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                  lambda = 0.01, normalize = TRUE)

enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans),
                    s = .1, mode = "fraction",
                    type = "fit")
## A list is returned with several items:
names(enetPred)

## The 'fit' component has the predicted values:
head(enetPred$fit)

enetCoef<- predict(enetModel, newx = as.matrix(solTestXtrans),
                   s = .1, mode = "fraction",
                   type = "coefficients")
tail(enetCoef$coefficients)

enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(solTrainXtrans, solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
#ch7

#Neural Networks
install.packages('earth')
install.packages('kernlab')
install.packages('nnet')

library(caret)
library(earth)
library(kernlab)
library(nnet)


library(AppliedPredictiveModeling)
data(solubility)

nnetFit <- nnet(predictors, outcome,
                size = 5,
                decay = 0.01,
                linout = TRUE,
                ## Reduce the amount of printed output
                trace = FALSE,
                ## Expand the number of iterations to find
                ## parameter estimates..
                maxit = 500,
                ## and the number of parameters used by the model
                MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)

nnetAvg <- avNNet(predictors, outcome,
                  size = 5,
                  decay = 0.01,
                  ## Specify how many models to average
                  repeats = 5,
                  linout = TRUE,
                  ## Reduce the amount of printed output
                  trace = FALSE,
                  ## Expand the number of iterations to find
                  ## parameter estimates..
                  maxit = 500,
                  ## and the number of parameters used by the model
                  MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)



predict(nnetFit, newData)
## or
predict(nnetAvg, newData)


## The findCorrelation takes a correlation matrix and determines the
## column numbers that should be removed to keep all pair-wise
## correlations below a threshold
tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff = .75)
trainXnnet <- solTrainXtrans[, -tooHigh]
testXnnet <- solTestXtrans[, -tooHigh]
## Create a specific candidate set of models to evaluate:
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = FALSE)
set.seed(100)
nnetTune <- train(solTrainXtrans, solTrainY,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl, #???
                  ## Automatically standardize data prior to modeling
                  ## and prediction
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 500)


#Multivariate Adaptive Regression Splines MARS model
marsFit <- earth(solTrainXtrans, solTrainY)
marsFit
summary(marsFit)

# Define the candidate models to test
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
# Fix the seed so that the results can be reproduced
set.seed(100)
marsTuned <- train(solTrainXtrans, solTrainY,
                   method = "earth",
                   # Explicitly declare the candidate models to test
                   tuneGrid = marsGrid,
                   trControl = trainControl(method = "cv"))
marsTuned
head(predict(marsTuned, solTestXtrans))

varImp(marsTuned)


#Support Vector Machines
svmFit <- ksvm(x = solTrainXtrans, y = solTrainY,
               kernel ="rbfdot", kpar = "automatic",
               C = 1, epsilon = 0.1)

svmRTuned <- train(solTrainXtrans, solTrainY,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneLength = 14,
                   trControl = trainControl(method = "cv"))
svmRTuned
svmRTuned$finalModel

#K-Nearest Neighbors
# Remove a few sparse and unbalanced fingerprints first
knnDescr <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]
set.seed(100)
knnTune <- train(knnDescr,
                 solTrainY,
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "cv"))


#ch8

install.packages('Cubist')
install.packages('gbm')
install.packages('ipred')
install.packages('party')
install.packages('partykit')
install.packages('randomForest')
install.packages('rpart')
install.packages('RWeka')

library(caret)
library(Cubist)
library(gbm)
library(ipred)
library(party)
library(partykit)
library(randomForest)
library(rpart)

install.packages('rJava')
options(java.home="C:\\Program Files\\Java\\jre1.8.0_311\\bin")
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_311\\bin')
library(rJava)

library(RWeka)

library(AppliedPredictiveModeling)
data(solubility)

#Single Trees

library(rpart)
trainData <- solTrainX #
y <- solTrainY #
rpartTree <- rpart(y ~ ., data = trainData)
# or,
ctreeTree <- ctree(y ~ ., data = trainData)

set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY,
                   method = "rpart2",
                   tuneLength = 10,
                   trControl = trainControl(method = "cv"))

plot(treeObject)#???


library(partykit)
rpartTree2 <- as.party(rpartTree)
plot(rpartTree2)


#Model Trees
library(RWeka)#???
m5tree <- M5P(y ~ ., data = trainData)
# or, for rules:
m5rules <- M5Rules(y ~ ., data = trainData)

m5tree <- M5P(y ~ ., data = trainData,
              control = Weka_control(M = 10))

set.seed(100)
m5Tune <- train(solTrainXtrans, solTrainY,
                method = "M5",
                trControl = trainControl(method = "cv"),
                ## Use an option for M5() to specify the minimum
                ## number of samples needed to further splits the
                ## data to be 10
                control = Weka_control(M = 10))


#Bagged Trees
library(ipred)
baggedTree <- ipredbagg(solTrainY, solTrainXtrans)
## or
baggedTree <- bagging(y ~ ., data = trainData)

library(party)
## The mtry parameter should be the number of predictors (the
## number of columns minus 1 for the outcome).
bagCtrl <- cforest_control(mtry = ncol(trainData) - 1)
baggedTree <- cforest(y ~ ., data = trainData, controls = bagCtrl)

#Random Forest
library(randomForest)
rfModel <- randomForest(solTrainXtrans, solTrainY)
## or
rfModel <- randomForest(y ~ ., data = trainData)

library(randomForest)
rfModel <- randomForest(solTrainXtrans, solTrainY,
                        importance = TRUE,
                        ntrees = 1000)


#Boosted Trees
library(gbm)
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, distribution = "gaussian")
## or
gbmModel <- gbm(y ~ ., data = trainData, distribution = "gaussian")

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage = c(0.01, 0.1))
set.seed(100)
gbmTune <- train(solTrainXtrans, solTrainY, #???
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 ## The gbm() function produces copious amounts
                 ## of output, so pass in the verbose option
                 ## to avoid printing a lot to the screen.
                 verbose = FALSE)

#Cubist
library(Cubist)
cubistMod <- cubist(solTrainXtrans, solTrainY)
predict(cubistMod, solTestXtrans)
cubistTuned <- train(solTrainXtrans, solTrainY, method = "cubist")

#ch10



load("C:/Users/samen/Desktop/R/Books/APM_R/ch10.RData")
library(AppliedPredictiveModeling)
data(concrete)
str(concrete)
str(mixtures)

#install.packages('ggplot2')
#install.packages('caret')
#install.packages('plyr')
#install.packages('lattice')
#install.packages('AppliedPredictiveModeling')

library(ggplot2)
library(caret)
library(plyr)
library(lattice)
library(AppliedPredictiveModeling)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)


featurePlot(x = concrete[, -9],
            y = concrete$CompressiveStrength,
            ## Add some space between the panels
            between = list(x = 1, y = 1),
            ## Add a background grid ('g') and a smoother ('smooth')
            type = c("g", "p", "smooth"))



#install.packages('Hmisc')
library(Hmisc)
describe(concrete)


#?? ??????? ???? ????? ??? ʘ??? ??? ? ????? ???? ?? ?? ?????? ??? ?????? ? ??????? 
averaged <- ddply(mixtures,
                  .(Cement, BlastFurnaceSlag, FlyAsh, Water,
                  Superplasticizer, CoarseAggregate,
                  FineAggregate, Age),
                  function(x) c(CompressiveStrength =
                                  mean(x$CompressiveStrength)))
set.seed(975)
forTraining <- createDataPartition(averaged$CompressiveStrength,
                                   p = 3/4)[[1]]
trainingSet <- averaged[ forTraining,]
testSet <- averaged[-forTraining,]

#????? ????? ?? ????? ?? ???? ???ǘ?? ? Ӂ? ?? ????? ????? ????? ?? ??? 
modFormula <- paste("CompressiveStrength ~ (.)^2 + I(Cement^2) + ",
                    "I(BlastFurnaceSlag^2) + I(FlyAsh^2) + I(Water^2) +",
                    " I(Superplasticizer^2) + I(CoarseAggregate^2) + ",
                    "I(FineAggregate^2) + I(Age^2)")
modFormula <- as.formula(modFormula)

#?????????? ?????? 10 ?????? ???? 
controlObject <- trainControl(method = "repeatedcv",
                              repeats = 1,
                              number = 2)

set.seed(669)
linearReg <- train(modFormula,
                   data = trainingSet,
                   method = "lm",
                   trControl = controlObject)
linearReg


set.seed(669)
plsModel <- train(modFormula, data = trainingSet,
                  method = "pls",
                  preProc = c("center", "scale"),
                  tuneLength = 15,
                  trControl = controlObject)
enetGrid <- expand.grid(.lambda = c(0, .001, .01, .1),
                        .fraction = seq(0.05, 1, length = 20))
set.seed(669)
enetModel <- train(modFormula, data = trainingSet,
                   method = "enet",
                   preProc = c("center", "scale"),
                   tuneGrid = enetGrid,
                   trControl = controlObject)
#install.packages('tictoc')
library(tictoc)

#MARs, ANN, SVM 
set.seed(669)
tic()
earthModel <- train(CompressiveStrength ~ ., data = trainingSet,
                    method = "earth",
                    tuneGrid = expand.grid(.degree = 1,
                                           .nprune = 2:25),
                    trControl = controlObject)
toc()
#38.25 sec elapsed
set.seed(669)
tic()
svmRModel <- train(CompressiveStrength ~ ., data = trainingSet,
                   method = "svmRadial",
                   tuneLength = 15,
                   preProc = c("center", "scale"),
                   trControl = controlObject)
nnetGrid <- expand.grid(.decay = c(0.001, .01, .1),
                        .size = seq(1, 27, by = 2),
                        .bag = FALSE)
toc()
#397.34 sec elapsed = 6 min
set.seed(669)

tic()
nnetModel <- train(CompressiveStrength ~ .,
                   data = trainingSet,
                   method = "avNNet",
                   tuneGrid = nnetGrid,
                   preProc = c("center", "scale"),
                   linout = TRUE,
                   trace = FALSE,
                   maxit = 1000,
                   trControl = controlObject)
toc()
#10288.64 sec elapsed

set.seed(669)
tic()
rpartModel <- train(CompressiveStrength ~ .,
                    data = trainingSet,
                    method = "rpart",
                    tuneLength = 30,
                    trControl = controlObject)
toc()
#5.61 sec elapsed
set.seed(669)

tic()
ctreeModel <- train(CompressiveStrength ~ .,
                    data = trainingSet,
                    method = "ctree",
                    tuneLength = 10,
                    trControl = controlObject)
toc()
#9.49 sec elapsed

set.seed(669)

##
#Error: package RWeka is required
tic()
mtModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "M5",
                 trControl = controlObject)
toc()
##

tic()
set.seed(669)
treebagModel <- train(CompressiveStrength ~ .,
                      data = trainingSet,
                      method = "treebag",
                      trControl = controlObject)
toc()
#7.61 sec elapsed
tic()
set.seed(669)
rfModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "rf",
                 tuneLength = 10,
                 ntrees = 1000,
                 importance = TRUE,
                 trControl = controlObject)
toc()
#416.47 sec elapsed
#install.packages('gbm')
library(gbm)
gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage = c(0.01, 0.1),
                       .n.minobsinnode = c(5, 10, 20, 30))#
tic()
set.seed(669)
gbmModel <- train(CompressiveStrength ~ .,
                  data = trainingSet,
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  verbose = FALSE,
                  trControl = controlObject)
toc()
#501.27 sec elapsed

cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                          .neighbors = c(0, 1, 3, 5, 7, 9))
tic()
set.seed(669)
cbModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "cubist",
                 tuneGrid = cubistGrid,
                 trControl = controlObject)
toc()
#354.55 sec elapsed
library(caret)
allResamples <- resamples(list("Linear Reg" = linearReg,
                               "PLS" = plsModel,
                               "Elastic Net" = enetModel,
                               MARS = earthModel,
                               SVM = svmRModel,
                               "Neural Networks" = nnetModel,
                               CART = rpartModel,
                               "Cond Inf Tree" = ctreeModel,
                               "Bagged Tree" = treebagModel,
                               "Boosted Tree" = gbmModel,
                               "Random Forest" = rfModel,
                               Cubist = cbModel))

## Plot the RMSE values
#install.packages('lattice')
#library(lattice)
#install.packages('parallelPlot')
library(parallelPlot)
parallelPlot(as.data.frame(allResamples))
## Using R-squared:
parallelplot(allResamples, metric = "Rsquared")

nnetPredictions <- predict(nnetModel, testSet)
gbmPredictions <- predict(gbmModel, testSet)
cbPredictions <- predict(cbModel, testSet)

age28Data <- subset(trainingSet, Age == 28)
## Remove the age and compressive strength columns and
## then center and scale the predictor columns
pp1 <- preProcess(age28Data[, -(8:9)], c("center", "scale"))
scaledTrain <- predict(pp1, age28Data[, 1:7])
set.seed(91)
startMixture <- sample(1:nrow(age28Data), 1)
starters <- scaledTrain[startMixture, 1:7]

#??? ????? ???? ??ǘ?? ??? ?????
pool <- scaledTrain
index <- maxDissim(starters, pool, 14)
startPoints <- c(startMixture, index)
starters <- age28Data[startPoints,1:7]

## Remove water
  startingValues <- starters[, -4]


## The inputs to the function are a vector of six mixture proportions
  ## (in argument 'x') and the model used for prediction ('mod')
  modelPrediction <- function(x, mod)
    {
    ## Check to make sure the mixture proportions are
    ## in the correct range
    if(x[1] < 0 | x[1] > 1) return(10^38)
    if(x[2] < 0 | x[2] > 1) return(10^38)
    if(x[3] < 0 | x[3] > 1) return(10^38)
    if(x[4] < 0 | x[4] > 1) return(10^38)
    if(x[5] < 0 | x[5] > 1) return(10^38)
    if(x[6] < 0 | x[6] > 1) return(10^38)
    
    ## Determine the water proportion
    x <- c(x, 1 - sum(x))
    ## Check the water range
    if(x[7] < 0.05) return(10^38)
    ## Convert the vector to a data frame, assign names
    ## and fix age at 28 days
    tmp <- as.data.frame(t(x))
    names(tmp) <- c('Cement','BlastFurnaceSlag','FlyAsh',
                    'Superplasticizer','CoarseAggregate',
                    'FineAggregate', 'Water')
    tmp$Age <- 28
    ## Get the model prediction, square them to get back to the
    ## original units, then return the negative of the result
    -predict(mod, tmp)
    }


cbResults <- startingValues
cbResults$Water <- NA
cbResults$Prediction <- NA
## Loop over each starting point and conduct the search
  for(i in 1:nrow(cbResults))
    {
    results <- optim(unlist(cbResults[i,1:6]),
                     modelPrediction,
                     method = "Nelder-Mead",
                     ## Use method = 'SANN' for simulated annealing
                     control=list(maxit=5000),
                     ## The next option is passed to the
                     ## modelPrediction() function
                     mod = cbModel)
    ## Save the predicted compressive strengthcbResults$Prediction[i] <- -results$value
    ## Also save the final mixture values
    cbResults[i,1:6] <- results$par
    }
## Calculate the water proportion
cbResults$Water <- 1 - apply(cbResults[,1:6], 1, sum)
## Keep the top three mixtures
cbResults <- cbResults[order(-cbResults$Prediction),][1:3,]
cbResults$Model <- "Cubist"

nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA
for(i in 1:nrow(nnetResults))
  {
  results <- optim(unlist(nnetResults[i, 1:6,]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = nnetModel)
  nnetResults$Prediction[i] <- -results$value
  nnetResults[i,1:6] <- results$par
  }
nnetResults$Water <- 1 - apply(nnetResults[,1:6], 1, sum)
nnetResults <- nnetResults[order(-nnetResults$Prediction),][1:3,]
nnetResults$Model <- "NNet"

## Run PCA on the data at 28\,days
pp2 <- preProcess(age28Data[, 1:7], "pca")
## Get the components for these mixtures
pca1 <- predict(pp2, age28Data[, 1:7])
pca1$Data <- "Training Set"
## Label which data points were used to start the searches
pca1$Data[startPoints] <- "Starting Values"
## Project the new mixtures in the same way (making sure to
## re-order the columns to match the order of the age28Data object).
pca3 <- predict(pp2, cbResults[, names(age28Data[, 1:7])])
pca3$Data <- "Cubist"
pca4 <- predict(pp2, nnetResults[, names(age28Data[, 1:7])])
pca4$Data <- "Neural Network"
## Combine the data, determine the axis ranges and plot
pcaData <- rbind(pca1, pca3, pca4)
pcaData$Data <- factor(pcaData$Data,
                       levels = c("Training Set","Starting Values",
                                  "Cubist","Neural Network"))
lim <- extendrange(pcaData[, 1:2])
xyplot(PC2 ~ PC1, data = pcaData, groups = Data,
       auto.key = list(columns = 2),
       xlim = lim, ylim = lim,
       type = c("g", "p"))

#ch11
library(AppliedPredictiveModeling)
set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest <- quadBoundaryFunc(1000)
head(simulatedTrain)


library(randomForest)
rfModel <- randomForest(class ~ X1 + X2,
                        data = simulatedTrain,
                        ntree = 2000)
library(MASS) ## for the qda() function
qdaModel <- qda(class ~ X1 + X2, data = simulatedTrain)

qdaTrainPred <- predict(qdaModel, simulatedTrain)
names(qdaTrainPred)
head(qdaTrainPred$class)

head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, simulatedTest)
simulatedTrain$QDAprob <- qdaTrainPred$posterior[,"Class1"]
simulatedTest$QDAprob <- qdaTestPred$posterior[,"Class1"]

rfTestPred <- predict(rfModel, simulatedTest, type = "prob")
head(rfTestPred)

simulatedTest$RFprob <- rfTestPred[,"Class1"]
simulatedTest$RFclass <- predict(rfModel, simulatedTest)


#Sensitivity and Specificity
library(caret)
# Class 1 will be used as the event of interest
sensitivity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = "Class1")

specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = "Class2")

posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1")

negPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class2")

# Change the prevalence manually
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1",
             prevalence = .9)


#Confusion Matrix
confusionMatrix(data = simulatedTest$RFclass,
                reference = simulatedTest$class,
                positive = "Class1")


#Receiver Operating Characteristic Curves ROC curve
library(pROC)
rocCurve <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                ## This function assumes that the second
                ## class is the event of interest, so we
                ## reverse the labels.
                levels = rev(levels(simulatedTest$class)))

auc(rocCurve)

ci.roc(rocCurve) ##??

plot(rocCurve, legacy.axes = TRUE)
## By default, the x-axis goes backwards, used
## the option legacy.axes = TRUE to get 1-spec
## on the x-axis moving from 0 to 1

## Also, another curve can be added using
## add = TRUE the next time plot.auc is used.

  
#Lift Charts
  
labs <- c(RFprob = "Random Forest",
          QDAprob = "Quadratic Discriminant Analysis")
liftCurve <- lift(class ~ RFprob + QDAprob, data = simulatedTest,
                  labels = labs)
liftCurve
  
## Add lattice options to produce a legend on top
xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))
  
#Calibrating Probabilities
calCurve <- calibration(class ~ RFprob + QDAprob, data = simulatedTest)
calCurve
xyplot(calCurve, auto.key = list(columns = 2))

## The glm() function models the probability of the second factor
## level, so the function relevel() is used to temporarily reverse the
## factors levels.

sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAprob,
                    data = simulatedTrain,
                    family = binomial)
coef(summary(sigmoidalCal))

sigmoidProbs <- predict(sigmoidalCal,
                        newdata = simulatedTest[,"QDAprob", drop = FALSE],
                        type = "response")
simulatedTest$QDAsigmoid <- sigmoidProbs

#install.packages('klaR')
library(klaR)
BayesCal <- NaiveBayes(class ~ QDAprob, data = simulatedTrain,
                       usekernel = TRUE)
## Like qda(), the predict function for this model creates
## both the classes and the probabilities
BayesProbs <- predict(BayesCal,
                      newdata = simulatedTest[, "QDAprob", drop = FALSE])
simulatedTest$QDABayes <- BayesProbs$posterior[, "Class1"]
## The probability values before and after calibration
head(simulatedTest[, c(5:6, 8, 9)])

calCurve2 <- calibration(class ~ QDAprob + QDABayes + QDAsigmoid,
                         data = simulatedTest)
xyplot(calCurve2)


#ch12
length(fullSet)
head(fullSet)
length(reducedSet)
head(reducedSet)

reducedCovMat <- cov(training[, reducedSet])
library(subselect)
trimmingResults <- trim.matrix(reducedCovMat)
names(trimmingResults)
## See if any predictors were eliminated:
trimmingResults$names.discarded

## See if any predictors were eliminated:
trimmingResults$names.discarded

fullCovMat <- cov(training[, fullSet])
fullSetResults <- trim.matrix(fullCovMat)

## A different choices for the day to exclude was
## made by this function
fullSetResults$names.discarded

ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)


ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008))

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

#Logistic Regression
levels(training$Class)
modelFit <- glm(Class ~ Day,
                ## Select the rows for the pre-2008 data:
                data = training[pre2008,],
                ## 'family' relates to the distribution of the data.
                ## A value of 'binomial' is used for logistic regression
                family = binomial)
modelFit

successProb <- 1 - predict(modelFit,
                           ## Predict for several days
                           newdata = data.frame(Day = c(10, 150, 300,350)),
                           ## glm does not predict the class, but can
                           ## produce the probability of the event
                           type = "response")
successProb

daySquaredModel <- glm(Class ~ Day + I(Day^2),
                         + data = training[pre2008,],
                         + family = binomial)
daySquaredModel

library(rms)
rcsFit <- lrm(Class ~ rcs(Day), data = training[pre2008,])
rcsFit

dayProfile <- Predict(rcsFit,
                      ## Specify the range of the plot variable
                      Day = 0:365,
                      ## Flip the prediction to get the model for
                      ## successful grants
                      fun = function(x) -x)
plot(dayProfile, ylab = "Log Odds")

training$Day2 <- training$Day^2
fullSet <- c(fullSet, "Day2")
reducedSet <- c(reducedSet, "Day2")

library(caret)
set.seed(476)
lrFull <- train(training[,fullSet],
                y = training$Class,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)
lrFull



set.seed(476)
lrReduced <- train(training[,reducedSet],
                   y = training$Class,
                   method = "glm",
                   metric = "ROC",
                   trControl = ctrl)
lrReduced

head(lrReduced$pred)

confusionMatrix(data = lrReduced$pred$pred,
                reference = lrReduced$pred$obs)


reducedRoc <- roc(response = lrReduced$pred$obs,
                  predictor = lrReduced$pred$successful,
                  levels = rev(levels(lrReduced$pred$obs)))
plot(reducedRoc, legacy.axes = TRUE)
auc(reducedRoc)

#Linear Discriminant Analysis
library(MASS)
## First, center and scale the data
grantPreProcess <- preProcess(training[pre2008, reducedSet])
grantPreProcess

scaledPre2008 <- predict(grantPreProcess,
                         newdata = training[pre2008, reducedSet])
scaled2008HoldOut <- predict(grantPreProcess,
                             newdata = training[-pre2008, reducedSet])
ldaModel <- lda(x = scaledPre2008,
                grouping = training$Class[pre2008])

head(ldaModel$scaling)

ldaHoldOutPredictions <- predict(ldaModel, scaled2008HoldOut)


set.seed(476)
ldaFit1 <- train(x = training[, reducedSet],
                 y = training$Class,
                 method = "lda",
                 preProc = c("center","scale"),
                 metric = "ROC",
                 ## Defined above
                 trControl = ctrl)
ldaFit1


ldaTestClasses <- predict(ldaFit1,
                          newdata = testing[,reducedSet])
ldaTestProbs <- predict(ldaFit1,
                        newdata = testing[,reducedSet],
                        type = "prob")

#Partial Least Squares Discriminant Analysis

plsdaModel <- plsda(x = training[pre2008,reducedSet],
                    y = training[pre2008, "Class"],
                    ## The data should be on the same scale for PLS. The
                    ## 'scale' option applies this pre-processing step
                    scale = TRUE,
                    ## Use Bayes method to compute the probabilities
                    probMethod = "Bayes",
                    ## Specify the number of components to model
                    ncomp = 4)
## Predict the 2008 hold-out set
plsPred <- predict(plsdaModel,
                   newdata = training[-pre2008, reducedSet])
head(plsPred)

plsProbs <- predict(plsdaModel,
                    newdata = training[-pre2008, reducedSet],
                    type = "prob")
head(plsProbs)

set.seed(476)
plsFit2 <- train(x = training[, reducedSet],
                 y = training$Class,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)

plsImpGrant <- varImp(plsFit2, scale = FALSE)
plsImpGrant

plot(plsImpGrant, top = 20, scales = list(y = list(cex = .95)))

#Penalized Models

library(glmnet)
glmnetModel <- glmnet(x = as.matrix(training[,fullSet]),
                      y = training$Class,
                      family = "binomial")
## Compute predictions for three difference levels of regularization.
  ## Note that the results are not factors
predict(glmnetModel,
        newx = as.matrix(training[1:5,fullSet]),
        s = c(0.05, 0.1, 0.2),
        type = "class")

## Which predictors were used in the model?
predict(glmnetModel,
        newx = as.matrix(training[1:5,fullSet]),
        s = c(0.05, 0.1, 0.2),
        type = "nonzero")

## Specify the tuning values:
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 40))
set.seed(476)
glmnTuned <- train(training[,fullSet],
                   y = training$Class,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)

library(sparseLDA)
sparseLdaModel <- sda(x = as.matrix(training[,fullSet]),
                      y = training$Class,
                      lambda = 0.01,
                      stop = -6)

#Nearest Shrunken Centroids

## Switch dimensions using the t() function to transpose the data.
## This also implicitly converts the training data frame to a matrix.
inputData <- list(x = t(training[, fullSet]), y = training$Class)
library(pamr)
nscModel <- pamr.train(data = inputData)

exampleData <- t(training[1:5, fullSet])
pamr.predict(nscModel, newx = exampleData, threshold = 5)


## Which predictors were used at this threshold? The predict
## function shows the column numbers for the retained predictors.
thresh17Vars <- pamr.predict(nscModel, newx = exampleData,
                                 + threshold = 17, type = "nonzero")
fullSet[thresh17Vars]

## We chose the specific range of tuning parameters here:
nscGrid <- data.frame(.threshold = 0:25)
set.seed(476)
nscTuned <- train(x = training[,fullSet],
                  y = training$Class,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

predictors(nscTuned)
varImp(nscTuned, scale = FALSE)


#ch13

#Nonlinear Discriminant Analysis
library(mda)
mdaModel <- mda(Class ~ .,
                ## Reduce the data to the relevant predictors and the
                ## class variable to use the formula shortcut above
                data = training[pre2008, c("Class", reducedSet)],
                subclasses = 3)
mdaModel

predict(mdaModel,
        newdata = head(training[-pre2008, reducedSet]))


set.seed(476)
mdaFit <- train(training[,reducedSet], training$Class,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(.subclasses = 1:8),
                trControl = ctrl)

#Neural Networks

head(class.ind(training$Class))

set.seed(800)
nnetMod <- nnet(Class ~ NumCI + CI.1960,
                data = training[pre2008,],
                size = 3, decay = .1)

nnetMod

predict(nnetMod, newdata = head(testing))

predict(nnetMod, newdata = head(testing), type = "class")

nnetGrid <- expand.grid(.size = 1:10,
                        .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1)
set.seed(476)
nnetFit <- train(x = training[,reducedSet],
                 y = training$Class,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 ## ctrl was defined in the previous chapter
                 trControl = ctrl)

#Flexible Discriminant Analysis

library(mda)
library(earth)
fdaModel <- fda(Class ~ Day + NumCI, data = training[pre2008,],
                method = earth)

summary(fdaModel$fit)

predict(fdaModel, head(training[-pre2008,]))

#Support Vector Machines

class.weights = c(successful = 1, unsuccessful = 5)

set.seed(202)
sigmaRangeReduced <- sigest(as.matrix(training[,reducedSet]))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                                 + .C = 2^(seq(-4, 4)))
set.seed(476)
svmRModel <- train(training[,reducedSet], training$Class,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel

library(kernlab)
predict(svmRModel, newdata = head(training[-pre2008, reducedSet]))

predict(svmRModel, newdata = head(training[-pre2008, reducedSet]),
        type = "prob")

#K-Nearest Neighbors
set.seed(476)
knnFit <- train(training[,reducedSet], training$Class,
                method = "knn",
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = c(4*(0:5)+1,
                                             20*(1:5)+1,
                                             50*(2:9)+1)),
                trControl = ctrl)

knnFit$pred <- merge(knnFit$pred, knnFit$bestTune)
knnRoc <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$successful,
              levels = rev(levels(knnFit$pred$obs)))
plot(knnRoc, legacy.axes = TRUE)

#Na???ve Bayes

## Some predictors are already stored as factors
  factors <- c("SponsorCode", "ContractValueBand", "Month", "Weekday")
## Get the other predictors from the reduced set
  nbPredictors <- factorPredictors[factorPredictors %in% reducedSet]
nbPredictors <- c(nbPredictors, factors)
## Leek only those that are needed
  nbTraining <- training[, c("Class", nbPredictors)]
nbTesting <- testing[, c("Class", nbPredictors)]
## Loop through the predictors and convert some to factors
for(i in nbPredictors)
  {
  varLevels <- sort(unique(training[,i]))
  if(length(varLevels) <= 15)
    {
    nbTraining[, i] <- factor(nbTraining[,i],
                              levels = paste(varLevels))
    nbTesting[, i] <- factor(nbTesting[,i],
                             levels = paste(varLevels))
  }
  }


library(klaR)
nBayesFit <- NaiveBayes(Class ~ .,
                        data = nbTraining[pre2008,],
                        ## Should the non-parametric estimate
                        ## be used?
                        usekernel = TRUE,
                        ## Laplace correction value
                        fL = 2)
predict(nBayesFit, newdata = head(nbTesting))

#14
#Classification Trees

library(rpart)
cartModel <- rpart(factorForm, data = training[pre2008,])

rpart(Class ~ NumCI + Weekday, data = training[pre2008,])

library(RWeka)
J48(Class ~ NumCI + Weekday, data = training[pre2008,])

library(C50)
C5tree <- C5.0(Class ~ NumCI + Weekday, data = training[pre2008,])
C5tree


summary(C5tree)

set.seed(476)
rpartGrouped <- train(x = training[,factorPredictors],
                      y = training$Class,
                      method = "rpart",
                      tuneLength = 30,
                      metric = "ROC",
                      trControl = ctrl)

#Rules
PART(Class ~ NumCI + Weekday, data = training[pre2008,])

C5rules <- C5.0(Class ~ NumCI + Weekday, data = training[pre2008,],
                rules = TRUE)
C5rules

summary(C5rules)

#Bagged Trees
bagging(Class ~ Weekday + NumCI, data = training[pre2008,])

#Random Forest
library(randomForest)
randomForest(Class ~ NumCI + Weekday, data = training[pre2008,])

#Boosted Trees
library(gbm)
forGBM <- training
forGBM$Class <- ifelse(forGBM$Class == "successful", 1, 0)
gbmModel <- gbm(Class ~ NumCI + Weekday,
                data = forGBM[pre2008,],
                distribution = "bernoulli",
                interaction.depth = 9,
                n.trees = 1400,
                shrinkage = 0.01,
                ## The function produces copious amounts
                ## of output by default.
                verbose = FALSE)

gbmPred <- predict(gbmModel,
                   newdata = head(training[-pre2008,]),
                   type = "response",
                   ## The number of trees must be
                   ## explicitly set
                   n.trees = 1400)
gbmPred

gbmClass <- ifelse(gbmPred > .5, "successful", "unsuccessful")
gbmClass <- factor(gbmClass, levels = levels(training$Class))
gbmClass


library(C50)
C5Boost <- C5.0(Class ~ NumCI + Weekday, data = training[pre2008,],
                trials = 10)
C5Boost


#16
library(DWD) #???
data(ticdata)


recodeLevels <- function(x)
  {
  x <- as.numeric(x)
  ## Add zeros to the text version:
  x <- gsub(" ", "0",format(as.numeric(x)))
  factor(x)
  }
## Find which columns are regular factors or ordered factors
  isOrdered <- unlist(lapply(ticdata, is.ordered))
isFactor <- unlist(lapply(ticdata, is.factor))
convertCols <- names(isOrdered)[isOrdered | isFactor]
for(i in convertCols) ticdata[,i] <- recodeLevels(ticdata[,i])
## Make the level 'insurance' the first factor level
ticdata$CARAVAN <- factor(as.character(ticdata$CARAVAN),
                          levels = rev(levels(ticdata$CARAVAN)))


library(caret)
## First, split the training set off
set.seed(156)
split1 <- createDataPartition(ticdata$CARAVAN, p = .7)[[1]]
other <- ticdata[-split1,]
training <- ticdata[ split1,]
## Now create the evaluation and test sets
set.seed(934)
split2 <- createDataPartition(other$CARAVAN, p = 1/3)[[1]]
evaluation <- other[ split2,]
testing <- other[-split2,]
## Determine the predictor names
predictors <- names(training)[names(training) != "CARAVAN"]



## The first column is the intercept, which is eliminated:
trainingInd <- data.frame(model.matrix(CARAVAN ~ .,
                                       data = training))[,-1]
evaluationInd <- data.frame(model.matrix(CARAVAN ~ .,
                                         data = evaluation))[,-1]
testingInd <- data.frame(model.matrix(CARAVAN ~ .,
                                      data = testing))[,-1]
## Add the outcome back into the data set
trainingInd$CARAVAN <- training$CARAVAN
evaluationInd$CARAVAN <- evaluation$CARAVAN
testingInd$CARAVAN <- testing$CARAVAN
## Determine a predictor set without highly sparse and unbalanced
distributions:
isNZV <- nearZeroVar(trainingInd)
noNZVSet <- names(trainingInd)[-isNZV]

## For accuracy, Kappa, the area under the ROC curve,
## sensitivity and specificity:
fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))
## Everything but the area under the ROC curve:
  fourStats <- function (data, lev = levels(data$obs), model = NULL)
    {
    accKapp <- postResample(data[, "pred"], data[, "obs"])
    out <- c(accKapp,
             sensitivity(data[, "pred"], data[, "obs"], lev[1]),
             specificity(data[, "pred"], data[, "obs"], lev[2]))
    names(out)[3:4] <- c("Sens", "Spec")
    out
    }


ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     verboseIter = TRUE)
ctrlNoProb <- ctrl
ctrlNoProb$summaryFunction <- fourStats
ctrlNoProb$classProbs <- FALSE

set.seed(1410)
rfFit <- train(CARAVAN ~ ., data = trainingInd,
               method = "rf",
               trControl = ctrl,
               ntree = 1500,
               tuneLength = 5,
               metric = "ROC")
set.seed(1410)
lrFit <- train(CARAVAN ~ .,
               data = trainingInd[, noNZVSet],
               method = "glm",
               trControl = ctrl,
               metric = "ROC")
set.seed(1401)
fdaFit <- train(CARAVAN ~ ., data = training,
                method = "fda",
                tuneGrid = data.frame(.degree = 1, .nprune = 1:25),
                metric = "ROC",
                trControl = ctrl)

evalResults <- data.frame(CARAVAN = evaluation$CARAVAN)
evalResults$RF <- predict(rfFit,
                          newdata = evaluationInd,
                          type = "prob")[,1]

evalResults$FDA <- predict(fdaFit,
                           newdata = evaluation[, predictors],
                           type = "prob")[,1]
evalResults$LogReg <- predict(lrFit,
                              newdata = valuationInd[, noNZVSet],
                              type = "prob")[,1]

library(pROC)
rfROC <- roc(evalResults$CARAVAN, evalResults$RF,
             levels = rev(levels(evalResults$CARAVAN)))
## Create labels for the models:
labs <- c(RF = "Random Forest", LogReg = "Logistic Regression",
          FDA = "FDA (MARS)")
lift1 <- lift(CARAVAN ~ RF + LogReg + FDA, data = evalResults,
              labels = labs)
rfROC

lift1

plot(rfROC, legacy.axes = TRUE)
xyplot(lift1,
       ylab = "%Events Found", xlab = "%Customers Evaluated",
       lwd = 2, type = "l")

#Alternate Cutoffs

rfThresh <- coords(rfROC, x = "best", best.method = "closest.topleft")
rfThresh

newValue <- factor(ifelse(evalResults$RF > rfThresh,
                          "insurance", "noinsurance"),
                   levels = levels(evalResults$CARAVAN))

#Sampling Methods

set.seed(1103)
upSampledTrain <- upSample(x = training[,predictors],
                           y = training$CARAVAN,
                           ## keep the class variable name the same:
                           yname = "CARAVAN")
dim(training)
dim(upSampledTrain)
table(upSampledTrain$CARAVAN)

library(DMwR) #???
set.seed(1103)
smoteTrain <- SMOTE(CARAVAN ~ ., data = training)
dim(smoteTrain)

table(smoteTrain$CARAVAN)


#Cost-Sensitive Training

library(kernlab)
## We will train over a large cost range, so we precompute the sigma
## parameter and make a custom tuning grid:
set.seed(1157)
sigma <- sigest(CARAVAN ~ ., data = trainingInd[, noNZVSet], frac = .75)
names(sigma) <- NULL
svmGrid <- data.frame(.sigma = sigma[2],
                        + .C = 2^seq(-6, 1, length = 15))
## Class probabilities cannot be generated with class weights, so
## use the control object 'ctrlNoProb' to avoid estimating the
## ROC curve.
set.seed(1401)
SVMwts <- train(CARAVAN ~ .,
                data = trainingInd[, noNZVSet],
                method = "svmRadial",
                tuneGrid = svmGrid,
                preProc = c("center", "scale"),
                class.weights = c(insurance = 18, noinsurance = 1),
                metric = "Sens",
                trControl = ctrlNoProb)
SVMwts

costMatrix <- matrix(c(0, 1, 20, 0), ncol = 2)
rownames(costMatrix) <- levels(training$CARAVAN)
colnames(costMatrix) <- levels(training$CARAVAN)
costMatrix

library(rpart)
set.seed(1401)
cartCosts <- train(x = training[,predictors],
                   y = training$CARAVAN,
                   method = "rpart",
                   trControl = ctrlNoProb,
                   metric = "Kappa",
                   tuneLength = 10,
                   parms = list(loss = costMatrix))
                     
c5Matrix <- matrix(c(0, 20, 1, 0), ncol = 2)
rownames(c5Matrix) <- levels(training$CARAVAN)
colnames(c5Matrix) <- levels(training$CARAVAN)
c5Matrix


library(C50)
set.seed(1401)
C5Cost <- train(x = training[, predictors],
                y = training$CARAVAN,
                method = "C5.0",
                metric = "Kappa",
                cost = c5Matrix,
                trControl = ctrlNoProb)

#17
library(AppliedPredictiveModeling)
data(HPC)
set.seed(1104)
inTrain <- createDataPartition(schedulingData$Class,
                               p = .8,
                               list = FALSE)
schedulingData$NumPending <- schedulingData$NumPending + 1
trainData <- schedulingData[ inTrain,]
testData <- schedulingData[-inTrain,]

cost <- function(pred, obs)
  {
  isNA <- is.na(pred)
  if(!all(isNA))
    {
    pred <- pred[!isNA]
    obs <- obs[!isNA]
    cost <- ifelse(pred == obs, 0, 1)
    if(any(pred == "VF" & obs == "L"))
      cost[pred == "L" & obs == "VF"] <- 10
    if(any(pred == "F" & obs == "L"))
      cost[pred == "F" & obs == "L"] <- 5
    if(any(pred == "F" & obs == "M"))
      cost[pred == "F" & obs == "M"] <- 5
    if(any(pred == "VF" & obs == "M"))
      cost[pred == "VF" & obs == "M"] <- 5
    out <- mean(cost)
    } else out <- NA
    out
    }
costSummary <- function (data, lev = NULL, model = NULL)
  {
  if (is.character(data$obs)) data$obs <- factor(data$obs,
                                                     levels = lev)
  c(postResample(data[, "pred"], data[, "obs"]),
    Cost = cost(data[, "pred"], data[, "obs"]))
  }

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     summaryFunction = costSummary)

costMatrix <- ifelse(diag(4) == 1, 0, 1)
costMatrix[1,4] <- 10
costMatrix[1,3] <- 5
costMatrix[2,4] <- 5
costMatrix[2,3] <- 5
rownames(costMatrix) <- levels(trainData$Class)
colnames(costMatrix) <- levels(trainData$Class)
costMatrix

modForm <- as.formula(Class ~ Protocol + log10(Compounds) +
                        log10(InputFields)+ log10(Iterations) +
                        log10(NumPending) + Hour + Day)

## Cost-Sensitive CART
  set.seed(857)
rpFitCost <- train(x = trainData[, predictors],
                   y = trainData$Class,
                   method = "rpart",
                   metric = "Cost",
                   maximize = FALSE,
                   tuneLength = 20,
                   ## rpart structures the cost matrix so that
                   ## the true classes are in rows, so we
                   ## transpose the cost matrix
                   parms =list(loss = t(costMatrix)),
                   trControl = ctrl)
## Cost- Sensitive C5.0
set.seed(857)
c50Cost <- train(x = trainData[, predictors],
                 y = trainData$Class,
                 method = "C5.0",
                 metric = "Cost",
                 maximize = FALSE,
                 costs = costMatrix,
                 tuneGrid = expand.grid(.trials = c(1, (1:10)*10),
                                        .model = "tree",
                                        .winnow = c(TRUE, FALSE)),
                 trControl = ctrl)
## Cost-Sensitive bagged trees
rpCost <- function(x, y)
  {
  costMatrix <- ifelse(diag(4) == 1, 0, 1)
  costMatrix[4, 1] <- 10
  costMatrix[3, 1] <- 5
  costMatrix[4, 2] <- 5
  costMatrix[3, 2] <- 5
  library(rpart)
  tmp <- x
  tmp$y <- y
  rpart(y~.,
        data = tmp,
        control = rpart.control(cp = 0),
        parms = list(loss = costMatrix))
  }
rpPredict <- function(object, x) predict(object, x)
rpAgg <- function (x, type = "class")
  {
  pooled <- x[[1]] * NA
  n <- nrow(pooled)
  classes <- colnames(pooled)
  for (i in 1:ncol(pooled))
    {
    tmp <- lapply(x, function(y, col) y[, col], col = i)
    tmp <- do.call("rbind", tmp)
    pooled[, i] <- apply(tmp, 2, median)
    }
  pooled <- apply(pooled, 1, function(x) x/sum(x))
  if (n != nrow(pooled)) pooled <- t(pooled)
  out <- factor(classes[apply(pooled, 1, which.max)],
                levels = classes)
  out
  }
set.seed(857)
rpCostBag <- train(trainData[, predictors],
                   trainData$Class,
                   "bag",
                   B = 50,
                   bagControl = bagControl(fit = rpCost,
                                           predict = rpPredict,
                                           aggregate = rpAgg,
                                           downSample = FALSE),
                   trControl = ctrl)
## Weighted SVM
set.seed(857)
svmRFitCost <- train(modForm, data = trainData,
                     method = "svmRadial",
                     metric = "Cost",
                     maximize = FALSE,
                     preProc = c("center", "scale"),
                     class.weights = c(VF = 1, F = 1,
                                       M = 5, L = 10),
                     tuneLength = 15,
                     trControl = ctrl)

confusionMatrix(rpFitCost, norm = "none")

#18
#Numeric Outcomes
library(AppliedPredictiveModeling)
data(solubility)
cor(solTrainXtrans$NumCarbon, solTrainY)

## Determine which columns have the string "FP" in the name and
## exclude these to get the numeric predictors
fpCols<- grepl("FP", names(solTrainXtrans))
## Exclude these to get the numeric predictor names
numericPreds <- names(solTrainXtrans)[!fpCols]
corrValues <- apply(solTrainXtrans[, numericPreds],
                    MARGIN = 2,
                    FUN = function(x, y) cor(x, y),
                    y = solTrainY)
head(corrValues)

smoother <- loess(solTrainY ~ solTrainXtrans$NumCarbon)
smoother

xyplot(solTrainY ~ solTrainXtrans$NumCarbon,
       type = c("p", "smooth"),
       xlab = "# Carbons",
       ylab = "Solubility")

loessResults <- filterVarImp(x = solTrainXtrans[, numericPreds],
                             y = solTrainY,
                             nonpara = TRUE)
head(loessResults)

library(minerva)
micValues <- mine(solTrainXtrans[, numericPreds], solTrainY)
## Several statistics are calculated
names(micValues)

head(micValues$MIC)

t.test(solTrainY ~ solTrainXtrans$FP044)

getTstats <- function(x, y)
  {
  tTest <- t.test(y~x)
  out <- c(tStat = tTest$statistic, p = tTest$p.value)
  out
  }
tVals <- apply(solTrainXtrans[, fpCols],
               MARGIN = 2,
               FUN = getTstats,
               y = solTrainY)
## switch the dimensions
tVals <- t(tVals)
head(tVals)

#Categorical Outcomes
library(caret)
data(segmentationData)
cellData <- subset(segmentationData, Case == "Train")
cellData$Case <- cellData$Cell <- NULL
## The class is in the first column
head(names(cellData))

rocValues <- filterVarImp(x = cellData[,-1],
                            + y = cellData$Class)
## Column is created for each class
head(rocValues)

library(CORElearn)
reliefValues <- attrEval(Class ~ ., data = cellData,
                         ## There are many Relief methods
                         ## available. See ?attrEval
                         estimator = "ReliefFequalK",
                         ## The number of instances tested:
                         ReliefIterations = 50)
head(reliefValues)

perm <- permuteRelief(x = cellData[,-1],
                      y = cellData$Class,
                      nperm = 500,
                      estimator = "ReliefFequalK",
                      ReliefIterations = 50)

head(perm$permutations)

histogram(~ value|Predictor,
          data = perm$permutations)

head(perm$standardized)

micValues <- mine(x = cellData[,-1],
                  y = ifelse(cellData$Class == "PS", 1, 0))
head(micValues$MIC)

Sp62BTable <- table(training[pre2008, "Sponsor62B"],
                    training[pre2008, "Class"])
Sp62BTable

fisher.test(Sp62BTable)


ciTable <- table(training[pre2008, "CI.1950"],
                 training[pre2008, "Class"])
ciTable
fisher.test(ciTable)

DayTable <- table(training[pre2008, "Weekday"],
                    + training[pre2008, "Class"])
DayTable

chisq.test(DayTable)

#Model-Based Importance Scores
library(randomForest)
set.seed(791)
rfImp <- randomForest(Class ~ ., data = segTrain,
                      ntree = 2000,
                      importance = TRUE)

head(varImp(rfImp))



#19
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
## Manually create new dummy variables
predictors$E2 <- predictors$E3 <- predictors$E4 <- 0
predictors$E2[grepl("2", predictors$Genotype)] <- 1
predictors$E3[grepl("3", predictors$Genotype)] <- 1
predictors$E4[grepl("4", predictors$Genotype)] <- 1

## Split the data using stratified sampling
  set.seed(730)
split <- createDataPartition(diagnosis, p = .8, list = FALSE)
## Combine into one data frame
adData <- predictors
adData$Class <- diagnosis
training <- adData[ split, ]
testing <- adData[-split, ]
## Save a vector of predictor variable names
predVars <- names(adData)[!(names(adData) %in% c("Class", "Genotype"))]
## Compute the area under the ROC curve, sensitivity, specificity,
## accuracy and Kappa
fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))
## Create resampling data sets to use for all models
set.seed(104)
index <- createMultiFolds(training$Class, times = 5)
## Create a vector of subset sizes to evaluate
varSeq <- seq(1, length(predVars)-1, by = 2)

#Forward, Backward, and Stepwise Selection

initial <- glm(Class ~ tau + VEGF + E4 + IL_3, data = training,
               family = binomial)
library(MASS)
stepAIC(initial, direction = "both")

#Recursive Feature Elimination

library(caret)
## The built-in random forest functions are in rfFuncs.
str(rfFuncs)

newRF <- rfFuncs
newRF$summary <- fiveStats

## The control function is similar to trainControl():
ctrl <- rfeControl(method = "repeatedcv",
                   repeats = 5,
                   verbose = TRUE,
                   functions = newRF,
                   index = index)
set.seed(721)
rfRFE <- rfe(x = training[, predVars],
             y = training$Class,
             sizes = varSeq,
             metric = "ROC",
             rfeControl = ctrl,
             ## now pass options to randomForest()
             ntree = 1000)
rfRFE

predict(rfRFE, head(testing))

svmFuncs <- caretFuncs
svmFuncs$summary <- fivestats
ctrl <- rfeControl(method = "repeatedcv",
                   repeats = 5,
                   verbose = TRUE,
                   functions = svmFuncs,
                   index = index)
set.seed(721)
svmRFE <- rfe(x = training[, predVars],
              y = training$Class,
              sizes = varSeq,
              metric = "ROC",
              rfeControl = ctrl,
              ## Now options to train()
              method = "svmRadial",
              tuneLength = 12,
              preProc = c("center", "scale"),
              ## Below specifies the inner resampling process
              trControl = trainControl(method = "cv",
                                       verboseIter = FALSE,
                                       classProbs = TRUE))
svmRFE


#Filter Methods
pScore <- function(x, y)
  {
  numX <- length(unique(x))
  if(numX > 2)
    {
    ## With many values in x, compute a t-test
    out <- t.test(x ~ y)$p.value
    } else {
      ## For binary predictors, test the odds ratio == 1 via
      ## Fisher's Exact Test
      out <- fisher.test(factor(x), y)$p.value
      }
  out
  }
## Apply the scores to each of the predictor columns
scores <- apply(X = training[, predVars],
                MARGIN = 2,
                FUN = pScore,
                y = training$Class)
tail(scores)

pCorrection <- function (score, x, y)
  {
  ## The options x and y are required by the caret package
  ## but are not used here
  score <- p.adjust(score, "bonferroni")
  ## Return a logical vector to decide which predictors
  ## to retain after the filter
  keepers <- (score <= 0.05)
  keepers
  }
tail(pCorrection(scores))

str(ldaSBF)

ldaWithPvalues <- ldaSBF
ldaWithPvalues$score <- pScore
ldaWithPvalues$summary <- fiveStats
ldaWithPvalues$filter <- pCorrection
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = ldaWithPvalues,
                      index = index)
ldaFilter <- sbf(training[, predVars],
                 training$Class,
                 tol = 1.0e-12,
                 sbfControl = sbfCtrl)
ldaFilter

#20

library(AppliedPredictiveModeling)
data(solubility)
set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

set.seed(100)
mtryVals <- floor(seq(10, ncol(solTrainXtrans), length = 10))
mtryGrid <- data.frame(.mtry = mtryVals)
rfTune <- train(x = solTrainXtrans, y = solTrainY,
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 1000,
                importance = TRUE,
                trControl = ctrl)
ImportanceOrder <- order(rfTune$finalModel$importance[,1],
                         decreasing = TRUE)
top20 <- rownames(rfTune$finalModel$importance[ImportanceOrder,])[1:20]
solTrainXimp <- subset(solTrainX, select = top20)
solTestXimp <- subset(solTestX, select = top20)

permutesolTrainXimp <- apply(solTrainXimp, 2, function(x) sample(x))
solSimX <- rbind(solTrainXimp, permutesolTrainXimp)
groupVals <- c("Training", "Random")
groupY <- factor(rep(groupVals, each = nrow(solTrainX)))

rfSolClass <- train(x = solSimX, y = groupY,
                    method = "rf",
                    tuneLength = 5,
                    ntree = 1000,
                    control = trainControl(method = "LGOCV"))
solTestGroupProbs <- predict(rfSolClass, solTestXimp, type = "prob")
