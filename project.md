---
title: "Prediction Assignment Writeup"
author: "Gustavo García"
date: "23/11/2020"
output: 
  html_document:
    keep_md: true
    self_contained: true
---



## 1. Overview

The goal of this project is to predict the manner in which 6 participants did exercise using data from accelerometers on the belt, forearm, arm, and dumbell. This is the "classe" variable in the training set. The machine learning algorithm described here is applied to predict 20 different test cases.

## 2. Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## 3. Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## 4. Exploratory Analysis

First, we load some useful packages using:


```r
# Require the necessary packages
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.0.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

Then,we begin by exporting the data. One can simply download the training and testing datasets using:


```r
# Download the training and testing datasets

trainFile<-"traininig.csv"
testFile<-"testing.csv"

if(!file.exists(trainFile))
{
   download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",trainFile,method = "curl")
}
if(!file.exists(testFile))
{
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",testFile,method = "curl")
}
```

Now we load the training and testing datasets:


```r
training <- read.csv(trainFile)
testing  <- read.csv(testFile)
```

Now we split the training data into two parts. We'll use 70% of this data to actually train our model and the remaining 30% to validate it:


```r
set.seed(12435)
inTrain <- createDataPartition( y = training$classe,
                                   p = 0.7,
                                   list = FALSE)
trainSet <- training[inTrain,]
validationSet <- training[-inTrain,]
```


```r
dim(trainSet)
```

```
## [1] 13737   160
```

```r
dim(validationSet)
```

```
## [1] 5885  160
```

We can see that the created datasets have 160 variables. Let’s clean NA, The Near Zero variance (NZV) variables and the ID variables as well.


```r
# clean-up the Near Zero variance (NZV) variables

nzv <- nearZeroVar(trainSet)
trainSet <- trainSet[,-nzv]
validationSet <- validationSet[,-nzv]

# Remove variables that are mostly NA
mostlyNA <- sapply(trainSet,function(x) mean(is.na(x))) > 0.95
trainSet <- trainSet[,mostlyNA==FALSE]
validationSet <- validationSet[,mostlyNA==FALSE]

# The first 5 variables are identifiers that are
# not probably useful for prediction so we remove them
trainSet <- trainSet[,-(1:5)]
validationSet <- validationSet[,-(1:5)]
```


```r
dim(trainSet)
```

```
## [1] 13737    54
```

```r
dim(validationSet)
```

```
## [1] 5885   54
```

After cleaning, we can see that the number of variables for the analysis are now only 54(53 for prediction).

## 5. Prediction Model Building

We'll build four models: Linear Discriminant Analysis (LDA),  k-Nearest Neighbors (kNN), a random forest and a generalized boosted model. We'll train these in the training portion of the original training dataset and then test them in the validation portion of the original training dataset:


```r
# Build a LDA model
set.seed(12435)
fitLDA <- train(classe~., data=trainSet, method="lda", trControl=trainControl(method="cv", number=10), metric="Accuracy")

# Build a knn model
set.seed(12435)
fitKNN <- train(classe~., data=trainSet, method="knn", trControl= trainControl(method="cv", number=10), metric= "Accuracy")

set.seed(12435)
# Now let's build a random forest model
fitRF  <- train( classe ~.,
                   data = trainSet,
                   method = "rf",
                   trControl = trainControl(method="cv",number=3) )
set.seed(12435)
# Build a generalized boosted model
fitGBM <- train( classe ~.,
                  data = trainSet,
                  method = "gbm",
                  trControl = trainControl(method="repeatedcv",number = 5,repeats = 1),
                  verbose = FALSE)
```

Then let's see how well these four models perform predicting the values in the validation dataset.


```r
predictLDA <- predict(fitLDA, newdata=validationSet)
conf_matrix_LDA <- confusionMatrix(predictLDA, as.factor(validationSet$classe))
print(conf_matrix_LDA)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1404  165   96   49   51
##          B   34  718   90   40  173
##          C  119  163  680  114  101
##          D  110   52  143  721   91
##          E    7   41   17   40  666
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7118          
##                  95% CI : (0.7001, 0.7234)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6352          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8387   0.6304   0.6628   0.7479   0.6155
## Specificity            0.9143   0.9290   0.8977   0.9195   0.9781
## Pos Pred Value         0.7955   0.6806   0.5777   0.6455   0.8638
## Neg Pred Value         0.9345   0.9128   0.9265   0.9490   0.9187
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2386   0.1220   0.1155   0.1225   0.1132
## Detection Prevalence   0.2999   0.1793   0.2000   0.1898   0.1310
## Balanced Accuracy      0.8765   0.7797   0.7802   0.8337   0.7968
```


```r
predictKNN <- predict(fitKNN, newdata=validationSet)
conf_matrix_KNN <- confusionMatrix(predictKNN, as.factor(validationSet$classe))
print(conf_matrix_KNN)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1620   51   11   22   21
##          B   15  999   29    5   21
##          C   18   43  946   65   19
##          D   18   32   28  854   35
##          E    3   14   12   18  986
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9184          
##                  95% CI : (0.9111, 0.9253)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8967          
##                                           
##  Mcnemar's Test P-Value : 3.491e-13       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9677   0.8771   0.9220   0.8859   0.9113
## Specificity            0.9751   0.9853   0.9702   0.9770   0.9902
## Pos Pred Value         0.9391   0.9345   0.8671   0.8831   0.9545
## Neg Pred Value         0.9870   0.9709   0.9833   0.9776   0.9802
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2753   0.1698   0.1607   0.1451   0.1675
## Detection Prevalence   0.2931   0.1816   0.1854   0.1643   0.1755
## Balanced Accuracy      0.9714   0.9312   0.9461   0.9315   0.9507
```


```r
predictRF <- predict(fitRF,validationSet)
conf_matrix_RF <- confusionMatrix(predictRF,as.factor(validationSet$classe))
print(conf_matrix_RF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    3    0    0    0
##          B    0 1136    3    0    0
##          C    0    0 1023    2    0
##          D    0    0    0  962    1
##          E    1    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9969, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9971   0.9979   0.9991
## Specificity            0.9993   0.9994   0.9996   0.9998   0.9998
## Pos Pred Value         0.9982   0.9974   0.9980   0.9990   0.9991
## Neg Pred Value         0.9998   0.9994   0.9994   0.9996   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1738   0.1635   0.1837
## Detection Prevalence   0.2848   0.1935   0.1742   0.1636   0.1839
## Balanced Accuracy      0.9993   0.9984   0.9983   0.9989   0.9994
```


```r
predictGBM <- predict(fitGBM,validationSet)
conf_matrixGBM <- confusionMatrix(predictGBM,as.factor(validationSet$classe))
print(conf_matrixGBM)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   15    0    0    0
##          B    4 1111    7    4    4
##          C    0   13 1016    7    1
##          D    0    0    3  952    9
##          E    0    0    0    1 1068
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9884         
##                  95% CI : (0.9854, 0.991)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9854         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9754   0.9903   0.9876   0.9871
## Specificity            0.9964   0.9960   0.9957   0.9976   0.9998
## Pos Pred Value         0.9911   0.9832   0.9797   0.9876   0.9991
## Neg Pred Value         0.9990   0.9941   0.9979   0.9976   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1888   0.1726   0.1618   0.1815
## Detection Prevalence   0.2863   0.1920   0.1762   0.1638   0.1816
## Balanced Accuracy      0.9970   0.9857   0.9930   0.9926   0.9934
```

## 6. Applying the selected Model to the Test Data

We see the random forest has better performance (Accuracy : 0.9983) than the generalized boosted model (Accuracy : 0.9884), LDA (Accuracy: 0.7118 ) and kNN (Accuracy: 0.9183). Let's test our model in the actual testing dataset:


```r
predict_testRF <- predict(fitRF,testing)
print(predict_testRF)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
