## Practical Machine Learning Course Project
============================================================
### Michael Shipman - June 22, 2014

## Data Sourcing and Processing
The training and testing datasets were loaded directly from the cloudfront.net URL given in the assignement instructions.

```r
sourcetrain <- read.csv("pml-training.csv", header = TRUE)
sourcetest <- read.csv("pml-testing.csv", header = TRUE)
```
The structure of the source (raw) datasets were observed to have many statistical summary variables that would not contribute well to the prediction algorithms, such as min, max, average, kertosis, skewness, etc. The source training and testing data sets were conditioned by removing the statistical summary variables and checked to remove any rows with "NA" data entries. Further conditioning is performed to remove the "raw timestamps" part 1 and 3, This will produce 'tidy' datasets to work with while fitting and testing the data.

```r
#Create a matching vector of character strings that pertain to summary statistics phases.
match <- c("kurtosis", "skewness", "max", "min", "amplitude", "var", "avg", "stddev")
#Create a vector of variables to remove that contain summary statistics. 
removeVar <- names(sourcetrain)[grep(paste(match, collapse="|"), names(sourcetrain))]
#Remove the statistical summary variables from the source training and testing datasets.
train <- sourcetrain[ ,!(names(sourcetrain) %in% removeVar)]
test <- sourcetest[ ,!(names(sourcetest) %in% removeVar)]
#Ensure that all the data rows are complete (w/o "NA")
train <- train[complete.cases(train), ]
test <- test[complete.cases(test), ]
```
The training dataset was split into 75% / 25% training and validation dataset to fine tune the model.

```r
set.seed(28462)
split <- sample(1:dim(train)[1], 
                size=dim(train)*0.75,
                replace=F)
train <- train[split,]
val <- train[-split,]
```

##Selection and Training the Machine Learning Algorith Using Training Data
When the data sets were conditioned into 'tidy' format, the next step was to select a model to fit to the training data set. The output variable is a categorical variable listed from A to E corresponding to the excersize type being performed while the input variables were being measured. A Characterization and Regression Tree (CART) model was chosen based on the muliple discrete outcomes of the output variable.  The CART model will produce a decision tree based on the input variables run through the algorithm. The 'rpart' algorithm in the Caret package was used as the first model to fit.  A random number generator seed of 28462 was chosen and reset each time a new training algorithm was begun.

```r
library(caret)
set.seed(28462)
#Calls the rpart model and fits to the conditioned training dataset.
fit <- rpart(classe ~ .,
               method = "class",
               data = train)
```
The descision tree final model is shown in the output and graphics below.

```r
#displays model fitting
fit
```

```
## n= 14716 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 14716 10540 A (0.28 0.19 0.17 0.17 0.18)  
##    2) X< 5580 4181     0 A (1 0 0 0 0) *
##    3) X>=5580 10535  7696 B (0 0.27 0.24 0.23 0.26)  
##      6) X< 9378 2839     0 B (0 1 0 0 0) *
##      7) X>=9378 7696  5001 E (0 0 0.33 0.32 0.35)  
##       14) X< 1.602e+04 5001  2442 C (0 0 0.51 0.49 0)  
##         28) X< 1.28e+04 2559     0 C (0 0 1 0 0) *
##         29) X>=1.28e+04 2442     0 D (0 0 0 1 0) *
##       15) X>=1.602e+04 2695     0 E (0 0 0 0 1) *
```

```r
#displays the decision tree model graphically.
library(rattle)
fancyRpartPlot(fit, main = "Fit Using rpart Algorithm on Training Dataset")
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 

##Predicting the Excersize Outcome Using Validation Data
The CART model decision tree is then fit to the training dataset to see the accuracy of the model.  A confusion matrix is set up to show the accuracy of the predicted vs. the actual excersize class.

```r
#Create a column of predicted values in the validation data set.
val$Pred <- predict(fit, newdata = val, type = "class")
#Produce a comparison using confusion matrix function.
confusionMatrix(val$Pred, val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1065    0    0    0    0
##          B    0  729    0    0    0
##          C    0    0  622    0    0
##          D    0    0    0  619    0
##          E    0    0    0    0  655
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.289     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.289    0.198    0.169    0.168    0.178
## Detection Rate          0.289    0.198    0.169    0.168    0.178
## Detection Prevalence    0.289    0.198    0.169    0.168    0.178
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

##Cross Validation By Training the Model with Validation Data
With an accuracy of 100% and Kappa = 1, the model shows that there is a very good chance of overfitting using the 'rpart' algorithm.  A cross validation was performed using the validation data set then checking the results using the training dataset.

```r
set.seed(28462)
#Remove the Pred column from the val dataset before training.
val <- val[-61]
#Calls the rpart model and fits to the conditioned training dataset.
fit2 <- rpart(classe ~ .,
               method = "class",
               data = val)
```
The second CART descision tree final model is shown in the output and graphics below.

```r
#displays model fitting
fit2
```

```
## n= 3690 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 3690 2625 A (0.29 0.2 0.17 0.17 0.18)  
##    2) X< 5580 1065    0 A (1 0 0 0 0) *
##    3) X>=5580 2625 1896 B (0 0.28 0.24 0.24 0.25)  
##      6) X< 9378 729    0 B (0 1 0 0 0) *
##      7) X>=9378 1896 1241 E (0 0 0.33 0.33 0.35)  
##       14) X< 1.602e+04 1241  619 C (0 0 0.5 0.5 0)  
##         28) X< 1.28e+04 622    0 C (0 0 1 0 0) *
##         29) X>=1.28e+04 619    0 D (0 0 0 1 0) *
##       15) X>=1.602e+04 655    0 E (0 0 0 0 1) *
```

```r
#displays the decision tree model graphically.
fancyRpartPlot(fit2, main = "Fit2 Using rpart Algorithm on Validation Dataset")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8.png) 
The CART model decision tree is then fit to the training dataset to see the accuracy of the model.  A confusion matrix is set up to show the accuracy of the predicted vs. the actual excersize class.

```r
#Create a column of predicted values in the training data set.
train$Pred <- predict(fit2 , newdata = train, type = "class")
#Produce a comparison using confusion matrix function.
confusionMatrix(train$Pred, train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4180    0    0    0    0
##          B    1 2839    0    0    0
##          C    0    0 2558    0    0
##          D    0    0    1 2442    2
##          E    0    0    0    0 2693
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    0.999
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    0.999    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.166    0.183
## Detection Rate          0.284    0.193    0.174    0.166    0.183
## Detection Prevalence    0.284    0.193    0.174    0.166    0.183
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
The accuracy is still very close to 100% given the cross validation of the validation data fit to model.

##Predicting the Test Outcomes Using the Model.
The prediction of the test dataset outcomes was fit to the model and shown in the final table

```r
#Create a column of predicted values in the testing data set.
test$Pred <- predict(fit, newdata = test, type = "class")
#Produce a table showing the outcomes of the 20 test cases.
subset(test, select = c(user_name, Pred))
```

```
##    user_name Pred
## 1      pedro    A
## 2     jeremy    A
## 3     jeremy    A
## 4     adelmo    A
## 5     eurico    A
## 6     jeremy    A
## 7     jeremy    A
## 8     jeremy    A
## 9   carlitos    A
## 10   charles    A
## 11  carlitos    A
## 12    jeremy    A
## 13    eurico    A
## 14    jeremy    A
## 15    jeremy    A
## 16    eurico    A
## 17     pedro    A
## 18  carlitos    A
## 19     pedro    A
## 20    eurico    A
```

