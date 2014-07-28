## Practical Machine Learning Course Project
============================================================
### Michael Shipman - July 27, 2014

## Data Sourcing and Processing
The training and testing datasets were loaded directly from the cloudfront.net URL given in the assignement instructions.

```r
sourcetrain <- read.csv("pml-training.csv", header = TRUE)
sourcetest <- read.csv("pml-testing.csv", header = TRUE)
```
The structure of the source (raw) datasets were observed to have many statistical summary variables that would not contribute well to the prediction algorithms, such as min, max, average, kertosis, skewness, etc. The source training and testing data sets were conditioned by removing the statistical summary variables and checked to remove any rows with "NA" data entries. Further conditioning is performed to remove the "raw timestamps" part 1, part 3, and the observation number "X" column. This will produce 'tidy' datasets to work with while fitting and testing the data.

```r
#Create a matching vector of character strings that pertain to summary statistics phases.
match <- c("kurtosis", "skewness", "max", "min", "amplitude", "var", "avg", "stddev")
#Create a vector of variables to remove that contain summary statistics. 
removeVar <- names(sourcetrain)[grep(paste(match, collapse="|"), names(sourcetrain))]
#add the "X" column to the columns needed to remove
removeVar <- c(removeVar, "X")
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
When the data sets were conditioned into 'tidy' format, the next step was to select a model to fit to the training data set. The output variable is a categorical variable listed from A to E corresponding to the excersize type being performed while the input variables were being measured. A Characterization and Regression Tree (CART) model was chosen based on the muliple discrete outcomes of the output variable. The CART model will produce a decision tree based on the input variables run through the algorithm. The 'rpart' algorithm in the Caret package was used as the first model to fit. A random number generator seed of 28462 was chosen and reset each time a new training algorithm was begun.

```r
library(caret)
library(rpart)
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
##    2) cvtd_timestamp=02/12/2011 13:32,02/12/2011 13:33,02/12/2011 13:34,02/12/2011 14:56,02/12/2011 14:57,05/12/2011 11:23,05/12/2011 11:24,05/12/2011 14:22,05/12/2011 14:23,28/11/2011 14:13,28/11/2011 14:14,30/11/2011 17:10,30/11/2011 17:11 9325  5144 A (0.45 0.3 0.2 0.046 0)  
##      4) cvtd_timestamp=02/12/2011 13:32,02/12/2011 13:33,02/12/2011 14:56,05/12/2011 11:23,05/12/2011 14:22,28/11/2011 14:13,30/11/2011 17:10 2905   261 A (0.91 0.09 0 0 0)  
##        8) raw_timestamp_part_1< 1.323e+09 2132     0 A (1 0 0 0 0) *
##        9) raw_timestamp_part_1>=1.323e+09 773   261 A (0.66 0.34 0 0 0)  
##         18) user_name=carlitos,charles,pedro 507     0 A (1 0 0 0 0) *
##         19) user_name=adelmo 266     5 B (0.019 0.98 0 0 0) *
##      5) cvtd_timestamp=02/12/2011 13:34,02/12/2011 14:57,05/12/2011 11:24,05/12/2011 14:23,28/11/2011 14:14,30/11/2011 17:11 6420  3857 B (0.24 0.4 0.3 0.066 0)  
##       10) magnet_dumbbell_z< -0.5 3189  1704 A (0.47 0.41 0.13 0.0016 0)  
##         20) raw_timestamp_part_1< 1.323e+09 2817  1332 A (0.53 0.46 0.012 0.0018 0)  
##           40) magnet_dumbbell_x< -455.5 1529   353 A (0.77 0.21 0.014 0.0033 0)  
##             80) raw_timestamp_part_1< 1.323e+09 1305   129 A (0.9 0.084 0.011 0.0038 0) *
##             81) raw_timestamp_part_1>=1.323e+09 224     7 B (0 0.97 0.031 0 0) *
##           41) magnet_dumbbell_x>=-455.5 1288   321 B (0.24 0.75 0.0093 0 0)  
##             82) num_window< 68.5 215     0 A (1 0 0 0 0) *
##             83) num_window>=68.5 1073   106 B (0.088 0.9 0.011 0 0) *
##         21) raw_timestamp_part_1>=1.323e+09 372     0 C (0 0 1 0 0) *
##       11) magnet_dumbbell_z>=-0.5 3231  1741 C (0.016 0.39 0.46 0.13 0)  
##         22) magnet_dumbbell_x>=-464.5 995   239 B (0.008 0.76 0.12 0.12 0) *
##         23) magnet_dumbbell_x< -464.5 2236   861 C (0.02 0.23 0.61 0.14 0)  
##           46) pitch_belt< -43.25 196    18 B (0 0.91 0.092 0 0) *
##           47) pitch_belt>=-43.25 2040   683 C (0.022 0.16 0.67 0.15 0)  
##             94) magnet_belt_y>=554.5 1934   577 C (0.023 0.17 0.7 0.1 0) *
##             95) magnet_belt_y< 554.5 106     0 D (0 0 0 1 0) *
##    3) cvtd_timestamp=02/12/2011 13:35,02/12/2011 14:58,02/12/2011 14:59,05/12/2011 11:25,05/12/2011 14:24,28/11/2011 14:15,30/11/2011 17:12 5391  2696 E (0 0.0028 0.12 0.37 0.5)  
##      6) roll_belt< 125.5 3968  1971 D (0 0.0038 0.17 0.5 0.33)  
##       12) roll_dumbbell< -66.03 803   184 C (0 0.0012 0.77 0.086 0.14)  
##         24) raw_timestamp_part_1< 1.323e+09 687    68 C (0 0.0015 0.9 0.09 0.0073) *
##         25) raw_timestamp_part_1>=1.323e+09 116     7 E (0 0 0 0.06 0.94) *
##       13) roll_dumbbell>=-66.03 3165  1237 D (0 0.0044 0.014 0.61 0.37)  
##         26) accel_forearm_x< -83.5 1937   344 D (0 0.0072 0.014 0.82 0.16)  
##           52) magnet_belt_y>=578.5 1743   179 D (0 0.008 0.015 0.9 0.079) *
##           53) magnet_belt_y< 578.5 194    29 E (0 0 0 0.15 0.85) *
##         27) accel_forearm_x>=-83.5 1228   353 E (0 0 0.015 0.27 0.71)  
##           54) accel_dumbbell_y< 48.5 476   188 D (0 0 0.038 0.61 0.36) *
##           55) accel_dumbbell_y>=48.5 752    47 E (0 0 0 0.062 0.94) *
##      7) roll_belt>=125.5 1423    20 E (0 0 0 0.014 0.99) *
```

```r
#displays the decision tree model graphically.
library(rattle)
fancyRpartPlot(fit, main = "Fit Using rpart Algorithm on Training Dataset")
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 

##Cross-Validating the CART Model
The CART model decision tree is then fit to the validation dataset to see the accuracy of the model. A confusion matrix is set up to show the accuracy of the predicted vs. the actual excersize class.

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
##          A 1035   24    4    2    0
##          B   20  618   38   32    0
##          C   10   83  567   64    2
##          D    0    4   13  498   81
##          E    0    0    0   23  572
## 
## Overall Statistics
##                                         
##                Accuracy : 0.892         
##                  95% CI : (0.881, 0.901)
##     No Information Rate : 0.289         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.863         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.972    0.848    0.912    0.805    0.873
## Specificity             0.989    0.970    0.948    0.968    0.992
## Pos Pred Value          0.972    0.873    0.781    0.836    0.961
## Neg Pred Value          0.989    0.963    0.981    0.961    0.973
## Prevalence              0.289    0.198    0.169    0.168    0.178
## Detection Rate          0.280    0.167    0.154    0.135    0.155
## Detection Prevalence    0.289    0.192    0.197    0.162    0.161
## Balanced Accuracy       0.980    0.909    0.930    0.886    0.933
```
The model accuracy is 89% using the rpart algorithm. This is acceptable, since the accuracy is above the 50% and below 95%. Any accuracy above 95% would indicate a likely 'overfitting' situation.

##Predicting the Test Outcomes Using the Model.
The prediction of the test dataset outcomes was fit to the model and shown in the final table.

```r
#Create a column of predicted values in the testing data set.
test$Pred <- predict(fit, newdata = test, type = "class")
#Produce a table showing the outcomes of the 20 test cases.
subset(test, select = c(user_name, Pred))
```

```
##    user_name Pred
## 1      pedro    B
## 2     jeremy    A
## 3     jeremy    C
## 4     adelmo    A
## 5     eurico    A
## 6     jeremy    E
## 7     jeremy    D
## 8     jeremy    C
## 9   carlitos    A
## 10   charles    A
## 11  carlitos    B
## 12    jeremy    C
## 13    eurico    B
## 14    jeremy    A
## 15    jeremy    E
## 16    eurico    E
## 17     pedro    A
## 18  carlitos    B
## 19     pedro    B
## 20    eurico    B
```
