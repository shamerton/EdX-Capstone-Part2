# ==================================================
# DATA SET
# ==================================================
#
# This script is part 2 of the EdX Data Science Capstone course
# The objective of this project is to demonstrate machine learning capabilities
# in a user-chosen data set. The objective of this script is to predict
# the ring count (i.e. the age) of an abalone.
# 
# The data chosen for this project was the abalone data set from the UCI Machine Learning Repository
# Original source at this location:
# https://archive.ics.uci.edu/dataset/1/abalone
#
# --------------------------------------------------
# Additional Information
# --------------------------------------------------
# 
# Predicting the age of abalone from physical measurements. 
# The age of abalone is determined by cutting the shell through the cone,
# staining it, and counting the number of rings through a microscope -- a boring
# and time-consuming task. Other measurements, which are easier to obtain,
# are used to predict the age. Further information, such as weather patterns and
# location (hence food availability) may be required to solve the problem.
# 
# From the original data examples with missing values were removed (the
# majority having the predicted value missing), and the ranges of the continuous
# values have been scaled for use with an ANN (by dividing by 200).
#
# --------------------------------------------------
# Variables Table
# --------------------------------------------------
#
# Variable Name | Role | Type | Description | Units | Missing Values
# Sex | Feature | Categorical | M, F, and I (infant) | no
# Length | Feature | Continuous | Longest shell measurement | mm | no
# Diameter | Feature | Continuous | perpendicular to length | mm | no
# Height | Feature | Continuous | with meat in shell | mm | no
# Whole_weight | Feature | Continuous | whole abalone | grams | no
# Shucked_weight | Feature | Continuous | weight of meat | grams | no
# Viscera_weight | Feature | Continuous | gut weight (after bleeding) | grams | no
# Shell_weight | Feature | Continuous | after being dried	grams | no
# Rings	Target | Integer | +1.5 gives the age in years | - | no
#
# --------------------------------------------------
# Claude Code Disclosure
# --------------------------------------------------
#
# The Claude Code large language model (LLM) by Anthropic was used during the
# development of this project. The primary use of Claude was to validate and
# troubleshoot manually written code. As this was my first significant attempt
# at software development, Claude's ability to identify errors and correct
# oversights was invaluable to both skill development and submission deadlines.
# 
# Claude assisted sections of code are commented as "Claude assisted:" followed'
# by an explanation of the nature of the assistance. Every line of code is
# understood and can be explained on demand.
# 
# All comments were written by me.
#
# ==================================================
# PACKAGES
# ==================================================

# Some package and data loading steps taken from Part 1 of the capstone

# Automatically install required packages
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(splines)) install.packages("splines", repos = "http://cran.us.r-project.org")
if(!require(foreach)) install.packages("foreach", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# ==================================================
# DATA LOADING
# ==================================================

# time out option so that the script doesn't get stuck
options(timeout = 120)

# Set the working directory to the same directory as the R file to keep all the files together
# Claude assisted: Claude wrote this line. I couldn't figure it out. I did eventually find the same answer buried in a StackOverflow thread, but some users there reported that it didn't work. It works for me ("it works on my machine lol"). Honestly, I'm a bit shocked I had to go to Claude for this and that something like setwd(.) doesn't work.
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# A list of headers to use when loading the data. The original data is headerless
abalone_vars <- c("Sex", "Length", "Diameter", "Height", "WeightWhole", "WeightShucked", "WeightViscera", "WeightShell", "Rings")

# If the zip file is missing, try to download it.
zip <- "abalone.zip"
if (!file.exists(zip)) download.file("https://archive.ics.uci.edu/static/public/1/abalone.zip", zip)

# If the data is missing, try to extract the zip file.
raw <- "abalone.data"
if (!file.exists(raw)) unzip(zip)

# if the data file is still missing after attempting to download and extract, stop the script
if (!file.exists(raw)) stop("abalone.data not found. Please manually download and extract from the link in the R file or the gitHub repo.")

abalone <- read.csv("abalone.data", header = FALSE, col.names = abalone_vars)

# --------------------------------------------------

# This is a predefined RMSE function based on the advice from this blog:
# https://www.r-bloggers.com/2021/07/how-to-calculate-root-mean-square-error-rmse-in-r/
RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

set.seed(1996) # The year of the original data set

# ==================================================
# DATA EXPLORATION
# ==================================================

# First we create a copy of the dataset for visualization purposes.

unscale <- 200 # According to the information supplied with the data set, the original data was scaled by a factor of 200. This will be used to reverse that scale

vis <- abalone |> mutate(
  Length = Length * unscale,
  Diameter = Diameter * unscale,
  Height = Height * unscale,
  WeightWhole = WeightWhole * unscale,
  WeightShucked = WeightShucked * unscale,
  WeightViscera = WeightViscera * unscale,
  WeightShell = WeightShell * unscale)

vis <- vis |> filter(Height < 60) # Two extreme outliers were identified in the height data that are removed by this cut-off

# --------------------------------------------------
# Univariate Data Exploration
# --------------------------------------------------

# All plots are stored instead of rendered. This prepares them for use in the
# RMD file and eliminates repetition of the code within both script and RMD.
# They can be viewed at any time from the console

# This is a basic plot of the Rings data to see if if is normally distributed (it is).
plotAgeDist <- vis |>
  ggplot(aes(Rings)) +
  geom_bar() +
  geom_vline(xintercept = mean(vis$Rings), colour = "red") + # The Mean marked in red
  geom_vline(xintercept = mean(vis$Rings) - 2 * sd(vis$Rings), colour = "blue") + # The lower 2SD bound
  geom_vline(xintercept = mean(vis$Rings) + 2 * sd(vis$Rings), colour = "blue") + # The upper 2SD bound
  ylab("Count") +
  xlab("Ring Count (Age)") +
  ggtitle("Abalone Age Distribution")

# Claude assisted: in the next section I wanted a facet wrap that unified
# similar plots into multiple panes. I could create each individual plot, but
# after about an hour of failed attempts, I asked Claude for assistance.
# My error was failing to use "pivot_longer" prior to drawing the plots.

# A density plot of each variables
plotVarDensity <- vis |>
  pivot_longer(cols = Length:WeightShell, names_to = "Variable") |>
  ggplot(aes(value, fill = Variable)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Variable, scales = "free") +
  ylab("Density") +
  xlab("Measurement") +
  ggtitle("Measurement Density by Variable")

# The next line is commented out because I was reviewing the data to understand
# what Claude did that made it workable. It's retrospect, it's kind of obvious.

# vis_claude <- vis |> pivot_longer(cols = Length:WeightShell, names_to = "Measurement", values_to = "Value")

# --------------------------------------------------
# Bivariate Data Exploration
# --------------------------------------------------

# Does Sex affect the ring count?
plotGenderBox <- vis |>
  ggplot(aes(Sex, Rings, colour = Sex)) +
  geom_boxplot() +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) + # Claude assisted: I just wanted custom labels on the legend, man!! Why was it so hard?!!
  ylab("Ring Count (Age)") +
  xlab("Sex") +
  ggtitle("Gender Boxplots by Sex")

# Is there a variable with a strong visual regression shape?
plotVarScatter <- vis |>
  pivot_longer(cols = Length:WeightShell, names_to = "Variable", values_to = "Value") |>
  ggplot(aes(Rings, Value, group = Rings, colour = Sex)) +
  geom_jitter() +
  facet_wrap(~Variable, scales = "free_y") +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) +
  ylab("Measurement") +
  xlab("Ring Count (Age)") +
  ggtitle("Measurement Scatterplots by Variable")

# This plot isolates the height outliers for display in the report
plotHeightOutliers <- abalone |>
  ggplot(aes(Rings, Height, colour = Sex)) +
  geom_jitter() +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) +
  ylab("Measurement") +
  xlab("Ring Count (Age)") +
  ggtitle("Height Outliers")

# Is there a variable with a stronger indicator by mean or sd?
plotVarBox <- vis |>
  pivot_longer(cols = Length:WeightShell, names_to = "Variable", values_to = "Value") |>
  ggplot(aes(Rings, Value, group = Rings, colour = Variable)) +
  geom_boxplot() +
  facet_wrap(~Variable, scales = "free_y") +
  ylab("Measurement") +
  xlab("Ring Count (Age)") +
  ggtitle("Measurement Boxplots by Variable")

# Ratios of weights added to the visualization data
vis <- vis |> mutate(ratioShellWhole = WeightShell / WeightWhole, 
                     ratioShuckedWhole = WeightShucked / WeightWhole, 
                     ratioLengthShell = Length / WeightShell
                     )

# Is there a ratio that better indicates the age of the abalone?
# For example, perhaps older abalone have a heavier shell as a portion
# of their overall weight
plotRatioShellWhole <- vis |> ggplot(aes(Rings, log10(ratioShellWhole), colour = Sex)) +
  geom_jitter() +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) +
  ylab("Shell:Whole Weight (Log 10) ") +
  xlab("Ring Count (Age)") +
  ggtitle("Shell Weight to Whole Weight Ratio")

plotRatioShuckedWhole <- vis |> ggplot(aes(Rings, log10(ratioShuckedWhole), colour = Sex)) +
  geom_jitter() +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) +
  ylab("Meat:Whole Weight (Log 10) ") +
  xlab("Ring Count (Age)") +
  ggtitle("Harvest Weight to Whole Weight Ratio")

# In this plot we can see a very clear relationship between the ratio of 
# length to shell weight and the "infant" Sex identification
plotRatioLengthShell <- vis |> ggplot(aes(Rings, log10(ratioLengthShell), colour = Sex)) +
  geom_jitter() +
  scale_colour_discrete(labels = c("F (female)", "I (infant)", "M (male)")) +
  ylab("Shell Length:Shell Weight (Log 10) ") +
  xlab("Ring Count (Age)") +
  ggtitle("Shell Length to Shell Weight Ratio")

infantRatio <- mean(vis$Sex == "I") # this ratio is the percentage of infant observations in the data. This is stored for use in the report

# ==================================================
# DATA SETS
# ==================================================
#
# The data is given numerical classes instead of continuous values
#
# --------------------------------------------------

# Classes of continuous variables were added were added to the data.
# The idea here was that since ring count is an integer
# that perhaps classified continuous variables would help to identify
# ring count. This was wrong. These values are used for one model and that
# model performs terribly.

dat <- vis |> mutate(LengthClass = cut(Length, breaks = 20, labels = FALSE),
                         DiamClass = cut(Diameter, breaks = 20, labels = FALSE),
                         HeightClass = cut(Height, breaks = 20, labels = FALSE),
                         WeightWholeClass = cut(WeightWhole, breaks = 20, labels = FALSE),
                         WeightShuckedClass = cut(WeightShucked, breaks = 20, labels = FALSE),
                         WeightVisceraClass = cut(WeightViscera, breaks = 20, labels = FALSE),
                         WeightShellClass = cut(WeightShell, breaks = 20, labels = FALSE),
                         ratioShellWhole = WeightShell / WeightWhole, 
                         ratioShuckedWhole = WeightShucked / WeightWhole, 
                         ratioLengthShell = Length / WeightShell)

# This next section is a holdout dataset used prior to implementing kfold
# Due to the reduced number of observations, k-fold presented a stronger
# form of cross validation and was adopted over a hold out.

# dat_i <- createDataPartition(dat_ex$Rings, times = 1, p = 0.8, list = FALSE)

# dat <- dat_ex[dat_i,] # The training data set (includes training and k-fold cross validation)
# dat_v <- dat_ex[-dat_i,] # A holdout validation data set

# --------------------------------------------------
# K-FOLD CROSS VALIDATION
#
# k-Fold parameters are based on the advice in this blog:
# https://www.statology.org/k-fold-cross-validation-in-r/
#
# --------------------------------------------------

# A fold of 4 was taken for the number of folds
# Manual testing of folds 3, 4, 5 revealed 4 to have the slightly stronger performance
kFold <- trainControl(method = "cv", number = 4, savePredictions = "final")

# ==================================================
# MODEL EXPLORATION
# ==================================================
#
# List of caret models taken from course textbook for reference:
#
# models <- c("glm", "lda",  "naive_bayes",  "svmLinear", "gamboost",  
#             "gamLoess", "qda", "knn", "kknn", "loclda", "gam", "rf", 
#             "ranger","wsrf", "Rborist", "avNNet", "mlp", "monmlp", "gbm", 
#             "adaboost", "svmRadial", "svmRadialCost", "svmRadialSigma")
#
# Not all models were deployed
#
# --------------------------------------------------
# Models
# --------------------------------------------------

# A naive initial RMSE calculated from the mean of Rings only
yRingsInit <- RMSE(mean(dat$Rings), dat$Rings)

# A simple linear regression model
yRingsLM <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "lm", trControl = kFold)

# A linear model with classified data
# The classified data performs markedly worse than unclassified.
# No other models use classified data
yRingsClassLM <- train(Rings ~ LengthClass + DiamClass + HeightClass + WeightWholeClass + WeightShuckedClass + WeightVisceraClass + WeightShellClass, data = dat, method = "lm", trControl = kFold)

# A generalized linear model
yRingsGLM <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "glm", trControl = kFold)

# A localized regression model
yRingsLoess <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "gamLoess", trControl = kFold, tuneGrid = expand.grid(span = seq(0.1, 0.5, 0.05), degree = 1))

# A k-nearest neighbours model
yRingsKNN <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "knn", trControl = kFold, tuneGrid = data.frame(k = seq(10,30,1)))

# A random forest model
yRingsRF <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "rf", trControl = kFold, tuneGrid = expand.grid(mtry = c(3, 5, 7)))

# Claude assisted: Claude suggested the Ranger and GBM models as an alternative to
# RandomForest. Ranger model performed extremely well and prompted additional
# reading on caret models that were not covered in the course.
# The values for mtry and min.node.size were taken by default. Further
# exploration of these values happens below.

# A 'ranger' model, which is a fast random forest
yRingsRanger <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "ranger", trControl = kFold, tuneGrid = expand.grid(mtry = c(3, 5, 7), splitrule = "variance", min.node.size = c(1, 5, 10, 20, 50)))
# Additional information about Ranger on this blog:
# https://www.geeksforgeeks.org/r-language/ranger-function-in-r/

# A gradient boost model
yRingsGBM <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell, data = dat, method = "gbm", trControl = kFold, verbose = FALSE)
# Additional GBM information taken from these blogs:
# https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/
# https://towardsdatascience.com/gradient-boosting-regressor-explained-a-visual-guide-with-code-examples-c098d1ae425c/


# Claude assisted: in this section I wanted a clean presentation of the
# performance of each model. Claude introduced getTrainPerf() from caret package.
# getTrainPerf() extracts the same information from different model results
# which makes it possible to easily create a single data frame.
# My attempt at this section was to use rbind() which failed due to the
# inconsistencies in the results section.

modelPerf <- bind_rows( # a convenient data frame of the models and their performance
  getTrainPerf(yRingsLM),
  getTrainPerf(yRingsClassLM),
  getTrainPerf(yRingsGLM),
  getTrainPerf(yRingsLoess),
  getTrainPerf(yRingsKNN),
  getTrainPerf(yRingsRF),
  getTrainPerf(yRingsRanger),
  getTrainPerf(yRingsGBM)
) |>
  mutate(Model = c("Linear Regression", "Linear (Classified)", "Linear (Generalized)", "Linear (Localized)", "KNN", "Random Forest", "Ranger", "GBM"),
         Improvement = yRingsInit - TrainRMSE) |>
  select(Model, RMSE = TrainRMSE, Rsquared = TrainRsquared, MAE = TrainMAE, Improvement)

print(arrange(modelPerf, desc(Improvement))) # the top performing models in descending order

# --------------------------------------------------
# # Additional explorations of the Ranger model and its parameters mtry and min.node.size
# # Commented out as these are process-heavy and are not required for the final results
# 
# mtryRanger <- seq(1,7,1)
# rmsesMtryRanger <- sapply(mtryRanger, function(mtryRanger) {
#   rangerTest <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell + ratioShellWhole + ratioShuckedWhole + ratioLengthShell,
#         data = dat, method = "ranger", trControl = kFold,
#         tuneGrid = expand.grid(mtry = mtryRanger, splitrule = "variance", min.node.size = 10))
#   
#   min(rangerTest$results$RMSE)
# 
#   # yhatRangerTest <- predict(rangerTest, newdata = dat_v)
#   # RMSE(dat_v$Rings, yhatRangerTest)
# })
# plot(rmsesMtryRanger)
# 
# # An mtry value of 3 emerges as a leader in this plot
# 
# minNodeRanger <- seq(5,50,5)
# rmsesNodeRanger <- sapply(minNodeRanger, function(minNodeRanger) {
#   rangerTest <- train(Rings ~ Length + Diameter + Height + WeightWhole + WeightShucked + WeightViscera + WeightShell + ratioShellWhole + ratioShuckedWhole + ratioLengthShell,
#                       data = dat, method = "ranger", trControl = kFold,
#                       tuneGrid = expand.grid(mtry = 3, splitrule = "variance", min.node.size = minNodeRanger))
#   
#   min(rangerTest$results$RMSE)
#   # yhatRangerTest <- predict(rangerTest, newdata = dat_v)
#   # RMSE(dat_v$Rings, yhatRangerTest)
# })
# 
# plot(rmsesNodeRanger)
# # The min node value does not appear to have a significant reliable impact
# # on the RMSE. The original values of 5,10 seem reasonable to retain.

# ==================================================
# ENSEMBLE RESULTS
# ==================================================

# The top performing models were included in an ensemble model

# caretEnsemble() found based on this blog:
# https://moldstud.com/articles/p-integrating-caret-with-other-r-packages-for-enhanced-machine-learning-performance

eList <- caretList(Rings ~ . -Sex - ratioShellWhole - ratioShuckedWhole,
                   trControl = kFold,
                   data = dat,
                   tuneList = list(
                     "ranger" = caretModelSpec(method = "ranger", tuneGrid = yRingsRanger$bestTune), # mtry = 3 and min.node = 10 for this seed
                     "loess" = caretModelSpec(method = "gamLoess", tuneGrid = yRingsLoess$bestTune),
                     "knn" = caretModelSpec(method = "knn", tuneGrid = yRingsKNN$bestTune), # k = 22 for this seed
                     "gbm" = caretModelSpec(method = "gbm", verbose = FALSE)
                   ))

eListS <- caretList(Rings ~ . - ratioShellWhole - ratioShuckedWhole,
                   trControl = kFold,
                   data = dat,
                   tuneList = list(
                     "ranger" = caretModelSpec(method = "ranger", tuneGrid = yRingsRanger$bestTune), # mtry = 3 and min.node = 10 for this seed
                     "loess" = caretModelSpec(method = "gamLoess", tuneGrid = yRingsLoess$bestTune),
                     "knn" = caretModelSpec(method = "knn", tuneGrid = yRingsKNN$bestTune), # k = 22 for this seed
                     "gbm" = caretModelSpec(method = "gbm", verbose = FALSE)
                   ))

# Note from caretEnsemble() Help:
# "Every model in the "library" must be a separate train object. For example, if
# you wish to combine a random forests with several different values of mtry,
# you must build a model for each value of mtry. If you use several values of
# mtry in one train model, (e.g. tuneGrid = expand.grid(.mtry=2:5)), caret will
# select the best value of mtry before we get a chance to include it in the
# ensemble. By default, RMSE is used to ensemble regression models, and AUC is
# used to ensemble Classification models. This function does not currently
# support multi-class problems"
#
# The tuneGrid values of $bestTune for each model are applied for this reason

yRingsEnsemble <- caretEnsemble(eList) # the final ensemble model (without Sex variable)
yRingsEnsembleS <- caretEnsemble(eListS) # the final ensemble model (all variables)

sliceEnsemble <- slice_sample(yRingsEnsemble$ens_model$pred, n = 10, replace = FALSE) # a random selection of predictions for the report
sliceEnsembleS <- slice_sample(yRingsEnsembleS$ens_model$pred, n = 10, replace = FALSE) # a random selection of predictions for the report

save.image("FinalAssessmentPart2_report.RData") # Save a copy of the data for use in the RMD report
