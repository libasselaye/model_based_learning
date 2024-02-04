#Mouhamadou Mansour Lo
#25/02/2022

## Test des Modeles ####

#source(file="Algo_EM.R")
#source(file="classif_EM.R")

Train_image <- file("train-images.idx3-ubyte", "rb")
Train_Label <- file("train-labels.idx1-ubyte", "rb")

Test_image <- file("t10k-images.idx3-ubyte", "rb")
Test_Label <- file("t10k-labels.idx1-ubyte", "rb")

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

#Code de chargement source
#https://gist.github.com/daviddalpiaz/ae62ae5ccd0bada4b9acd6dbc9008706
# load image files
load_image_file = function(f) {
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(f) {
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train = load_image_file(Train_image)
test  = load_image_file(Test_image)

# load labels
train$y = as.factor(load_label_file(Train_Label))
test$y  = as.factor(load_label_file(Test_Label))

# view test image
show_digit(train[1, ])






variances<-apply(train[,-length(train)], 2, var)
ind_supp <- names(variances[which(variances<=12350)])

train1 <- as.data.frame(train[1:10000, ])
train1_reduit <- train1[ , !names(train1) %in% ind_supp]

test1_reduit <- test[ , !names(test) %in% ind_supp]


library(EMpackage)


#MODELE EM
mytest <- Classif_MM(train1_reduit[,-length(train1_reduit)],train1_reduit$y,test1_reduit[,-length(test1_reduit)],3)
saveRDS(mytest, file = "result_EM_10000_45.rds")

result <- readRDS(file = "result_EM_10000_45.rds")
t <- table(result$prediction,test1_reduit$y)
library(caret)
confusionMatrix(t)





#RANDOM FOREST
fit = randomForest::randomForest(y ~ ., data = train1_reduit)
fit$confusion
test_pred = predict(fit, test1_reduit)
mean(test_pred == test1_reduit$y)
table(predicted = test_pred, actual = test1_reduit$y)




#SVM
library("e1071")

svm_model <- svm(y ~ ., data=train1_reduit)
summary(svm_model)

test_pred = predict(svm_model, test1_reduit)
mean(test_pred == test1_reduit$y)
table(predicted = test_pred, actual = test1_reduit$y)