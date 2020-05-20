install.packages("caret")
install.packages("glmnet")
install.packages("tidyr")
install.packages("dplyr")
install.packages("randomForest")
install.packages("gridExtra")
library("caret")
library("tidyr")
library("dplyr")
library("glmnet")
library("randomForest")
library("ggplot2")
library("gridExtra")

stock = read.csv("/Users/Joanne/Desktop/stock.csv")
glimpse(stock)
head(dataset)
stock %>% select(Price.Variation) %>% glimpse()
stock %>% select(-Price.Variation, -Company) %>% glimpse()
standardize = function(x) {x / sqrt(mean((x - mean(x))^2))}
stock = stock %>%
  select(-Price.Variation,-Company) %>%
  mutate_all(standardize) %>%
  mutate(Price.Variation=stock$Price.Variation)
stock %>% glimpse()

# Randomly split the dataset into two mutually exclusive datasets
n = dim(stock)[1] # number of observations
p = dim(stock)[2]-1 # number of predictors
y = stock[,p]
X = data.matrix(stock[,-p])
set.seed(1)

n.train = floor(0.8*n)
n.test = n - n.train

# Lambda
lam.las = c(seq(1e-3,0.1,length=100),seq(0.12,2.5,length=100)) 
lam.rid = lam.las*1000

M = 100

# R squared
Rsq.test.en = rep(0,M)  #en = elastic net
Rsq.train.en = rep(0,M)
Rsq.test.ridge = rep(0,M)  # Ridge
Rsq.train.ridge = rep(0,M)
Rsq.test.lasso = rep(0,M)  # Lasso
Rsq.train.lasso = rep(0,M)
Rsq.test.rf = rep(0,M)  # rf= randomForest
Rsq.train.rf = rep(0,M)

for (m in c(1:M)) {
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]

  # Fit elastic-net and calculate and record the train and test R squares 
  cv.en.fit = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  en.fit = glmnet(X.train, y.train, alpha = 0.5, lambda = cv.en.fit$lambda.min)
  y.train.hat.en = predict(en.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.en = predict(en.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m] = 1-mean((y.test - y.test.hat.en)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m] = 1-mean((y.train - y.train.hat.en)^2)/mean((y - mean(y))^2)  
  
  # Fit lasso and calculate and record the train and test R squares
  cv.lasso.fit = cv.glmnet(X.train, y.train, alpha = 1, lambda = lam.las, nfolds = 10)
  lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = cv.lasso.fit$lambda.min)
  y.train.hat.lasso = predict(lasso.fit, newx = X.train, type = "response")  # y.train.hat=X.train %*% fit$beta + fit$a0   
  y.test.hat.lasso = predict(lasso.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.lasso[m] = 1-mean((y.test - y.test.hat.lasso)^2)/mean((y - mean(y))^2)
  Rsq.train.lasso[m] = 1-mean((y.train - y.train.hat.lasso)^2)/mean((y - mean(y))^2)  

  # Fit ridge and calculate and record the train and test R squares
  cv.ridge.fit = cv.glmnet(X.train, y.train, alpha = 0, lambda = lam.rid, nfolds = 10)
  ridge.fit = glmnet(X.train, y.train, alpha = 0, lambda = cv.ridge.fit$lambda.min)
  y.train.hat.ridge = predict(ridge.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.ridge = predict(ridge.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ridge[m] = 1-mean((y.test - y.test.hat.ridge)^2)/mean((y - mean(y))^2)
  Rsq.train.ridge[m] = 1-mean((y.train - y.train.hat.ridge)^2)/mean((y - mean(y))^2)  

  # Fit RF and calculate and record the train and test R squares 
  rf = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100)
  y.test.hat.rf = predict(rf, X.test)
  y.train.hat.rf = predict(rf, X.train)
  Rsq.test.rf[m] = 1-mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m] = 1-mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2)  

  cat(sprintf("m=%3.f| ,  Rsq.test.en=%.2f|,  Rsq.train.en=%.2f|, 
              Rsq.test.ridge=%.2f|,  Rsq.train.ridge=%.2f|,
              Rsq.test.lasso=%.2f|,  Rsq.train.lasso=%.2f|,
              Rsq.test.rf=%.2f|,  Rsq.train.rf=%.2f|\n", 
              m, Rsq.test.en[m], Rsq.train.en[m],
              Rsq.test.ridge[m], Rsq.train.ridge[m],
              Rsq.test.lasso[m], Rsq.train.lasso[m],
              Rsq.test.rf[m], Rsq.train.rf[m]
              )) 
}

# (b) Show the side-by-side boxplots of Rtest^2, Rtrain^2
# Rtest
boxplot(Rsq.test.rf, Rsq.test.en, Rsq.test.lasso, Rsq.test.ridge,
        main = "Boxplot of R^2 Test",
        names = c("RF", "Elastic Net", "Lasso", "Ridge"),
        col = c("red","yellow", "green", "blue"))
text(1, 0, paste("Avg:",round(mean(Rsq.test.rf),4)))
text(2, 0, paste("Avg:",round(mean(Rsq.test.en),4)))
text(3, 0, paste("Avg:",round(mean(Rsq.test.lasso),4)))
text(4, 0, paste("Avg:",round(mean(Rsq.test.ridge),4)))

# Rtrain
boxplot(Rsq.train.rf, Rsq.train.en, Rsq.train.lasso, Rsq.train.ridge,
        main = "Boxplot of R^2 Train",
        names = c("RF", "Elastic Net", "Lasso", "Ridge"),
        col = c("red","yellow", "green", "blue"))
text(1, 0.80, paste("Avg:",round(mean(Rsq.train.rf),4)))
text(2, 0.90, paste("Avg:",round(mean(Rsq.train.en),4)))
text(3, 0.90, paste("Avg:",round(mean(Rsq.train.lasso),4)))
text(4, 0.90, paste("Avg:",round(mean(Rsq.train.ridge),4)))

# (c) For one on the 100 samples, create 10-fold CV curves for lasso, elastic-net α = 0.5, ridge.
# boxplot for 10-fold CV cruves EN
shuffled_indexes = sample(n)
train = shuffled_indexes[1:n.train]
test = shuffled_indexes[(1+n.train):n]
X.train = X[train, ]
y.train = y[train]
X.test = X[test, ]
y.test = y[test]

cv.lasso = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
cv.el = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
cv.ridge = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

plot(cv.el , 
     main="10-fold CV Curves for Elastic Net\n")

# boxplot for  10-fold CV cruves Ridge
plot(cv.ridge, 
     main="10-fold CV Curves for Ridge\n")

# boxplot for  10-fold CV cruves Lasso
plot(cv.lasso, 
     main="10-fold CV Curves for Lasso\n")


# (d) For one on the 100 samples, show the side-by-side boxplots of train and test residuals.
# Creating Residual variables
shuffled_indexes = sample(n)
train = shuffled_indexes[1:n.train]
test = shuffled_indexes[(1+n.train):n]
X.train = X[train, ]
y.train = y[train]
X.test = X[test, ]
y.test = y[test]

# Lasso
cv.lasso.fit = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = cv.lasso.fit$lambda.min)
y.train.hat.la = predict(lasso.fit, newx = X.train, type = "response") 
y.test.hat.la = predict(lasso.fit, newx = X.test, type = "response")  
Res.test.la = y.test - y.test.hat.la
Res.train.la = y.train - y.train.hat.la

# Elastic Net
cv.en.fit = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
en.fit = glmnet(X.train, y.train, alpha = 0.5, lambda = cv.en.fit$lambda.min)
y.train.hat.en = predict(en.fit, newx = X.train, type = "response") 
y.test.hat.en = predict(en.fit, newx = X.test, type = "response")  
Res.test.en = y.test - y.test.hat.en
Res.train.en = y.train - y.train.hat.en

# Ridge
cv.ri.fit = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ri.fit = glmnet(X.train, y.train, alpha = 0, lambda = cv.ri.fit$lambda.min)
y.train.hat.ri = predict(ri.fit, newx = X.train, type = "response") 
y.test.hat.ri = predict(ri.fit, newx = X.test, type = "response")  
Res.test.ri = y.test - y.test.hat.ri
Res.train.ri = y.train - y.train.hat.ri

# RF
rf = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100)
y.test.hat.rf = predict(rf, X.test)
y.train.hat.rf = predict(rf, X.train)
Res.test.rf = y.test - y.test.hat.rf
Res.train.rf = y.train - y.train.hat.rf

boxplot(Res.test.rf, Res.test.en, Res.test.la, Res.test.ri,
        main = "Boxplot of Test Residual",
        names = c("RF", "Elastic Net", "Lasso", "Ridge"),
        col = c("red","yellow", "green", "blue"))

boxplot(Res.train.rf, Res.train.en, Res.train.la, Res.train.ri,
        main = "Boxplot of Train Residuals",
        names = c("RF", "Elastic Net", "Lasso", "Ridge"),
        col = c("red","yellow", "green", "blue"))


# (e) Present bar-plots (with bootstrapped error bars) of the estimated coefficients, and the importance of the parameters.
bootstrapSamples = 100
beta.rf.bs = matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs = matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.ridge.bs = matrix(0, nrow = p, ncol = bootstrapSamples) 
beta.lasso.bs = matrix(0, nrow = p, ncol = bootstrapSamples) 

for (m in 1:bootstrapSamples){
  bs_indexes = sample(n, replace=T)
  X.bs = X[bs_indexes, ]
  y.bs = y[bs_indexes]
  
  # fit bs rf
  rf = randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE, ntree=100, nodesize = 100)
  beta.rf.bs[,m] = as.vector(rf$importance[,1])
  
  # fit bs en
  a = 0.5 # elastic-net
  cv.fit = cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m] = as.vector(fit$beta)
  
  # fit bs lasso
  a = 1 # lasso
  cv.fit = cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.lasso.bs[,m] = as.vector(fit$beta)
  
  # fit bs ridge
  a = 0 # Ridge
  cv.fit = cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ridge.bs[,m] = as.vector(fit$beta)
  
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd = apply(beta.rf.bs, 1, "sd")
en.bs.sd = apply(beta.en.bs, 1, "sd")
lasso.bs.sd = apply(beta.lasso.bs, 1, "sd")
ridge.bs.sd = apply(beta.ridge.bs, 1, "sd")

# fit rf to the whole data
rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE, ntree=100)

# fit elastic-net to the whole data
cv_en.fit = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
enfit = glmnet(X, y, alpha = 0.5, lambda = cv_en.fit$lambda.min)

# fit lasso to the whole data
cv_lasso.fit = cv.glmnet(X, y, alpha = 1, nfolds = 10)
lassofit = glmnet(X, y, alpha = 1, lambda = cv_lasso.fit$lambda.min)  

# fit Ridge to the whole data
cv_ridge.fit = cv.glmnet(X, y, alpha = 0, nfolds = 10)
ridgefit = glmnet(X, y, alpha = 0, lambda = cv_ridge.fit$lambda.min)


betaS.rf = data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf) = c( "feature", "value", "err")

betaS.en = data.frame(names(X[1,]), as.vector(enfit$beta), 2*en.bs.sd)
colnames(betaS.en) = c( "feature", "value", "err")

betaS.lasso = data.frame(names(X[1,]), as.vector(lassofit$beta), 2*lasso.bs.sd)
colnames(betaS.lasso) = c( "feature", "value", "err")

betaS.ridge = data.frame(names(X[1,]), as.vector(ridgefit$beta), 2*ridge.bs.sd)
colnames(betaS.ridge) = c( "feature", "value", "err")

# We need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature = factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature = factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.lasso$feature = factor(betaS.lasso$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ridge$feature = factor(betaS.ridge$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

# Compare random forest and elastic net
rfPlot = ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Feature Importance of Random Forest") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank())
 

enPlot = ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Feature Importance of Elastic Net") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank())


grid.arrange(rfPlot, enPlot, nrow = 2,
             top = "Random Forest vs. Elastic Net Importance Parameter")


# Compare elastic net and lasso/ridge
laPlot = ggplot(betaS.lasso, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Feature Importance of Lasso") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank())


riPlot = ggplot(betaS.ridge, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Feature Importance of Elastic Ridge") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank())

grid.arrange(laPlot, riPlot, nrow = 2,
             top = "Lasso vs. Ridge Importance Parameter")


grid.arrange(enPlot, laPlot, nrow = 2)
grid.arrange(enPlot, riPlot, nrow = 2)



## Summarize the performance and the time need to train each model in a table and comment on it.
# Calculate the time needed to train each model
start_lasso = Sys.time()
cv.lasso = cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso = glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)
end_lasso = Sys.time()
time_lasso = end_lasso - start_lasso

start_elast = Sys.time()
cv.elast = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
elast = glmnet(X, y, alpha = 0.5, lambda = cv.elast$lambda.min)
end_elast = Sys.time()
time_elast = end_elast - start_elast

start_ridge = Sys.time()
cv.ridge = cv.glmnet(X, y, alpha = 0, nfolds = 10)
ridge = glmnet(X, y, alpha = 0, lambda = cv.ridge$lambda.min)
end_ridge = Sys.time()
time_ridge = end_ridge - start_ridge

start_rf = Sys.time()
rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
end_rf = Sys.time()
time_rf = end_rf - start_rf


model = c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest')
performance = round(c(mean(Rsq.test.lasso), mean(Rsq.test.en), mean(Rsq.test.ridge), mean(Rsq.test.rf)), 3)
time = round(c(time_lasso, time_elast, time_ridge, time_rf), 2)

summary_table = data.frame(model, performance, time)
write.csv(summary_table, 'summary_table.csv')


