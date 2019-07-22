#set encoding
Sys.setlocale(locale = "UTF-8")

# install.packages("RMySQL")
library(RMySQL)

# install.packages("data.table")
library(data.table)

# install.packages("randomForest")
library(randomForest)

# install.packages("ggplot2")
library(ggplot2)

# install.packages("gridExtra")
library(gridExtra)


setwd("/Users/Alexandre/Documents/CoinKeeper/Archive/Documents/User_Experience_Improvement/classifier_post")
load('sample_data_post.Rdata')

# category name growth rate
df <- final[order(final$charged_at), ]
df <- df[, c("days", "cats") := list(as.numeric(factor(format(as.POSIXct(charged_at), "%Y-%m-%d"))),
                                              cumsum(ifelse(duplicated(category_name), 0, 1)))]
df <- df[,list(cats = max(cats)), by = list(days)]

ggplot(df, aes(days, cats)) + 
  geom_point(colour="salmon") +
  labs(x = "Number of days", y = "Number of unique category names", title = "Number of unique category names in the system over time") +
  theme(plot.title = element_text(hjust = 0.5))
rm(df)

# merchant name growth rate
df <- final[order(final$charged_at), ]
df <- df[, c("days", "merch") := list(as.numeric(factor(format(as.POSIXct(charged_at), "%Y-%m-%d"))),
                                     cumsum(ifelse(duplicated(merchant_name), 0, 1)))]
df <- df[,list(merch = max(merch)), by = list(days)]

ggplot(df, aes(days, merch)) + 
  geom_point(colour="salmon") +
  labs(x = "Number of days", y = "Number of unique merchant names", title = "Number of unique merchant names in the system over time") +
  theme(plot.title = element_text(hjust = 0.5))
rm(df)

# category and merchant name growth rate by user
df <- final[order(final$charged_at), ]
df <- df[, c("days", "merch", "cats") := list(as.numeric(factor(format(as.POSIXct(charged_at), "%Y-%m-%d"))),
                                              cumsum(ifelse(duplicated(merchant_name), 0, 1)),
                                              cumsum(ifelse(duplicated(category_name), 0, 1))),
         by = user_id]
df <- df[,list(merch = max(merch), cats = max(cats)), by = list(user_id, days)]

p1 <- ggplot(df, aes(days, cats, group = days)) + 
  geom_boxplot(aes(fill = days), alpha = 0.7) +
  geom_smooth(aes(group=1), method = 'loess', col = "salmon", se = F, size = 2) +
  labs(x = element_blank(), y = "Number of unique category names", title = "Massively categorical variables at the user level over time") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none", 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
p2 <- ggplot(df, aes(days, merch, group = days)) + 
  geom_boxplot(aes(fill = days), alpha = 0.7) +
  geom_smooth(aes(group=1), method = 'loess', col = "salmon", se = F, size = 2) +
  labs(x = "Days", y = "Number of unique merchant names") +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 1)
rm(p1, p2, df)

# select one observation by class in dataframe
oneObsByClass <- function (df = data.frame()){
  seq_class <- ave(df$category_name, df$category_name, FUN = seq_along)
  s <- which(seq_class==1)
  return(s)
}

# calculate macro average f1 score
calculate.f1 <- function(values = vector(), pred = vector()){
  recall <- mean(sapply(unique(values), function (z) mean(values[values==z]==pred[values==z])), na.rm = T)
  precision <- mean(sapply(unique(values), function (z) mean(values[pred==z]==pred[pred==z])), na.rm = T)
  f1 <- (2*recall*precision)/(recall+precision)
  return(f1)
}


# calculate counts for bayes classifier
fit.bayes <- function (outputName = character(1), inputName = character(), data = data.table()){
  
  results <- list()
  
  # count of observations by class 
  classDF <- data[, list(total = .N), by = c(outputName)]
  
  for (k in 1:length(inputName)){
    # count observation by class input variable
    probDF <- data[, list(prob = .N), by = c(inputName[k], outputName)]
    
    # store as data frame in a list
    results[[length(results)+1]] <- as.data.frame(probDF)
    names(results)[length(results)] <- inputName[k]
  }
  
  # calculate prior probability and store in list
  classDF$prior <- classDF$total/nrow(data)
  results[[length(results)+1]] <- as.data.frame(classDF)
  names(results)[length(results)] <- outputName
  
  return(results)
}



# bayes classifier 
predict.bayes <- function (counts = list(), test = data.table()){
  
  pred <- character(0)
  
  # retrieve output name from model  
  outputName <- names(counts)[length(counts)]
  
  for (i in 1:nrow(test)){
    temp <- counts
    
    # subset count data for each feature with new data values and merge result into one table 
    temp[-length(temp)] <- mapply(function(x, y, z) list(subset(x, x[,z]==y)[,c(outputName, "prob")]), temp[-length(temp)], test[i,], names(test))
    probDF <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = outputName, all=TRUE), temp))
    probDF[is.na(probDF)] <- 0
    
    # for each feature count data apply Laplace smoothing and compute conditional probability
    probDF[,grepl("prob", names(probDF))] <- apply(probDF[,grepl("prob", names(probDF))], 2, function(z) (z+1)/(probDF$total+nrow(probDF)))
    probDF <- probDF[,-which(names(probDF)=="total")]
    
    # compute total score and select best alternative
    probDF$score <- apply(probDF[,-1], 1, prod)
    pred <- append(pred, probDF[which.max(probDF$score),outputName])
  }
  
  return(pred)
}



# bayes classifier with limited search
predict.bayes.filter <- function (counts = list(), test = data.table(), filters = character()){
  
  pred <- character(0)
  
  # retrieve output name from model  
  outputName <- names(counts)[length(counts)]
  
  for (i in 1:nrow(test)){
    temp <- counts
    
    # if some filters are provided try them sequentially and break when possible alternatives have been restricted
    if (length(filters)>0){
      for (j in 1:length(filters)){
        filter_value <- as.character(test[i, filters[j], with=F])
        df <- temp[[filters[j]]]
        
        # if filter value has some history in dataset subset only previously assigned classes in count data
        if (filter_value %in% unique(df[,filters[j]])){
          cats <- unique(df[df[,filters[j]] %in% filter_value, outputName])
          temp <- lapply(temp, function(x) subset(x, x[,outputName] %in% cats))
          break
        }
      }
    }
    
    # subset count data for each feature with new data values and merge result into one table 
    temp[-length(temp)] <- mapply(function(x, y, z) list(subset(x, x[,z]==y)[,c(outputName, "prob")]), temp[-length(temp)], test[i,], names(test))
    probDF <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = outputName, all=TRUE), temp))
    probDF[is.na(probDF)] <- 0
    
    # for each feature count data apply Laplace smoothing and compute conditional probability
    probDF[,grepl("prob", names(probDF))] <- apply(probDF[,grepl("prob", names(probDF))], 2, function(z) (z+1)/(probDF$total+nrow(probDF)))
    probDF <- probDF[,-which(names(probDF)=="total")]
    
    # compute total score and select best alternative
    probDF$score <- apply(probDF[,-1], 1, prod)
    pred <- append(pred, probDF[which.max(probDF$score),outputName])
  }
  
  return(pred)
}

# some feature engineering
final$amountUSD <- final$amount/final$rate
final$amountCat <- NA
amount_cat <- list(c(0, 5),
                   c(6, 20),
                   c(21, 50),
                   c(51, 100),
                   c(101, 500),
                   c(501, 2000),
                   c(2001, Inf))
for (i in 1:length(amount_cat)){
  final$amountCat[round(final$amountUSD)>=amount_cat[[i]][1] & round(final$amountUSD)<=amount_cat[[i]][2]] <- i
} 
rm(amount_cat)

final$hour <- as.numeric(strftime(as.POSIXct(final$charged_at), format = "%H"))

# # algorithms comparison
# # uncomment to run, quite long loop, or load result file
#
# rf <- final[, list(freq=.N, num_cat = length(unique(category_name)),
#                    accuracy = NA, f1 = NA, type = "Random Forest"), by = user_id]
# 
# nb <- final[, list(freq=.N, num_cat = length(unique(category_name)),
#                    accuracy = NA, f1 = NA, type = "Naive Bayes"), by = user_id]
# 
# nbMerch <- final[, list(freq=.N, num_cat = length(unique(category_name)),
#                         accuracy = NA, f1 = NA, type = "Naive Bayes with merchant"), by = user_id]
# 
# nbFilt <- final[, list(freq=.N, num_cat = length(unique(category_name)),
#                        accuracy = NA, f1 = NA, type = "Naive Bayes with filter"), by = user_id]
# 
# for (i in 1:nrow(rf)){
#   
#   temp <- subset(final, final$user_id==rf$user_id[i])
#   s <- oneObsByClass(temp)
#   set.seed(12)
#   test_subset <- sample(c(1:nrow(temp))[-s], nrow(temp)*0.2)
#   train_subset <- append(s, sample(c(1:nrow(temp))[-c(s, test_subset)], (nrow(temp)-length(c(s, test_subset)))))
#
#   data <- temp
#   data$sign <- ifelse(data$sign=="MINUS", -1, 1)
#   data$mcc <- as.numeric(data$mcc)
#   test <- data[test_subset,]
#   train <- data[train_subset,]
#   
#   final.rf <- randomForest(category_name~ hour + mcc + sign + amountUSD + currency, data = train)
#   pred <- as.character(predict(final.rf, test, type = "class"))
#   test$category_name <- as.character(test$category_name)
#   rf$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   rf$f1[i] <- calculate.f1(test$category_name, pred)
#   
#
#   data <- temp
#   data$hour <- as.character(data$hour)
#   data$amountCat <- as.character(data$amountCat)
#   data$mcc <- as.character(data$mcc)
#   data$currency <- as.character(data$currency)
#   data$sign <- as.character(data$sign)
#   data$merchant_name <- as.character(data$merchant_name)
#   data$category_name <- as.character(data$category_name)
#   test <- data[test_subset,]
#   train <- data[train_subset,]
#   
#   features <- c("hour", "mcc", "sign", "amountCat", "currency")
#   final.nb <- fit.bayes("category_name", features, train)
#   pred <- predict.bayes(final.nb, test[,features,with=F])
#   nb$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   nb$f1[i] <- calculate.f1(test$category_name, pred)
#   
#   
#   features <- c("hour", "mcc", "merchant_name", "sign", "amountCat", "currency")
#   final.nb <- fit.bayes("category_name", features, train)
#   pred <- predict.bayes(final.nb, test[,features,with=F])
#   nbMerch$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   nbMerch$f1[i] <- calculate.f1(test$category_name, pred)
#   
#   
#   features <- c("hour", "mcc", "merchant_name", "sign", "amountCat", "currency")
#   final.nb <- fit.bayes("category_name", features, train)
#   pred <- predict.bayes.filter(final.nb, test[,features,with=F])
#   nbFilt$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   nbFilt$f1[i] <- calculate.f1(test$category_name, pred)
#   
#   print(i)
# }
# 
# graphDF <- Reduce(function(x, y) rbind(x, y), list(rf, nb, nbMerch, nbFilt))
# save(graphDF, file = "results_comparison.Rdata")

load("results_comparison.Rdata")

graphDF[is.na(graphDF)] <- 0

p1 <- ggplot(graphDF, aes(reorder(factor(type), accuracy, FUN = median), accuracy)) + 
  geom_boxplot(aes(fill = type)) +
  labs(x = element_blank(), y = "Accuracy", title = "Performance comparison") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none")

p2 <- ggplot(graphDF, aes(reorder(factor(type), accuracy, FUN = median), f1)) + 
  geom_boxplot(aes(fill = type)) +
  labs(x = "Algorithm used", y = "F1 macro-average") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none")

grid.arrange(p1, p2, ncol = 1)


# for following graphs we need users with at significant amount of data
final <- subset(final, final$user_id %in% user$user_id[user$freq>=250])

# # incremental approach graph
# # uncomment to run, quite long loop, otherwise just load result file
# 
# r <- final[, list(freq = .N, accuracy = 0, f1 = 0, percent = 1:10), by = user_id]
# 
# for (i in 1:nrow(r)){
#   temp <- subset(final, final$user_id %in% r$user_id[i])
#   temp <- temp[order(temp$charged_at),]
#   set.seed(12)
#   train_subset <- 1:round(nrow(temp)*0.5)
#   test_subset <- round(nrow(temp)*0.5):round((nrow(temp)*0.5)+(nrow(temp)*0.5*0.1*r$percent[i]))
#   
#   test <- temp[test_subset,]
#   train <- temp[train_subset,]
#   
#   features <- c("hour", "mcc", "merchant_name", "sign", "amountCat", "currency")
#   final.nb <- fit.bayes("category_name", features, train)
#   pred <- predict.bayes.filter(final.nb, test[,features,with=F], c("merchant_name", "mcc"))
#   
#   r$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   r$f1[i] <- calculate.f1(test$category_name, pred)
#   
#   print(i)
# }
# 
# save(r, file = "results_not_incremental.Rdata")

load("results_not_incremental.Rdata")

p1 <- ggplot(r, aes(factor(percent*10), accuracy)) + 
  geom_boxplot(aes(fill = percent), alpha = 0.7) +
  geom_smooth(aes(percent, accuracy), method = 'loess', col = "salmon", se = F, size = 1.5) +
  labs(x = element_blank(), y = "Accuracy", title = "Performance over time without incremental learning") +
  expand_limits(x = 1, y = 0) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none", 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())

p2 <- ggplot(r, aes(factor(percent*10), f1)) + 
  geom_boxplot(aes(fill = percent), alpha = 0.7) +
  geom_smooth(aes(percent, f1), method = 'loess', col = "salmon", se = F, size = 1.5) +
  labs(x = "Percent of validation data used", y = "F1 micro average") +
  expand_limits(x = 1, y = 0) +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 1)


# # incremental approach graph
# # uncomment to run, quite long loop, otherwise just load result file
#
# r <- final[, list(freq = .N, accuracy = 0, f1 = 0, percent = 1:10), by = user_id]
# 
# for (i in 1:nrow(r)){
#   temp <- subset(final, final$user_id %in% r$user_id[i])
#   set.seed(12)
#   train_subset <- 1:round(nrow(temp)*0.5+(nrow(temp)*0.5*(0.1*r$percent[i]-0.1)))
#   test_subset <- round(nrow(temp)*0.5+(nrow(temp)*0.5*(0.1*r$percent[i]-0.1))):round((nrow(temp)*0.5)+(nrow(temp)*0.5*0.1*r$percent[i]))
#   
#   test <- temp[test_subset,]
#   train <- temp[train_subset,]
#   
#   features <- c("hour", "mcc", "merchant_name", "sign", "amountCat", "currency")
#   final.nb <- fit.bayes("category_name", features, train)
#   pred <- predict.bayes.filter(final.nb, test[,features,with=F], c("merchant_name", "mcc"))
#   
#   r$accuracy[i] <- mean(pred == test$category_name, na.rm = T)
#   r$f1[i] <- calculate.f1(test$category_name, pred)
#   
#   print(i)
# }
# 
# save(r, file = "results_incremental.Rdata")

load("results_incremental.Rdata")

p1 <- ggplot(r, aes(factor(percent*10), accuracy)) + 
  geom_boxplot(aes(fill = percent), alpha = 0.7) +
  geom_smooth(aes(percent, accuracy), method = 'loess', col = "salmon", se = F, size = 1.5) +
  labs(x = element_blank(), y = "Accuracy", title = "Performance over time with incremental learning") +
  expand_limits(x = 1, y = 0) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none", 
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())

p2 <- ggplot(r, aes(factor(percent*10), f1)) + 
  geom_boxplot(aes(fill = percent), alpha = 0.7) +
  geom_smooth(aes(percent, f1), method = 'loess', col = "salmon", se = F, size = 1.5) +
  labs(x = "Percent of validation data used", y = "F1 micro average") +
  expand_limits(x = 1, y = 0) +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 1)


