## install libraries
packages <- c("tidyverse", "dplyr", "caret", "randomForest")

lapply(packages, function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }  
})

## read the data
the_data <- read_csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"), 
                 col_names = FALSE)
colnames(the_data) <- c("class", "age", "menopause", "tumor_size", 
                    "inv_nodes", "node_caps", "deg_malig", 
                    "breast", "breast_quad", "irradiat")
str(the_data)

##convert to valid names for the sake of modeling later on
the_data$class <- make.names(the_data$class)

##convert characters to factors
the_data <- the_data %>% mutate(across(where(is.character), as.factor))

colSums(is.na(the_data))
## no na values

## cross tabulate and visualise
round(prop.table(table(the_data$class, the_data$age), 1)*100, 2)
ggplot(the_data) + 
  geom_bar(aes(age, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8))

ggplot(the_data) + 
  geom_bar(aes(menopause, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8))

ggplot(the_data) + 
  geom_bar(aes(tumor_size, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90))

ggplot(the_data) + 
  geom_bar(aes(breast, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8))

ggplot(the_data) + 
  geom_bar(aes(breast_quad, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8))

ggplot(the_data) + 
  geom_bar(aes(deg_malig, fill = class), position = "fill") + 
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 8))

chi <- lapply(the_data[, -1], function(x) chisq.test(the_data[, 1], x))
chi <- do.call(rbind, chi)[,c(1,3)]
chi <- as.data.frame(chi)
chi$p.value <- format(as.numeric(chi$p.value), scientific = FALSE)
chi %>% arrange(p.value) %>% mutate(significance = case_when(
  p.value < 0.05 & p.value >= 0.01 ~ "*",
  p.value < 0.01 & p.value >= 0.001 ~ "**",
  p.value < 0.001 ~ "***",
  p.value >= 0.05 ~ ""
))

## Machine learning

set.seed(1982, sample.kind = "Rounding")

test_index <- createDataPartition(the_data$class, p = 0.5, list = FALSE)
train_set <- the_data[-test_index, ]
test_set <- the_data[test_index, ]


model_glm <- train(class ~ ., data = train_set, method = "glm")
yhat_glm <- predict(model_glm, test_set, type = "raw")

model_knn <- train(class ~ ., data = train_set, method = "knn",
                   metric = "Spec",
                   tuneGrid = data.frame(k = seq(1, 20, 1)),
                   trControl = trainControl(method = "cv", 
                                            number = 10, 
                                            p = 0.9,
                                            classProbs = TRUE,
                                            summaryFunction = twoClassSummary))
ggplot(model_knn, highlight = TRUE)
yhat_knn <- predict(model_knn, test_set, type = "raw")

model_rf <- train(class ~ ., data = train_set, method = "rf", metric = "Spec",
                  tuneGrid = data.frame(.mtry = seq(1, 9, 1)),
                  trControl = trainControl(method = "cv", 
                                           number = 10, 
                                           p = 0.9,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary))
ggplot(model_rf, highlight = TRUE)
yhat_rf <- predict(model_rf, test_set, type = "raw")

results <- data.frame(Method = "GLM", 
                      Accuracy = confusionMatrix(yhat_glm, test_set$class)$overall["Accuracy"],
                      Sensitivity = confusionMatrix(yhat_glm, test_set$class)$byClass[1],
                      Specificity = confusionMatrix(yhat_glm, test_set$class)$byClass[2])

results <- results %>% add_row(Method = "KNN", Accuracy = confusionMatrix(yhat_knn, test_set$class)$overall["Accuracy"],
                               Sensitivity = confusionMatrix(yhat_knn, test_set$class)$byClass[1],
                               Specificity = confusionMatrix(yhat_knn, test_set$class)$byClass[2])

results <- results %>% add_row(Method = "Random forest", Accuracy = confusionMatrix(yhat_rf, test_set$class)$overall["Accuracy"],
                               Sensitivity = confusionMatrix(yhat_rf, test_set$class)$byClass[1],
                               Specificity = confusionMatrix(yhat_rf, test_set$class)$byClass[2])

## ensemble
collect <- data.frame(GLM = yhat_glm, KNN = yhat_knn, RF = yhat_rf)
vote <- rowSums(collect == "no.recurrence.events")
yhat_ensemble <- ifelse(vote > 1, "no.recurrence.events", "recurrence.events")
##convert to factor
yhat_ensemble <- factor(yhat_ensemble, levels = levels(test_set$class))

results <- results %>% add_row(Method = "Ensemble", Accuracy = confusionMatrix(yhat_ensemble, test_set$class)$overall["Accuracy"],
                               Sensitivity = confusionMatrix(yhat_ensemble, test_set$class)$byClass[1],
                               Specificity = confusionMatrix(yhat_ensemble, test_set$class)$byClass[2])

results

fourfoldplot(confusionMatrix(yhat_knn, 
                             test_set$class)$table,
             color = c("#CC6666", "#99CC99"),
             conf.level = 0,
             main = "KNN Confusion Matrix")

## variable importance
mtry_best <- model_rf$bestTune$mtry
model_randomforest <- randomForest(class ~ ., data = train_set,
                                   mtry = mtry_best)
as.data.frame(importance(model_randomforest)) %>% 
  arrange(desc(MeanDecreaseGini))
