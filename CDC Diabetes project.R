# Load necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(readr)      # For reading CSV files
library(caret)      # For data splitting and modeling

# Set the path to the datasets
path_to_data_012 <- "C:/Users/syedi/OneDrive/Desktop/Healthcare Analytics/Project/Datasets/CDC Diabetes Health Indicators/archive/diabetes_012_health_indicators_BRFSS2015.csv"
path_to_data_5050 <- "C:/Users/syedi/OneDrive/Desktop/Healthcare Analytics/Project/Datasets/CDC Diabetes Health Indicators/archive/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
path_to_data_binary <- "C:/Users/syedi/OneDrive/Desktop/Healthcare Analytics/Project/Datasets/CDC Diabetes Health Indicators/archive/diabetes_binary_health_indicators_BRFSS2015.csv"

# Import the datasets
data_012 <- read_csv(path_to_data_012)
data_5050 <- read_csv(path_to_data_5050)
data_binary <- read_csv(path_to_data_binary)

# Basic Data Examination
# View the first few rows of each dataset
head(data_012)
head(data_5050)
head(data_binary)

# Structure of the datasets
str(data_012)
str(data_5050)
str(data_binary)

# Summary statistics for each dataset
summary(data_012)
summary(data_5050)
summary(data_binary)

library(dplyr)

# Recode all variables with descriptions and rename columns appropriately
data_visualization <- data_012 %>%
  mutate(
    # Recode binary variables and others as needed
    HighBP = factor(ifelse(HighBP == 1, "High BP", "No high BP")),
    HighChol = factor(ifelse(HighChol == 1, "High cholesterol", "No high cholesterol")),
    CholCheck = factor(ifelse(CholCheck == 1, "Cholesterol check in 5 years", "No cholesterol check in 5 years")),
    Smoker = factor(ifelse(Smoker == 1, "Smoker", "Non-smoker")),
    Stroke = factor(ifelse(Stroke == 1, "Had a stroke", "No stroke")),
    HeartDiseaseorAttack = factor(ifelse(HeartDiseaseorAttack == 1, "CHD or MI", "No CHD or MI")),
    PhysActivity = factor(ifelse(PhysActivity == 1, "Physically active", "No physical activity")),
    Fruits = factor(ifelse(Fruits == 1, "Consumes fruits", "Does not consume fruits")),
    Veggies = factor(ifelse(Veggies == 1, "Consumes vegetables", "Does not consume vegetables")),
    HvyAlcoholConsump = factor(ifelse(HvyAlcoholConsump == 1, "Heavy alcohol consumption", "No heavy alcohol consumption")),
    AnyHealthcare = factor(ifelse(AnyHealthcare == 1, "Has healthcare coverage", "No healthcare coverage")),
    NoDocbcCost = factor(ifelse(NoDocbcCost == 1, "Cost issues accessing doctor", "No cost issues accessing doctor")),
    GenHlth = factor(case_when(
      GenHlth == 1 ~ "Excellent",
      GenHlth == 2 ~ "Very good",
      GenHlth == 3 ~ "Good",
      GenHlth == 4 ~ "Fair",
      GenHlth == 5 ~ "Poor",
      TRUE         ~ "Unknown"
    )),
    DiffWalk = factor(ifelse(DiffWalk == 1, "Difficulty walking", "No difficulty walking")),
    Sex = factor(ifelse(Sex == 1, "Male", "Female")),
    Diabetes_012 = factor(case_when(
      Diabetes_012 == "0" ~ "No diabetes",
      Diabetes_012 == "1" ~ "Prediabetes",
      Diabetes_012 == "2" ~ "Diabetes",
      TRUE                ~ "Unknown"
    )),
    Age = case_when(
      Age == 1  ~ "18 to 24",
      Age == 2  ~ "25 to 29",
      Age == 3  ~ "30 to 34",
      Age == 4  ~ "35 to 39",
      Age == 5  ~ "40 to 44",
      Age == 6  ~ "45 to 49",
      Age == 7  ~ "50 to 54",
      Age == 8  ~ "55 to 59",
      Age == 9  ~ "60 to 64",
      Age == 10 ~ "65 to 69",
      Age == 11 ~ "70 to 74",
      Age == 12 ~ "75 to 79",
      Age == 13 ~ "80 or older",
      TRUE      ~ "Unknown"
    ),
    Education = factor(case_when(
      Education == 1 ~ "Never attended school or only kindergarten",
      Education == 2 ~ "Grades 1 through 8 (Elementary)",
      Education == 3 ~ "Grades 9 through 11 (Some high school)",
      Education == 4 ~ "Grade 12 or GED (High school graduate)",
      Education == 5 ~ "College 1 year to 3 years (Some college or technical school)",
      Education == 6 ~ "College 4 years or more (College graduate)",
      TRUE           ~ "Other"
    )),
    Income = factor(case_when(
      Income == 1 ~ "Less than $10,000",
      Income == 2 ~ "$10,000 to less than $15,000",
      Income == 3 ~ "$15,000 to less than $20,000",
      Income == 4 ~ "$20,000 to less than $25,000",
      Income == 5 ~ "$25,000 to less than $35,000",
      Income == 6 ~ "$35,000 to less than $50,000",
      Income == 7 ~ "$50,000 to less than $75,000",
      Income == 8 ~ "$75,000 or more",
      TRUE         ~ "Unknown"
    ))
  ) %>%
  # Reorder columns to match original dataset structure
  select(Diabetes_012, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income)

# Print the structure of the modified data to verify changes
str(data_visualization)


# Optionally, remove the original 'Age', 'Education', 'Income' columns if they are not needed
data_modified <- select(data_modified, -Age, -Education, -Income)

# Print out the first few rows of the new dataframe to check
head(data_modified)

# ########################### Data Visualization ###########################
# ########################### Data Visualization ###########################
# ########################### Data Visualization ###########################

# Plotting the distribution of the target variable in each dataset
ggplot(data_012, aes(x = factor(Diabetes_012))) + 
  geom_bar() +
  labs(title = "Distribution of Diabetes Classes in 012 Dataset", x = "Diabetes Class", y = "Count")

ggplot(data_5050, aes(x = factor(Diabetes_binary))) + 
  geom_bar() +
  labs(title = "Distribution of Diabetes Classes in 5050 Split Dataset", x = "Diabetes Class", y = "Count")

ggplot(data_binary, aes(x = factor(Diabetes_binary))) + 
  geom_bar() +
  labs(title = "Distribution of Diabetes Classes in Binary Dataset", x = "Diabetes Class", y = "Count")
library(ggplot2)

# BMI Distribution by Diabetes Status for each dataset
ggplot(data_012, aes(x = BMI, fill = factor(Diabetes_012))) +
  geom_histogram(bins = 30, alpha = 0.6) +
  labs(title = "BMI Distribution by Diabetes Status (012 Dataset)",
       x = "BMI",
       fill = "Diabetes Status")

ggplot(data_5050, aes(x = BMI, fill = factor(Diabetes_binary))) +
  geom_histogram(bins = 30, alpha = 0.6) +
  labs(title = "BMI Distribution by Diabetes Status (5050 Split Dataset)",
       x = "BMI",
       fill = "Diabetes Status")

ggplot(data_binary, aes(x = BMI, fill = factor(Diabetes_binary))) +
  geom_histogram(bins = 30, alpha = 0.6) +
  labs(title = "BMI Distribution by Diabetes Status (Binary Dataset)",
       x = "BMI",
       fill = "Diabetes Status")

# Age Distribution by Diabetes Status for each dataset
#ggplot(data_modified, aes(x = Recoded_Age, fill = factor(data_modified))) +
#  geom_histogram(bins = 13, alpha = 0.6) +
#  labs(title = "Age Distribution by Diabetes Status (012 Dataset)",
#       x = "Age",
#       fill = "Diabetes Status")


# Age Distribution by Diabetes Status for the updated dataset
ggplot(data_visualization, aes(x = Age, fill = Diabetes_012)) +
  geom_bar(position = "dodge", alpha = 0.6) +
  labs(title = "Age Distribution by Diabetes Status (012 Dataset)",
       x = "Age Group",
       fill = "Diabetes Status") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Relationship between Physical Activity and Diabetes
#ggplot(data_012, aes(x = factor(PhysActivity), fill = factor(Diabetes_012))) +
#  geom_bar(position = "fill") +
#  labs(title = "Physical Activity vs. Diabetes Status (012 Dataset)",
#       x = "Physical Activity (0 = No, 1 = Yes)",
#       y = "Proportion",
#       fill = "Diabetes Status")

# Relationship between Physical Activity and Diabetes
ggplot(data_visualization, aes(x = PhysActivity, fill = Diabetes_012)) +
  geom_bar(position = "fill") +
  labs(title = "Physical Activity vs. Diabetes Status (012 Dataset)",
       x = "Physical Activity",
       y = "Proportion",
       fill = "Diabetes Status")


# Relationship between General Health and Diabetes
#ggplot(data_012, aes(x = factor(GenHlth), fill = factor(Diabetes_012))) +
#  geom_bar(position = "fill") +
#  labs(title = "General Health vs. Diabetes Status (012 Dataset)",
#       x = "General Health (1 = Excellent, 5 = Poor)",
#       y = "Proportion",
#       fill = "Diabetes Status")

ggplot(data_visualization, aes(x = GenHlth, fill = Diabetes_012)) +
  geom_bar(position = "fill") +
  labs(title = "General Health vs. Diabetes Status (012 Dataset)",
       x = "General Health",
       y = "Proportion",
       fill = "Diabetes Status")

library(GGally)
library(corrplot)

# Correlation Matrix Heatmap
cor_matrix_012 <- cor(data_012[, sapply(data_012, is.numeric)], use = "complete.obs")
corrplot(cor_matrix_012, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 90,  # Set text rotation for y-axis labels to 90 degrees
         title = "Correlation Matrix of Variables (5050 Split Dataset)")

# Facet Grid for BMI by Age and Diabetes Status
ggplot(data_012, aes(x = Age, y = BMI)) +
  geom_point(aes(color = factor(Diabetes_binary)), alpha = 0.6) +
  facet_wrap(~Diabetes_binary) +
  labs(title = "BMI vs. Age by Diabetes Status (5050 Split Dataset)",
       x = "Age",
       y = "BMI",
       color = "Diabetes Status")

# Interaction Plot for General Health and Physical Activity
interaction.plot(data_012$GenHlth, data_012$PhysActivity,
                 data_012$Diabetes_012,
                 main = "Interaction of General Health and Physical Activity on Diabetes Status",
                 xlab = "General Health",
                 ylab = "Average Diabetes Status",
                 legend = TRUE,
                 col = c("red", "blue"))

library(ggplot2)
library(tidyr)

# Reshape data to long format for faceting
data_long <- data_012 %>%
  select(Diabetes_binary, BMI, Age, GenHlth) %>%
  pivot_longer(cols = -Diabetes_binary, names_to = "Variable", values_to = "Value")

# Create a density plot with faceting
ggplot(data_long, aes(x = Value, fill = factor(Diabetes_binary), alpha = 0.5)) +
  geom_density(adjust = 1.5) +
  facet_wrap(~Variable, scales = "free", ncol = 1) +
  scale_fill_manual(values = c("blue", "red"), labels = c("No Diabetes", "Diabetes")) +
  labs(title = "Density Plots by Variable and Diabetes Status",
       x = "Value",
       y = "Density") +
  theme_minimal()

numeric_columns <- sapply(data_5050, is.numeric)
correlations <- cor(data_5050[, numeric_columns])['Diabetes_binary', ]
correlations <- correlations[order(correlations, decreasing = TRUE)]  # Sort the correlations for better visualization

# Remove Diabetes_binary correlation with itself to avoid trivial results
correlations <- correlations[correlations != 1]

# Create a bar graph to display the correlations
ggplot(data = data.frame(Variable = names(correlations), Correlation = correlations), aes(x = Variable, y = Correlation, fill = Correlation > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Correlation with Diabetes_binary", x = "Variables", y = "Correlation Coefficient") +
  scale_fill_manual(values = c("red", "blue"), labels = c("Negative Correlation", "Positive Correlation")) +
  theme_minimal() +
  theme(legend.title = element_text(size = 12), legend.text = element_text(size = 10))

# ########################### FEATURE SELECTION ###########################
# ########################### FEATURE SELECTION ###########################
# ########################### FEATURE SELECTION ###########################

corr_matrix <- cor(data_012[, sapply(data_012, is.numeric)])
corrplot(corr_matrix, method = "circle", type = "upper", order = "hclust", tl.cex = 0.7)
high_corr <- findCorrelation(corr_matrix, cutoff = 0.5)
selected_features <- data_012[, -high_corr]


#library(caret)
#control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
#results <- rfe(data_012[, -1], data_012$Diabetes_012, sizes = c(1:10), rfeControl = control)
#print(results)
#selected_vars <- predictors(results)


# ########################### MODELLING ###########################
# ########################### MODELLING ###########################
# ########################### MODELLING ###########################

# Preprocess data

#data_012 <- data_012 %>%
#  mutate(Diabetes_012 = factor(Diabetes_012, levels = c(0, 1, 2)),
#         HighBP = factor(HighBP),
#         HighChol = factor(HighChol),
#         CholCheck = factor(CholCheck),
#         Smoker = factor(Smoker),
#         Stroke = factor(Stroke),
#         HeartDiseaseorAttack = factor(HeartDiseaseorAttack),
#         PhysActivity = factor(PhysActivity),
#         Fruits = factor(Fruits),
#         Veggies = factor(Veggies),
#         HvyAlcoholConsump = factor(HvyAlcoholConsump),
#         AnyHealthcare = factor(AnyHealthcare),
#         NoDocbcCost = factor(NoDocbcCost),
#         DiffWalk = factor(DiffWalk),
#         Sex = factor(Sex))

#data_binary$Diabetes_binary <- as.factor(data_binary$Diabetes_binary)

########################### MODELLING W/data_binary ###########################
########################### MODELLING W/data_binary ###########################
########################### MODELLING W/data_binary ###########################

set.seed(42)
split <- createDataPartition(data_binary$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- data_binary[split, ]
test_data <- data_binary[-split, ]

train_data$Diabetes_binary <- as.factor(train_data$Diabetes_binary)
test_data$Diabetes_binary <- as.factor(test_data$Diabetes_binary)

model <- glm(Diabetes_binary ~ ., data = train_data, family = binomial())
summary(model)

predicted_probabilities <- predict(model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, "1", "0")
predicted_classes <- factor(predicted_classes, levels = levels(test_data$Diabetes_binary))

conf_matrix <- confusionMatrix(predicted_classes, test_data$Diabetes_binary)
print(conf_matrix)

roc_result <- roc(test_data$Diabetes_binary, as.numeric(predicted_probabilities))
print(auc(roc_result))
plot(roc_result)

#predicted_probabilities <- predict(model, type = "response")
#predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
#table(predicted_classes, data_binary$Diabetes_binary)

#predicted_classes <- as.factor(predicted_classes)

#confusionMatrix(predicted_classes, data_binary$Diabetes_binary)
#conf_mat <- confusionMatrix(predicted_classes, data_binary$Diabetes_binary)

conf_matrix$overall['Accuracy']
conf_matrix$byClass['Precision']
conf_matrix$byClass['Sensitivity']  # Recall is the same as Sensitivity
conf_matrix$byClass['Specificity']

########################### MODELLING W/FEATURE SELECTION ###########################

data_binary$Diabetes_binary <- as.factor(data_binary$Diabetes_binary)

model_filtered <- glm(Diabetes_binary ~ . -Veggies -PhysActivity -Income -HvyAlcoholConsump -Fruits -Education, 
                      data = data_binary, family = binomial())
summary(model_filtered)

predicted_probabilities_filtered <- predict(model_filtered, type = "response")
predicted_classes_filtered <- ifelse(predicted_probabilities_filtered > 0.5, 1, 0)
predicted_classes_filtered <- as.factor(predicted_classes_filtered)

conf_mat_filtered <- confusionMatrix(predicted_classes_filtered, data_binary$Diabetes_binary)
print(conf_mat_filtered)

conf_mat_filtered$overall['Accuracy']
conf_mat_filtered$byClass['Precision']
conf_mat_filtered$byClass['Sensitivity']  
conf_mat_filtered$byClass['Specificity']

########################### MODELLING W/data_5050 ###########################
########################### MODELLING W/data_5050 ###########################
########################### MODELLING W/data_5050 ###########################

data_5050$Diabetes_binary <- as.factor(data_5050$Diabetes_binary)

model <- glm(Diabetes_binary ~ ., data = data_5050, family = binomial())
summary(model)

predicted_probabilities <- predict(model, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
table(predicted_classes, data_5050$Diabetes_binary)

predicted_classes <- as.factor(predicted_classes)

confusionMatrix(predicted_classes, data_5050$Diabetes_binary)
conf_mat <- confusionMatrix(predicted_classes, data_5050$Diabetes_binary)

conf_mat$overall['Accuracy']
conf_mat$byClass['Precision']
conf_mat$byClass['Sensitivity']  # Recall is the same as Sensitivity
conf_mat$byClass['Specificity']

########################### MODELLING W/FEATURE SELECTION ###########################

data_binary$Diabetes_binary <- as.factor(data_binary$Diabetes_binary)

model_filtered <- glm(Diabetes_binary ~ . -Veggies -PhysActivity -Income -HvyAlcoholConsump -Fruits -Education, 
                      data = data_binary, family = binomial())
summary(model_filtered)

predicted_probabilities_filtered <- predict(model_filtered, type = "response")
predicted_classes_filtered <- ifelse(predicted_probabilities_filtered > 0.5, 1, 0)
predicted_classes_filtered <- as.factor(predicted_classes_filtered)

conf_mat_filtered <- confusionMatrix(predicted_classes_filtered, data_binary$Diabetes_binary)
print(conf_mat_filtered)

conf_mat_filtered$overall['Accuracy']
conf_mat_filtered$byClass['Precision']
conf_mat_filtered$byClass['Sensitivity']  
conf_mat_filtered$byClass['Specificity']





















































# Create a training and test dataset
set.seed(42)
trainIndex <- createDataPartition(data_012$Diabetes_012, p = 0.7, list = FALSE)
train_data <- data_012[trainIndex, ]
test_data <- data_012[-trainIndex, ]

train_data_filtered <- train_data %>%
  select(-c(Veggies, PhysActivity, Income, HvyAlcoholConsump, Fruits, Education))

# Linear regression model
train_data$Diabetes_012 <- as.numeric(as.character(train_data$Diabetes_012))
fit <- lm(Diabetes_012 ~ ., data = train_data)

train_data_filtered$Diabetes_012 <- as.numeric(as.character(train_data_filtered$Diabetes_012))
fit2 <- lm(Diabetes_012 ~ ., data = train_data_filtered)

summary(fit)
summary(fit2)

predictions <- predict(fit, newdata = test_data)
predictions2 <- predict(fit2, newdata = test_data)

rounded_predictions <- round(predictions)
rounded_predictions2 <- round(predictions2)

factor_predictions <- factor(rounded_predictions, levels = c(0, 1, 2))
factor_predictions2 <- factor(rounded_predictions2, levels = c(0, 1, 2))
factor_actuals <- factor(test_data$Diabetes_012, levels = c(0, 1, 2))

conf_mat1 <- confusionMatrix(factor_predictions, factor_actuals)
conf_mat2 <- confusionMatrix(factor_predictions2, factor_actuals)

print(conf_mat1)
print(conf_mat2)

conf_mat1 <- confusionMatrix(factor_predictions, factor_actuals, mode = "everything")
conf_mat2 <- confusionMatrix(factor_predictions2, factor_actuals, mode = "everything")

print(conf_mat1$overall['Accuracy'])
print(conf_mat1$byClass['Sensitivity'])
print(conf_mat1$byClass['Specificity'])

print(conf_mat2$overall['Accuracy'])
print(conf_mat2$byClass['Sensitivity'])
print(conf_mat2$byClass['Specificity'])

# Logistic regression model
train_data$Diabetes_012 <- factor(train_data$Diabetes_012, levels = c(0, 1, 2))
model <- train(Diabetes_012 ~ ., data = train_data, method = "multinom", trControl = trainControl(method = "cv", number = 5))

train_data_filtered$Diabetes_012 <- factor(train_data_filtered$Diabetes_012, levels = c(0, 1, 2))
model2 <- train(Diabetes_012 ~ ., data = train_data_filtered, method = "multinom", trControl = trainControl(method = "cv", number = 5))

summary(model)
summary(model2)

predictions3 <- predict(model, newdata = test_data)

factor_vars <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex")
for (var in factor_vars) {
  test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data_filtered[[var]]))
}
predictions4 <- predict(model2, newdata = test_data)

test_data$Diabetes_012 <- factor(test_data$Diabetes_012, levels = c("0", "1", "2"))
confusionMatrix(predictions3, test_data$Diabetes_012)
confusionMatrix(predictions4, test_data$Diabetes_012)


