# Forecasting Rain Using Machine and Deep Learning

## Background : 
Weather forecasting is the application of science and technology to predict the conditions of the atmosphere for a given location and time. Weather forecasts are made by collecting quantitative data about the current state of the atmosphere at a given place and using meteorology to project how the atmosphere will change. One of the types of weather forecasting is rain forecasting.

## Problem Statement :
Weather forecasting has always been a tedious job as it requires to observe a lot of variables such as “humidity”, “wind speed”, “wind direction” and etc. Such a large number of variables has made human-produced-prediction fairly inaccurate and inefficient. To make the forecasting accurate and efficient, we hope to apply machine learning algorithms on large datasets, to uncover the hidden pattern within, in order to produce very accurate forecasts. Our proposed model explores the prediction of occurrence of rain the next day using various machine learning models.

## Solution : 
In this proposed solution, we explore several machine learning models to predict whether it will rain tomorrow. The following model is being trained and tested using a dataset provided by the Bureau of Meteorology, Australia. This dataset contains 10 years of daily weather observations from many locations across Australia. We aimed to produce a predictive model which provides fairly accurate predictions with the least amount of training time and human efforts, to maximize the outputs and resources. 

## Project Plan
1. The dataset was obtained from Kaggle. The datasets were compiled from Bureau of Meteorology Australia. Data understanding and Exploratory Data Analysis was carried out on the data. Categorical and numerical data was then identified and explored. Features of the dataset were then engineered to suit the needs of the solution. 

2. The explored and engineered data was preprocessed. 
    - The data was split into training and testing sets. 
    - Both training and testing sets were checked for duplication and missing values were checked. 
        - Missing categorical data was imputed using mode. 
        - Missing numerical data was imputed using median. 
    - Outliers were identified and cleaned. 
    - Binary encoding was performed on the categorical data. 
    - Features were then scaled to map the variables onto the same scale. 

3. 12 baseline models were trained, including Logistic Regression, Decision Tree Classifier, K-Neighbors Classifier, MLP Classifier, Gaussian-Naive Bayes, Bernoulli-Naive Bayes, Multinomial Naive Bayes, Support Vector Classifier, SGD Classifier, Random Forest Classifier, Gradient Boosting Classifier and AdaBoost Classifier. 
    - Nested cross validation was used to perform feature selection and hyperparameter tuning simulteneously, and then train the models, in order to mitigate overfitting. 
    - SelectKBest was used for feature selection.
    - RandomizedSearchCV was used for hyperparameter tuning.
    - StratifiedKFold was used for cross validation as the dataset is highly imbalanced.
    - The models were evaluated based on Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC, Cohen Kappa Score, Fit Time and Score Time. 

4. The best performing models were shortlisted by focusing on the ‘Balanced Accuracy’, ‘F1’, ‘PR-AUC’, ‘ROC-AUC’, ‘Cohen's Kappa Score’ and ‘Time Taken For Training’ of the various models. These metrics were chosen as they are most suited for a highly imbalanced dataset. 
    - Learning curves were also used to assess which models had overfitting.

5. The 3 best performing models, which are Logistic Regression, Support Vector Classifier and Gradient Boosting Classifier, were chosen for model stacking. 3 Ensemble Models were created using the combinations of these three classifiers. 

6. Lastly the 3 best performing models and the 3 Ensmble Models were evaluated using the same metrics as above and learning curves and the results were compare to determine the best model.

## Conclsuion and Results
- By considering the balanced accuracy, F1, PR-AUC, ROC-AUC, Cohen's Kappa, Training Time and learning curve, the best model is the Stacked Classifier 2, which consists of the MLP Classifier and SVC models with default parameters as level 0 and the SGD Classifier model with default parameters as level 1.
    - The model uses all the parameters but V1.

- The model has a balanced accuracy of 0.893828, F1 score of 0.913245, PR-AUC of 0.811808, ROC-AUC of 0.936575, Cohen's Kappa score of 0.826491 and Training Time of 2613.324426s.

- The model is able to generalise well on the testing data and can predict credit card fraud with an accuracy of 0.999440.
