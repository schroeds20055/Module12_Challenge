# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
Due to the loan classifications being imballanced between healthy loans and risky loans, the analysis, from building a model, will help identify the borrowers creditworthiness.
* Explain what financial information the data was on, and what you needed to predict.
The financial information is historical lending activity from peer-to-peer lending services company. The purpose of this challenge is to build a model that can identify the creditworthiness of borrowers.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The variables that are being predicted are the value_counts, train_test_split, X_train, y_train and X_test. 
* Describe the stages of the machine learning process you went through as part of this analysis.
To create a logistic regression model with the original data, I Fit a logistic rgression model by using the training data then saved the predictions on the testing data labels by using the testing feature data and the fitted model. I then evaluated the models performanc by calculating the accuracy score, generating a confusion matrix and printing the classification report.

To predict the logistic regression model with resampled training data, I first wued the RandomOverSampler model to resample the data. Then I used the LogisticRegression classifyer and the ressampled data to fit the model and make the predictions. From there I evaluated the models performanc by calculating hte accuracy score, generating the confusion matrix and printing hte classification report.


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
I used Logistic_Regression, balanced_accuracy_score, classification_report_imbalanced, train_test_split, RandomOverSampler and confusion_matrix


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
Accuracy - 0.99
Precision:
    0 - 1.00
    1 - 0.84
Recall:
    0 - 0.99
    1 - 0.91


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
Accuracy - 0.99
Precision:
    0 - 0.99
    1 - 0.91
Recall:
    0 - 1.00
    1 - 0.84

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
From reviewing the classification reports of the two models, I would choose the Machine Learning Model 2. Although the accuracy for both models is 0.99, the precision of the MLM 2 has less of a spread between the 1's and 0's. and the average recall for both is at 0.99.
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Since the analysis is to see the creditworthiness of borrowers, it is more important to predict the 1's (high risk of default). 

If you do not recommend any of the models, please justify your reasoning.
