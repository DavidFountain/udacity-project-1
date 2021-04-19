# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains information about bank customers and includes 20 predictor variables, including marital status, age and job, and the categorical target variable 'y' containing values 'yes' and 'no'. We are seeking to predict instances of 'yes', so this problem is a binary classification problem. Adding further complications, the classes are imbalanced as only 11% contain the target 'yes' and 89% are 'no'.

The best performing model was the VotingEnsemble model trained using AutoML, with a final accuracy score of 0.9169.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
The AutoML run trained 68 individual models (and 2 ensembles) using a variety of ML algorithms, including logistic regression, light GBM, random forest and SGD. Each pipeline included data scaling before feeding into the model, testing standard sclaing, maximum absolute scaling and sparse normalizer. The final model was a voting ensemble, containing 6 ensembled light GBM algorithms with weights 0.2727272727272727, 0.09090909090909091, 0.2727272727272727, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091 and alpha value of 0.5789473684210527 and lambda 0.631578947368421. The single best performing modelhad an accuracy of 0.9150899381918383, which means that the ensemble method increased the best previous score by 0.0018. 

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The AutoML run outperformed the logistic regression model trained using hyperdrive by about 0.01, which is a significant improvement. The main differences between the 2 runs were the variety and number of models trained during the AutoML run. The hyperdrive experiment only changed the parameters of the inverse regularization strength and max iterations, in a predefined small search space, whereas the AutoML run also tested a number of different data processing methods and ML algorithms, including more sophisticated gradient boosting algorithms. This is the main reason that the AutoML model performed better than the hyperdrive experiment (although the TruncatedSVDWrapper LogisticRegression trained in AutoML also outperformed the hyperdrive run, with an accuracy of 0.91).

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
As highlighted in the AutoML run, the dataset contains a large class imbalance in the target variable, with a ratio of close to 1:9 for positive and negative observations. This can cause issues, as there are only a few positive positive cases for the model to train on. The effect of this is that by optimising the accuracy, there is a risk that the models will just predict the negative class for each instance, as this will still return a high accuracy score. Instead, other metrics should be used for classification problems, such as weighted AUC or F-score. Furthermore, resampling methods like oversampling of the minority class or SMOTE can be used to rebalance the data before training. This typically helps the models to perform better on the minority class and increases the overall performance metric.
