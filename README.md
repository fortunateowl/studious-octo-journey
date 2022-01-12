# studious-octo-journey
All of the assignments, plus some additional work, from the UW data science program using the Adult census data set.

To get started with this repository I am uploading the initial data cleaning and a couple of early models, as well as the write up.  More will be added as it become ready for public consumption.

To date, the repository contains:
1. **Adult_Data_Cleaning.py** - loads data from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data and performs initial cleaning, including filling missing values, reducing the number of values in categorical features. Also outputs training and testing set for use in modeling.
2. **DataClean.py** - file with a class that contains functions for common data cleaning tasks.
3. **AccuracyMeasures.py** - file with a class that compiles and outputs results of bianary classification models.  Outputs include the ROC in .png format, the confusion matrix as a .txt file with latex tabular code, and a results (Accuracy, Recall, etc.) .txt file with latex tabular code.
4. **Adult_Logit_Model1_Train.py** - using the training data set as an input, it is an initial, naive, logistic model using almost all of the features in the data set.  Outputs scaling parameters and model.
5. **Adult_Logit_Model1_Test.py** - using the outputs of Adult_Logit_Model1_Train.py and the testing data set, uses predict_proba and applies AccuracyMeasure.py to the model to output results.
6. **Adult_Logit_Model2_Train.py** - using the training data set as an input, classes are balanced with SMOTE, but the model is the same as Adult_Logit_Model1_Train.py. Outputs scaling parameters and model.
7. **Adult_Logit_Model2_Test.py** - using the outputs of Adult_Logit_Model2_Train.py and the testing data set, uses predict_proba and applies AccuracyMeasure.py to the model to output results.
8. **Adult_Results.tex** - compiles the results output in the testing code into a .pdf with a brief write up.
9. All of the outputs from all of the above code.
