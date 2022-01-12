# Call in necessary functions
import pandas as pd
import pickle
from AccuracyMeasures import AccuracyMeasures as AM



#####################################################
# Load cleaned testing data
# select same columns for the training code
columns = ['age', 'education_num', 'hours_per_week',
       'workclass_gov', 'workclass_not_emp', 'workclass_self_emp',
       'marital_status_Married_AF_spouse', 'marital_status_Never_married',
       'marital_status_Separated', 'marital_status_Widowed',
       'occupation_Armed_Forces', 'occupation_Craft_repair',
       'occupation_Exec_managerial', 'occupation_Farming_fishing',
       'occupation_Handlers_cleaners', 'occupation_Machine_op_inspct',
       'occupation_Other_service', 'occupation_Priv_house_serv',
       'occupation_Prof_specialty', 'occupation_Protective_serv',
       'occupation_Sales', 'occupation_Tech_support',
       'occupation_Transport_moving', 'relationship_Not_in_family',
       'relationship_Other_relative', 'relationship_Own_child',
       'relationship_Unmarried', 'relationship_Wife',
       'race_Asian_Pac_Islander', 'race_Black', 'race_Other',
       'race_White', 'sex_Male', 'native_country_Canada',
       'native_country_Central_and_Carib', 'native_country_Europe',
       'native_country_Mexico', 'native_country_South',
       'native_country_South_America', 'native_country_US', 'income_>50K',
       'capital_inc_None', 'capital_inc_Small_gain',
       'capital_inc_Small_loss']


datafile = "Adult_Cleaned_Test.csv"
X_test = pd.read_csv(datafile, usecols = columns)


# Load scaling parameters from trainins set
filename = 'Adult_Logit_Model2_Scaling.pkl'
standardization_scale = pickle.load(open(filename,'rb'))


scale = pd.DataFrame(X_test.loc[:,["age", "education_num", "hours_per_week"]])

z = standardization_scale.transform(scale)
Adult_scale = pd.DataFrame(z)

# replace continuous columns with scaled data
X_test.loc[:,"age"] = Adult_scale[0]
X_test.loc[:,"education_num"] = Adult_scale[1]
X_test.loc[:,"hours_per_week"] = Adult_scale[2]


# Seperate expert label 
y_test = X_test['income_>50K']
X_test = X_test.drop('income_>50K', axis=1)

Adult_test=pd.DataFrame()
Adult_test.loc[:, 'income_>50K_actual'] = y_test.copy()



# Testing the Logistic Model
# load the trained model
filename = 'Adult_Logit_Model2.sav'
logitmodel = pickle.load(open(filename, 'rb'))

Adult_test.loc[:, 'income_>50K_lg_prob'] = logitmodel.predict_proba(X_test)[:,1]

# Set the probability threshold for the predicted values
thresh = 0.5

print ()
print ("###################################################")
print ("Logistic Regression Classifier Accuracy Measures")
print ("###################################################")
print ()
print ("Parameters:")
print ("\ttol = 0.01")
print ("\tmax_iter = 500")
print ("\tmulti_class = ovr")
print ("\tprobability threshold = 0.35")
print ()

# Innitiate function to measure accuracy of the classifier and output results

Output = AM()

Adult_test.loc[:, 'income_>50K_lg_predict'] = Output.AccuracyMeasures(Adult_test.loc[:, 'income_>50K_actual'], Adult_test.loc[:, 'income_>50K_lg_prob'], thresh, "Logistic Model 2")

