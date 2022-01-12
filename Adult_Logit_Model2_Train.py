# Call in necessary functions

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
import pickle



#####################################################
# Load cleaned training data
# select columns to be used in in this model

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

datafile = "Adult_Cleaned_Train.csv"
X_train = pd.read_csv(datafile, usecols = columns)


# Scale the continuous data using standard scalar
scale = pd.DataFrame(X_train.loc[:,["age", "education_num", "hours_per_week"]])

standardization_scale = StandardScaler()
z = standardization_scale.fit_transform(scale)
Adult_scale = pd.DataFrame(z)

# save the scaling parameters for use on testing set
filename = 'Adult_Logit_Model2_Scaling.pkl'
pickle.dump(standardization_scale, open(filename,'wb'))


# replace existing columns with scaled columns
X_train.loc[:,"age"] = Adult_scale[0]
X_train.loc[:,"education_num"] = Adult_scale[1]
X_train.loc[:,"hours_per_week"] = Adult_scale[2]


# Seperate expert label
y_train = X_train['income_>50K']
X_train = X_train.drop('income_>50K', axis=1)


# Synthetically Balance the Classes
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

#######################################################
########### LOGISTIC CLASSIFIER #######################
#######################################################


# The expert label variable is bianary so a logistic model makes sense
# Training the Logistic Model
logitmodel = LogisticRegression(tol=.01, max_iter=500, multi_class='ovr')
logitmodel.fit(X_res, y_res)

# Save fitted model
filename = 'Adult_Logit_Model2.sav'
pickle.dump(logitmodel, open(filename, 'wb'))

