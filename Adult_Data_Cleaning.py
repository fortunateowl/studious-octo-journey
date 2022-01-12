# This program handles the initial cleaning of the Adult dataset from the
# UCI machine learning archive.  This includes reducing the number of categories
# in the catigorical features and filling in missing values.  It also outputs
# training and testing sets for use in later modeling.

# Import functions
import pandas as pd
from DataClean import DataClean as DC
from sklearn.model_selection import train_test_split
import os



#####################################################
# Get Adult.data from the UCI machine learning archive

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
Adult.columns = ["age", "workclass", "fnlwgt", "education", "education_num", 
                "marital_status", "occupation", "relationship", "race", "sex", 
                "capital_gain", "capital_loss", "hours_per_week", "native_country",
                "income"]
#####################################################

# Strip annoying leading spaces from strings.
for i in range (0,Adult.shape[1]):
    
    if pd.api.types.is_string_dtype(Adult.iloc[:, i]) == 1:
        
        Adult.iloc[:, i] = Adult.iloc[:, i].str.lstrip()
#####################################################



# Split the data into train and test samples where the expert label is 'income'
# 20% of the dataset is used for testing
# This first split is used for only filling categorical missing values

X_train, X_test = train_test_split(Adult, test_size=0.2, random_state = 42, stratify = Adult['income'])



# Replace missing categorical data with most common, non-missing value from the training set.
# The method below works as long as the missing value is not the most comon value

# Training Data
X = X_train["workclass"].value_counts()
Adult.loc[Adult.loc[:, "workclass"] == "?", "workclass"] = X.index[0]
          
X = X_train["occupation"].value_counts()
Adult.loc[Adult.loc[:, "occupation"] == "?", "occupation"] = X.index[0]

X = X_train["native_country"].value_counts()
Adult.loc[Adult.loc[:, "native_country"] == "?", "native_country"] = X.index[0]



# Create new categorical variable from capital_gain and capital_loss variable
# Since the census topcodes individual monetary observations, using these features
# as a continuous variable does not make sense, therfore, they are combined into
# a single categorical feature

Adult.loc[:, "capital_temp"] = Adult.loc[:, "capital_gain"] - Adult.loc[:, "capital_loss"]

Adult.loc[Adult.loc[:, "capital_temp"] == 0, "capital_inc"] = "None"

Adult.loc[(Adult.loc[:, "capital_temp"] >= 1) & (Adult.loc[:, "capital_temp"] <= 50000), "capital_inc"] = "Small_gain"
Adult.loc[(Adult.loc[:, "capital_temp"] > 50000) & (Adult.loc[:, "capital_temp"] <= 99998), "capital_inc"] = "Moderate_gain"
Adult.loc[Adult.loc[:, "capital_temp"] == 99999, "capital_inc"] = "High_gain"

Adult.loc[(Adult.loc[:, "capital_temp"] <= -1) & (Adult.loc[:, "capital_temp"] >= -50000), "capital_inc"] = "Small_loss"
Adult.loc[(Adult.loc[:, "capital_temp"] < -50000) & (Adult.loc[:, "capital_temp"] >= -99998), "capital_inc"] = "Moderate_loss"
Adult.loc[Adult.loc[:, "capital_temp"] == -99999, "capital_inc"] = "High_loss"

Adult = Adult.drop("capital_temp", axis=1)
Adult = Adult.drop("capital_gain", axis=1)
Adult = Adult.drop("capital_loss", axis=1)


#####################################################

# The following code consolodates catigorical variables

######################################################

# Consolodate workclass into 4 categories
Adult.loc[Adult.loc[:, "workclass"] == "Local-gov", "workclass"] = "gov"
Adult.loc[Adult.loc[:, "workclass"] == "State-gov", "workclass"] = "gov"
Adult.loc[Adult.loc[:, "workclass"] == "Federal-gov", "workclass"] = "gov"
Adult.loc[Adult.loc[:, "workclass"] == "Self-emp-not-inc", "workclass"] = "self_emp"
Adult.loc[Adult.loc[:, "workclass"] == "Self-emp-inc", "workclass"] = "self_emp"
Adult.loc[Adult.loc[:, "workclass"] == "Without-pay", "workclass"] = "not_emp"
Adult.loc[Adult.loc[:, "workclass"] == "Never-worked", "workclass"] = "not_emp"

#####################################################

# Consolodate marital_status

Adult.loc[Adult.loc[:, "marital_status"] == "Married-civ-spouse", "marital_status"] = "Married"
Adult.loc[Adult.loc[:, "marital_status"] == "Never-married", "marital_status"] = "Never_married"
Adult.loc[Adult.loc[:, "marital_status"] == "Divorced", "marital_status"] = "Separated"
Adult.loc[Adult.loc[:, "marital_status"] == "Married-spouse-absent", "marital_status"] = "Separated"
Adult.loc[Adult.loc[:, "marital_status"] == "Married-AF-spouse", "marital_status"] = "Married_AF_spouse"


#####################################################
# Consolodate native_country by continent, except North America, Canada and Mexico remain their own categories
# Also consider dropping "South" because I don't know what it is. South Africa? Has it's own category for now.

# U.S and U.S. terratories
Adult.loc[Adult.loc[:, "native_country"] == "United-States", "native_country"] = "US"
Adult.loc[Adult.loc[:, "native_country"] == "Puerto-Rico", "native_country"] = "US"
Adult.loc[Adult.loc[:, "native_country"] == "Outlying-US(Guam-USVI-etc)", "native_country"] = "US"

# Central America and the Carabian
Adult.loc[Adult.loc[:, "native_country"] == "Cuba", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Jamaica", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Honduras", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Haiti", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Dominican-Republic", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "El-Salvador", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Guatemala", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Trinadad&Tobago", "native_country"] = "Central_and_Carib"
Adult.loc[Adult.loc[:, "native_country"] == "Nicaragua", "native_country"] = "Central_and_Carib"

# South America
Adult.loc[Adult.loc[:, "native_country"] == "Peru", "native_country"] = "South_America"
Adult.loc[Adult.loc[:, "native_country"] == "Ecuador", "native_country"] = "South_America"
Adult.loc[Adult.loc[:, "native_country"] == "Columbia", "native_country"] = "South_America"

# Europe
Adult.loc[Adult.loc[:, "native_country"] == "Holand-Netherlands", "native_country"] = "Europe"          
Adult.loc[Adult.loc[:, "native_country"] == "Hungary", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Ireland", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Greece", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Scotland", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Yugoslavia", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "France", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Portugal", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Poland", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Italy", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "Germany", "native_country"] = "Europe"  
Adult.loc[Adult.loc[:, "native_country"] == "England", "native_country"] = "Europe"  

# Aisa
Adult.loc[Adult.loc[:, "native_country"] == "India", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Philippines", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Cambodia", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Thailand", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Laos", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Taiwan", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "China", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Japan", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Vietnam", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Hong", "native_country"] = "Asia"
Adult.loc[Adult.loc[:, "native_country"] == "Iran", "native_country"] = "Asia" 

# No changes in relationship, occupation or race, but get rid of hyphens
# because I hate them.

Adult.loc[:, "relationship"] = Adult.loc[:, "relationship"].str.replace("-","_")
Adult.loc[:, "occupation"] = Adult.loc[:, "occupation"].str.replace("-","_")
Adult.loc[:, "race"] = Adult.loc[:, "race"].str.replace("-","_")

###########################################################
# education and education number are redundant, so I drop the categorical
# education feature and instead go with the education number.  Although not perfect
# it approximates years of education.

Adult = Adult.drop("education", axis=1)

# One hot encode Categorical variables
# The following are created using pandas get_dummies function

Adult = pd.get_dummies(Adult, drop_first = True)



#####################################################
########### END DATA CLEANING #######################
#####################################################


# Split the data into train and test samples where y_train and y_test are the
# expert label 'income_>50K'
# 20% of the dataset is used for testing
# identical to the split above, but now will contain modified data

X_train, X_test = train_test_split(Adult, test_size=0.2, random_state = 42, stratify = Adult['income_>50K'])




# Replace outliers

DC = DC()


X_train.loc[:, "age"] = DC.replace_outlier_train(X_train.loc[:, "age"])
X_test.loc[:, "age"] = DC.replace_outlier_test(X_test.loc[:, "age"])

X_train.loc[:, "fnlwgt"] = DC.replace_outlier_train(X_train.loc[:, "fnlwgt"])
X_test.loc[:, "fnlwgt"] = DC.replace_outlier_test(X_test.loc[:, "fnlwgt"])


X_train.loc[:, "hours_per_week"] = DC.replace_outlier_train(X_train.loc[:, "hours_per_week"])
X_test.loc[:, "hours_per_week"] = DC.replace_outlier_test(X_test.loc[:, "hours_per_week"])

# Output training and testing sets for use in modeling.

cwd = os.getcwd()
path = cwd + '/Adult_Cleaned_Train.csv'
X_train.to_csv(path)

path = cwd + '/Adult_Cleaned_Test.csv'
X_test.to_csv(path)



