#################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix(actual, predicted)
    print("Confusion Matrix\n", cm)
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()

#################### READ DATASET
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

# ################# GETTING THE DATA MODEL READY
# DROPPING ID given that ID is a unique identifier and ZIP Code based on ASSUMPTION in the FINDINGS section
data = data.drop(['ID'], axis=1)

# # Dropping Row where ZIP Code = '9307'
data = data[data['ZIP Code'] != 9307]

# Replace Negative Values for EXPERIENCE ################
# num = data['Experience']._get_numeric_data()
data['Experience'].replace({-1: 1, -2: 2, -3: 3}, inplace=True)

# #################### ONE HOT ENCODING FOR EDUCATION
data['Education'] = data['Education'].replace({1: 'Undergrad', 2: 'Grad', 3: 'AP'})
data = pd.get_dummies(data, columns=['Education'], drop_first=True)

# ################### BINNING and ONE HOT ENCODING
# Binning FAMILY into 3 Categories - 1=Single, 2=Couple, 3+ = WithKids
# And One Hot Encoding for the FAMILY column
data['Family_Status'] = pd.cut(x=data['Family'], bins=[0, 1, 2, 4], labels=['Single', 'Couple', 'WithKids'])
data = pd.get_dummies(data=data, columns=['Family_Status'], drop_first=True)

# Binning Mortgage into 2 categories - No Mortgage and With Mortgage as well as performing One Hot Encoding on it.
data['Mortgage_Status'] = pd.cut(x=data['Mortgage'], bins=[-1, 0, 1000], labels=['NoMort', 'WithMort'])
data = pd.get_dummies(data=data, columns=['Mortgage_Status'], drop_first=True)

# ################### CREATING x and y data sets to identify dependent variable and independent variable
# For now, all variables, except PERSONAL LOAN which is the independent variable, are included.
# FAMILY and MORTGAGE are reflected through the One Hot Encoded column FAMILY STATUS and hence dropped
# from the 'x' DataFrame
# ASSUMPTION for WHICH VALUE IS CONSIDERED POSITIVE and WHICH IS NEGATIVE
# It is assumed that a value of "1" in Personal Loan indicates that the person applied and got a loan. A value of 0
# indicates that the person did not take the personal loan that may have been offered to him/her.


# ###########################################################################
#               REMOVING ZIP CODE improves the model                        #
# ###########################################################################
# SUMMARY FINDINGS - Removing ZIP Code significantly improves the model.
# Default Parameters will be used for generating the model right now, given that Q5 deals with improving the model
# based on Parameters (will be explored in the next question).

x = data.drop(['Personal Loan', 'Family', 'Mortgage', 'ZIP Code'], axis=1)
y = data['Personal Loan']

# ################### Creating Test and Train data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# #### Validate if the split has appropriate number of personal loan customers before and after split
print("Original Loan True Values: {0}, ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 1]), (
        len(data.loc[data['Personal Loan'] == 1]) / len(data.index)) * 100))
print("Original Loan False Values: {0}, ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 0]), (
        len(data.loc[data['Personal Loan'] == 0]) / len(data.index)) * 100))
print("*" * 50)
print("Training Loan True Values: {0}, ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (
        len(y_train[y_train[:] == 1]) / len(y_train)) * 100))
print("Training Loan False Values: {0}, ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (
        len(y_train[y_train[:] == 0]) / len(y_train)) * 100))
print("*" * 50)
print("Test Loan True Values: {0}, ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (
        len(y_test[y_test[:] == 1]) / len(y_test)) * 100))
print("Test Loan False Values: {0}, ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (
        len(y_test[y_test[:] == 0]) / len(y_test)) * 100))
print("*" * 50)
# FINDINGS - It looks like the data is well matched with respect to who accepted the Personal Loan across the train
# and test data segments.
# ##############################################################################################
# ################### Building LOGIT Function
logit = sm.Logit(y_train, sm.add_constant(x_train))
lg = logit.fit()
#
# ############ LOGIT SUMMARY STATS
stats.chisqprob = lambda chisq, data: stats.chi2.sf(chisq, data)
print(lg.summary())
print("*" * 50)

# Calculate Odds Ratio, probability
# Create a data frame to collate Odds ratio, probability and p-value of the coef
lgcoef = pd.DataFrame(lg.params, columns=['coef'])
lgcoef.loc[:, "Odds_ratio"] = np.exp(lgcoef.coef)
lgcoef['probability'] = lgcoef['Odds_ratio']/(1+lgcoef['Odds_ratio'])
lgcoef['pval'] = lg.pvalues
pd.options.display.float_format = '{:.2f}'.format
print(lgcoef)
print("*" * 50)
# Figure out which Columns (attributes) are important factors used to accept Personal Loans
lgcoef = lgcoef.sort_values(by="Odds_ratio", ascending=False)
pval_filter = lgcoef['pval'] <= 0.1
print(lgcoef[pval_filter])
print("*" * 50)

# ############# CREATE REGRESSION MODEL (Most Parameters are DEFAULT FOR NOW, Q5 has a better model)
model = LogisticRegression(random_state=1, max_iter=5000, solver='liblinear')
model.fit(x_train, y_train)

# Prediction metric
y_predict = (model.predict(x_test))

# Model Scores
print("Training accuracy: ", model.score(x_train, y_train))
print("*" * 50)
print("Testing accuracy: ", model.score(x_test, y_test))
print("*" * 50)
print("Accuracy: ", model.score(x_test, y_test))
print("*" * 50)
print("Recall: ", metrics.recall_score(y_test, y_predict))
print("*" * 50)
print("Precision: ", metrics.precision_score(y_test, y_predict))
print("*" * 50)
print("F1 Score: ", metrics.f1_score(y_test, y_predict))
print("*" * 50)
print("ROC AUC Score: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 50)
print("Brier Score Loss for Improved Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 50)

# Confusion Matrix
draw_cm(y_test, y_predict)

# ROC and AUC Graph
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print("ROC AUC Score for Improved Model: ", metrics.auc(fpr, tpr))
print("*" * 50)