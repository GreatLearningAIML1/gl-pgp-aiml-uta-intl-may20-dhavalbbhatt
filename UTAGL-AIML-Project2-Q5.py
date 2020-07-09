#################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix(actual, predicted)
    print(cm)
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
num = data['Experience']._get_numeric_data()
data['Experience'].replace(num[num < 0], 0, inplace=True)

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

# ###########################################################################
#               REMOVING ZIP CODE to see how it improves the model          #
# ###########################################################################
# SUMMARY FINDINGS - Removing ZIP Code significantly improves the model, so will be keeping this model for now.
# Default Parameters will be used for generating the model right now, given that Q5 deals with improving the model
# based on Parameters (will be explored in the next question).

x = data.drop(['Personal Loan', 'Family', 'Mortgage', 'ZIP Code'], axis=1)
y = data['Personal Loan']

# ################### Creating Test and Train data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# ##############################################################################################

# ##################### Identifying parameters that may change (improve) the model. This analyses will help in
# # developing the "final" model"
# ##### SUMMARY OF FINDINGS:
# 1) Parameters solver = liblinear, a C value of 1 (or 0.75), penalty=l2 and class_weights=balanced influence the
#    AUC score the most, AUC=0.92. This is a high score for AUC, but as the following analyses shows, it may not
#    be the best model.
# 2) Further analyses suggests that class_weights seems to be largest influencer on the regression model. This
#    indicates imbalance in the data - meaning the number of positives and negatives is quite large with respect
#    to each other (which is true for this problem, only 9% of people have accepted the offer for personal loan,
#    whereas 91% did not).
# 3) Using class_weights='balanced' therefore skews the results and does not provide accurate balance between
#    between positives and negatives. This is clearly reflected in recall and precision (as determined by
#    the F1-score). While it does improve the AUC, the AUC does not account for class imbalance.
# 4) The question - which model performs better? - is a subjective question - and one that cannot be answered without
#    truly understanding the business requirements. Having said that, the case study presented here is a,
#    marketing problem - wherein business users want to target as many people as they possibly can, meaning,
#    it is OK if the model has more False Positives (indicating more people will purchase the loan, as compared
#    to fewer). This allows the marketing department to target more people based on different criteria
#    as compared to fewer people as would happen in the case of more False Negatives.
#    As a result, a good way to judge how effective a model will be is done by looking at the
#    Precision score - a higher Recall score and a lower Precision score is a better indicator
#    (or higher F1-score) of a good model.
# 5) Using the logic presented in bullet point number 4, I am taking the approach of not using class_weight='balanced'
#    rather assigning weights based on my understanding of where the marketing team wants to see how many people
#    may end up purchasing the loan. This results in an overall better AUC as well as better F1 score.

# Trying different SOLVERS (NB: all solver can be used with l2,
# only 'liblinear' and 'saga' work with both 'l1' and 'l2'
train_score = []
test_score = []
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
for i in solver:
    model = LogisticRegression(random_state=1, penalty='l2', C=0.75, solver=i, max_iter=5000)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    train_score.append(round(model.score(x_train, y_train), 3))
    test_score.append(round(model.score(x_test, y_test), 3))

print("Model scores for models based on different solvers but other params as defaults")
print(solver)
print("*" * 50)
print(train_score)
print("*" * 50)
print(test_score)
print("*" * 50)
#
# Trying with different penalty (l1) values (only works with "liblinear" and "saga" solvers)
train_score = []
test_score = []
solver = ['liblinear', 'saga']
for i in solver:
    model = LogisticRegression(random_state=1, penalty='l1', C=0.75, solver=i, max_iter=5000, class_weight='balanced')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    train_score.append(round(model.score(x_train, y_train), 3))
    test_score.append(round(model.score(x_test, y_test), 3))
print("Model scores for models based on different PENALTY values but default other params")
print(solver)
print("*" * 50)
print(train_score)
print("*" * 50)
print(test_score)
print("*" * 50)
#
# Trying with different values of C and class_weights='balanced' (with solver=liblinear and penalty=l2)
train_score = []
test_score = []
C = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
for i in C:
    model = LogisticRegression(random_state=1, penalty='l2', solver='liblinear', C=i, class_weight='balanced')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    train_score.append(round(model.score(x_train, y_train), 3))
    test_score.append(round(model.score(x_test, y_test), 3))

print("Model scores for models based on different PENALTY values but default other params")
print(C)
print("*" * 50)
print(train_score)
print("*" * 50)
print(test_score)
print("*" * 50)

# ############################################################################################################
# ############# FINAL MODEL DEVELOPMENT BASED ON PARAMETER CHANGES AND RESULTS OF VARIOUS SCORES
w = {0: 15, 1: 85}
model = LogisticRegression(random_state=1, max_iter=5000, solver='newton-cg', C=0.75, penalty='l2',
                           class_weight=w)
model.fit(x_train, y_train)

# Prediction metric
y_predict = (model.predict(x_test))
print("Default Params used for initial model ", model.get_params())
print("*" * 50)

# Model Scores
print("Training accuracy for Improved Model: ", model.score(x_train, y_train))
print("*" * 50)
print("Testing accuracy for Improved Model: ", model.score(x_test, y_test))
print("*" * 50)
print("Accuracy for Improved Model: ", model.score(x_test, y_test))
print("*" * 50)
print("Recall for Improved Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 50)
print("Precision for Improved Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 50)
print("F1 Score for Improved Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 50)
print("ROC AUC Score for Improved Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 50)
print("Brier Score Loss for Improved Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 50)
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
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()