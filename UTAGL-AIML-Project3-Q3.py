# ################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from yellowbrick.classifier import ClassificationReport, ROCAUC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import tree
import PIL
from PIL import Image
from os import system
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from IPython.core.display import display
from scipy import stats
import statsmodels.api as sm
import graphviz
import pydotplus
import IPython
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

# ########### SET DISPLAY OPTIONS ##############################
pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)
# ################## FUNCTIONS #################################
# # Correlation between any two features <- Will be used for Education and Jobs
def mat(df, feature1, feature2):
    f1Values = list(df[feature1].unique())
    f2Values = list(df[feature2].unique())
    gdf = []
    for i in f2Values:
        dff2 = df[df[feature2] == i]
        dff1 = dff2.groupby(feature1).count()[feature2]
        gdf.append(dff1)
    outputdf = pd.concat(gdf, axis=1)
    outputdf.columns = f2Values
    outputdf=outputdf.fillna(0)
    return outputdf

## function to get confusion matrix in a proper format
def draw_cm(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted)
    print(cm)
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()

# # Function to identify important FEATURES
def feat_importance(model, features):
    # v_feat_importance = model.tree_.compute_feature_importances(normalize=False)
    feat_imp_dict = dict(zip(features, model.feature_importances_))
    feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
    return print(feat_imp.sort_values(by=0, ascending=False))

# ##############################################################
# ########### READ the FILE ####################################
bankdata = pd.read_csv('bank.csv')

# ######### Data Imputation for CELL PHONE and TELEPHONE for contact feature.
#   Based on Hypothesis 2, there is no real difference between contacting using a telephone or a cell phone.
#   As a result I will impute both TELEPHONE and CELLULAR values with PHONE in contact.
bankdata['contact'] = bankdata['contact'].replace({"cellular": "phone", "telephone": "phone"})
# print(bankdata['contact'].value_counts())
# print('*' * 100)

# ######### Data Binning for pdays based on Hypothesis 4 ########
#   After binning, drop column pdays. Further, I will perform one-hot encoding on all features that are categorical.
bankdata['pdaysbin'] = pd.cut(x=bankdata['pdays'], bins=[-2, 0, 10, 31, 90, 180, 1000],
                              labels=("unknown", "1-10 days", "11-31 days", "32-90 days", "90-180 days", "180+ days"))
bankdata = bankdata.drop(['pdays'], axis=1)
# print(bankdata.groupby(['pdaysbin']).count())
# print(bankdata.info())
# print('*' * 100)


# ########## Data Imputation for UNKNOWN values in job ##########
#   This imputation will be done based on findings related to age of a person as well as findings based on education
#   and job.
bankdata.loc[(bankdata['age'] >= 60) & (bankdata['job'] == 'unknown'), 'job'] = 'retired'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'tertiary'), 'job'] = 'management'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'secondary'), 'job'] = 'technician'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'primary'), 'job'] = 'blue-collar'

# ########## Data Imputation for UNKNOWN values in education #####
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'admin.'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'blue-collar'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'entrepreneur'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'housemaid'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'management'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'retired'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'self-employed'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'services'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'student'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'technician'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'unemployed'), 'education'] = 'secondary'


# Change education, default, housing, loan, month and Target from categorical to continuous feature
RepStruct = {
    "education": {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": -1},
    "default": {"no": 0, "yes": 1},
    "housing": {"no": 0, "yes": 1},
    "loan": {"no": 0, "yes": 1},
    "month": {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
              "nov": 11, "dec": 12},
    "pdaysbin": {"180+ days": 1, "90-180 days": 2, "32-90 days": 3, "11-31 days": 4, "1-10 days": 5, "unknown": -1},
    "Target": {"no": 0, "yes": 1}
}
bankdata = bankdata.replace(RepStruct)

# ########### Changing DTYPE object to category ################
for feature in bankdata.columns:
    if bankdata[feature].dtype == 'object':
        bankdata[feature] = pd.Categorical(bankdata[feature])

# ########### IMPUTE DUMMY VALUES for CATEGORICAL DATA #########
cat_features = []
for i in bankdata.columns:
    if bankdata[i].dtype != 'int64':
        cat_features.append(i)

for i in cat_features:
    bankdata = pd.get_dummies(data=bankdata, columns=[i], drop_first=True)

# ####### The data is now ready for modeling. The following section will be used to create test/train datasets.
# ##############################################################################################################

# ########### CREATE TEST and TRAIN DATA ##############
X = bankdata.drop(['Target'], axis=1)
y = bankdata['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)

# ############################### MODEL DEVELOPMENT ##################################################
#
# ############################ LOGISTIC REGRESSION ###################
#   I will use the default params first and then try to tune the model
model = LogisticRegression(random_state=1, max_iter=5000, C=1, solver='liblinear', class_weight={0: 0.14, 1: 0.86},
                           penalty='l2')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Model Scores
print("Training accuracy for LogReg Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for LogReg Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for LogReg Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for LogReg Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for LogReg Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for LogReg Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for LogReg Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for LogReg Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# ############# ACCURACY SCORE FOR COMPARISON WITH OTHER MODELS
# Initialize the dataframe
comp_DF = pd.DataFrame({'Method': [], 'Accuracy': 0, 'Recall': 0, 'Precision': 0})
# Calculate various scores
acc_DT_log = metrics.accuracy_score(y_test, y_predict)
rec_log = metrics.recall_score(y_test, y_predict)
prec_log = metrics.precision_score(y_test, y_predict)
# Create DF for scores
comp_DF_log = pd.DataFrame(
    {'Method': ['Logistic Regression'], 'Accuracy': acc_DT_log, 'Recall': rec_log, 'Precision': prec_log})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
# Append DF to original DF
comp_DF = comp_DF.append(comp_DF_log)
print("Scores for Log Reg Model:\n{}".format(comp_DF))
print('*' * 100)

# ROC and AUC Graph
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
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

# Yellowbrick visualization
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()


# # Perform Gridsearch using various parameters
penalty = ['l1', 'l2']
C = [0.1, 0.25, 0.5]
solver = ['newton-cg', 'lbfgs', 'liblinear']
class_weight = [{0: 15, 1: 85}, {0: 0.14, 1: 0.86}]
miter = [5000]

hyperparams = dict(C=C, penalty=penalty, solver=solver, class_weight=class_weight, max_iter=miter)
RegLog = LogisticRegression()

finalmodel = GridSearchCV(RegLog, hyperparams, verbose=0, cv=5)
finalmodel.fit(X_train, y_train)
y_predict = finalmodel.predict(X_test)
print("GRID SEARCH MODEL SCORES")
print("*" * 100)
print(finalmodel.get_params())
print("*" * 100)
print("Training accuracy for Improved Model: ", finalmodel.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for Improved Model: ", finalmodel.score(X_test, y_test))
print("*" * 100)
print("Accuracy for Improved Model: ", finalmodel.score(X_test, y_test))
print("*" * 100)
print("Recall for Improved Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for Improved Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for Improved Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for Improved Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for Improved Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# Confusion Matrix
draw_cm(y_test, y_predict)
#
# # ROC and AUC Graph
# logit_roc_auc = roc_auc_score(y_test, finalmodel.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, finalmodel.predict_proba(X_test)[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()
#
# print("ROC AUC Score for Improved Model: ", metrics.auc(fpr, tpr))
# print("*" * 100)
#############################################################################
# ############################ DECISION TREE MODEL #####################
model = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=5, min_samples_leaf=5,
                               class_weight={0: 0.15, 1: 0.85})
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print("Training accuracy for DT Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for DT Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for DT Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for DT Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for DT Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for DT Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for DT Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for DT Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Yellowbrick visualization
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()

# Tree Visualization
train_char_label = ['No', 'Yes']
Tree_File = open('loan_deposit_DT.dot', 'w')
dot_data_dtree = tree.export_graphviz(model, out_file=Tree_File, feature_names=list(X_train),
                                      class_names=list(train_char_label))
Tree_File.close()

retCode = system("dot -Tpng loan_deposit_DT.dot -o loan_deposit_DT.png")
if(retCode > 0):
    print("system command returning error: "+str(retCode))
else:
    img = PIL.Image.open("loan_deposit_DT.png", mode='r')
    img.show()

# #The following identifies features that are important.
feat_importance(model, X.columns)
print('*' * 100)

# #### Adding to comparision matrix
acc_DT_DT = metrics.accuracy_score(y_test, y_predict)
rec_DT = metrics.recall_score(y_test, y_predict)
prec_DT = metrics.precision_score(y_test, y_predict)

comp_DF_DT = pd.DataFrame({'Method': ['Decision Tree'], 'Accuracy': acc_DT_DT, 'Recall': rec_DT, 'Precision': prec_DT})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_DT)
print("Scores for Decision Tree Model:\n{}".format(comp_DF))
print('*' * 100)

# ############################################################################################
# ########### RANDOM FOREST MODELS #####################
model = RandomForestClassifier(n_estimators=128, random_state=1, max_depth=6, min_samples_leaf=5,
                               class_weight={0: 0.13, 1: 0.87}, criterion='gini')
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print("Training accuracy for RF Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for RF Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for RF Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for RF Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for RF Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for RF Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for RF Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for RF Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Visualization using Yellowbricks
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()


# #### Adding to comparision matrix
acc_DT_RF = metrics.accuracy_score(y_test, y_predict)
rec_RF = metrics.recall_score(y_test, y_predict)
prec_RF = metrics.precision_score(y_test, y_predict)

comp_DF_RF = pd.DataFrame({'Method': ['Random Forest'], 'Accuracy': acc_DT_RF, 'Recall': rec_RF, 'Precision': prec_RF})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_RF)
print("Scores for Random Forest Model:\n{}".format(comp_DF))
print('*' * 100)

##########################################################################################
# ######## ADA BOOST CLASSIFIER ############
model = AdaBoostClassifier(random_state=1, n_estimators=128, learning_rate=0.9)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Training accuracy for AB Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for AB Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for AB Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for AB Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for AB Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for AB Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for AB Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for AB Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Visualization using Yellowbricks
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()

# #### Adding to comparision matrix
acc_DT_AB = metrics.accuracy_score(y_test, y_predict)
rec_AB = metrics.recall_score(y_test, y_predict)
prec_AB = metrics.precision_score(y_test, y_predict)

comp_DF_AB = pd.DataFrame({'Method': ['Ada Boost'], 'Accuracy': acc_DT_AB, 'Recall': rec_AB, 'Precision': prec_AB})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_AB)
print("Accuracy Score for comparision:\n{}".format(comp_DF))
print('*' * 100)

############################################################################################
# #################### Bagging Classifier ######################
model = BaggingClassifier(random_state=1, n_estimators=256, max_samples=0.7, bootstrap=True,
                          oob_score=True)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Training accuracy for BC Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for BC Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for BC Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for BC Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for BC Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for BC Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for BC Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for BC Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Visualization using Yellowbricks
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()


# #### Adding to comparision matrix
acc_DT_BC = metrics.accuracy_score(y_test, y_predict)
rec_BC = metrics.recall_score(y_test, y_predict)
prec_BC = metrics.precision_score(y_test, y_predict)

comp_DF_BC = pd.DataFrame(
    {'Method': ['Bagging Classifier'], 'Accuracy': acc_DT_BC, 'Recall': rec_BC, 'Precision': prec_BC})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_BC)
print("Scores for comparision:\n{}".format(comp_DF))
print('*' * 100)
# #######################################################################################
# ############## Gradient Boosting ################
model = GradientBoostingClassifier(random_state=1, n_estimators=128, learning_rate=0.05, max_depth=3,
                                   min_samples_leaf=3)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Training accuracy for GC Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for GC Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for GC Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for GC Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for GC Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for GC Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for GC Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for GC Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Visualization using Yellowbricks
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()

# #### Adding to comparision matrix
acc_DT_GC = metrics.accuracy_score(y_test, y_predict)
rec_GC = metrics.recall_score(y_test, y_predict)
prec_GC = metrics.precision_score(y_test, y_predict)

comp_DF_GC = pd.DataFrame(
    {'Method': ['Gradient Classifier'], 'Accuracy': acc_DT_GC, 'Recall': rec_GC, 'Precision': prec_GC})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_GC)
print("Scores for comparision:\n{}".format(comp_DF))
print('*' * 100)
# #######################################################################################
# ########### XGBOOST model ################
model = XGBClassifier(random_state=1, n_estimators=190, max_depth=6, gamma=0.125, scale_pos_weight=75)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Training accuracy for XG Model: ", model.score(X_train, y_train))
print("*" * 100)
print("Testing accuracy for XG Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Accuracy for XG Model: ", model.score(X_test, y_test))
print("*" * 100)
print("Recall for XG Model: ", metrics.recall_score(y_test, y_predict))
print("*" * 100)
print("Precision for XG Model: ", metrics.precision_score(y_test, y_predict))
print("*" * 100)
print("F1 Score for XG Model: ", metrics.f1_score(y_test, y_predict))
print("*" * 100)
print("ROC AUC Score for XG Model: ", metrics.roc_auc_score(y_test, y_predict))
print("*" * 100)
print("Brier Score Loss for XG Model: ", metrics.brier_score_loss(y_test, y_predict))
print("*" * 100)
print("*" * 100)
# #Confusion Matrix
draw_cm(y_test, y_predict)
print('*' * 100)

# Visualization using Yellowbricks
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()

roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.show()

# #### Adding to comparision matrix
acc_DT_XG = metrics.accuracy_score(y_test, y_predict)
rec_XG = metrics.recall_score(y_test, y_predict)
prec_XG = metrics.precision_score(y_test, y_predict)

comp_DF_XG = pd.DataFrame(
    {'Method': ['XGradient Classifier'], 'Accuracy': acc_DT_XG, 'Recall': rec_XG, 'Precision': prec_XG})
# comp_DF_log = comp_DF_log[['Method', 'Accuracy']]
comp_DF = comp_DF.append(comp_DF_XG)
print("Scores for comparision:\n{}".format(comp_DF))
print('*' * 100)

# ##### Feature Importance for XGBoost Model
feat_importance(model, X.columns)
print('*' * 100)

# ################ FINDINGS and INFERENCES #######################
# EXECUTIVE SUMMARY:
# 1) The problem identified in the project is a classification problem. As a result, I have used various models that
#    allow me to address the classification problem of whether a person will make a term deposit or not. It has also
#    been mentioned that the bank and especially the marketing department within the bank is looking to target
#    individuals that fit a certain profile. The problem has run such marketing campaigns before, with a varied
#    success rate, as can be seen in the 'poutcome' feature of the dataset.
# 2) Another aspect of the problem is, that the outcome is highly imbalanced (89% of people sampled in the dataset
#    are not ready to make term deposits and only 11% willing to make term deposits). Working with highly imbalanced
#    data (as the one presented) in this problem has its own challenges - not all models allow for appropriate
#    weights to be associated with the outcome classes. As a result, while some models present a higher rate of
#    accuracy, they don't necessarily allow the marketing team to 'target' appropriate individuals who may be
#    willing to make term deposits with the bank. This will be taken in to account while prescribing the final
#    model.
# 3) I have made the assumption that the marketing team needs to market to "more" people. This implies that it is
#    better to lower the False Negatives and err on the side of making higher False Positives (while increasing the
#    total True positives and negatives). As a result, "ACCURACY" score of the model is not the critical metric -
#    instead, a better RECALL/PRECISION (and by extension F1 SCORE) are of more importance. In fact, higher
#    importance is being placed on REDUCING FALSE NEGATIVES while keeping the overall accuracy at a higher rate.
# 4) With the above in mind, I am prescribing either the RANDOM FOREST model or XGBOOST model for modeling and
#    making predictions for this particular dataset as well as the business problem at hand. Both are very similar
#    to each other in terms of prediction of FALSE NEGATIVES as well as overall ACCURACY.
# 5) While the DECISION TREE model has the lowest FALSE NEGATIVE rate, its overall accuracy is lacking slightly. It
#    is for these reasons that I am going to discard the use of the DECISION TREE model.
# 6) I am also discounting the use of ADA-BOOSTING, BAGGING CLASSIFIER and GRAIDENT BOOSTING primarily because
#    there is no clear way of using these algorithms for class imbalanced datasets (as is the case with this problem).
# 7) The LOGISTIC REGRESSION model has been discarded primarily because it has not been able to provide
#    the lower levels of FALSE NEGATIVES as either the RANDOM FOREST or XGRADIENT BOOST models provide.
# 8) Between the RANDOM FOREST model and the XGRADIENT BOOST model, I would recommend the use of
#    XGRADIENT BOOST model. The model is robust in its use of dealing with class imbalance (allowing the use of
#    weights based on imbalance in the data). Additionally, like with RANDOM FOREST, the model also allows
#    model regularization with selecting maximum depth as well as minimum leaf samples. This also provides a good way
#    for cross validation of the dataset internally, without having the need to run external functions.
# 9) Lastly, the feature importance measure clearly identifies that having some form of contact with potential
#    customers is the biggest impact on whether a person will agree to make a term deposit or not. This is followed
#    by whether people have previously agreed to make term deposits or not. The month (as was hypothesized previously)
#    of contact also plays a major role in whether a person will make term deposit or not. These align with
#    normal business practices of marketing.
