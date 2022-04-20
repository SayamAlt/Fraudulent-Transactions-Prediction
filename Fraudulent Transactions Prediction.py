#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import imblearn
from imblearn.under_sampling import NearMiss


# In[2]:


plt.rcParams['figure.figsize'] = (12,8)


# In[3]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[4]:


df = pd.read_csv('Fraud.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df[df.duplicated()]


# Data is already clean as far as null and duplicate values are concerned so no data cleaning process is required over here.

# In[10]:


df.corr()


# In[11]:


df.drop(columns=['nameOrig','nameDest'],inplace=True)


# ### Eliminating outliers

# In[12]:


sns.boxenplot(df['amount'])
plt.ylabel('distribution')
plt.show()


# In[13]:


sns.boxenplot(df['oldbalanceOrg'])
plt.ylabel('distribution')


# In[14]:


sns.boxenplot(df['newbalanceOrig'])
plt.ylabel('distribution')


# In[15]:


sns.boxenplot(df['oldbalanceDest'])
plt.ylabel('distribution')


# In[16]:


sns.boxenplot(df['newbalanceDest'])
plt.ylabel('distribution')


# In[17]:


def remove_outliers(df,col):
    lower_quantile = df[col].quantile(0.25)
    upper_quantile = df[col].quantile(0.75)
    IQR = upper_quantile - lower_quantile
    lower_whisker = lower_quantile - 1.5 * IQR
    upper_whisker = upper_quantile + 1.5 * IQR
    temp = df.loc[(df[col]>lower_whisker)&(df[col]<upper_whisker)]
    return temp[col]


# In[18]:


df['amount'] = remove_outliers(df,'amount')
df['oldbalanceOrg'] = remove_outliers(df,'oldbalanceOrg')
df['newbalanceOrig'] = remove_outliers(df,'newbalanceOrig')
df['oldbalanceDest'] = remove_outliers(df,'oldbalanceDest')
df['newbalanceDest'] = remove_outliers(df,'newbalanceDest')


# ## Exploratory Data Analysis

# In[19]:


sns.heatmap(df.corr(),annot=True,cmap='plasma')


# The only noteworthy thing from the heatmap is that the transaction recipient's old and new balances are strongly positively correlated with each other.

# In[20]:


df.groupby('isFraud').describe().T


# In[21]:


values = df['type'].value_counts().values
labels = df['type'].value_counts().keys()
explode = (0.1,0,0,0,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()


# A vast majority of the transactions are of the type CASH_OUT having an overall proportion of a little more than one-third, closely followed by Payment mode having a share of nearly 34%. The proportion of CASH_IN transaction stood at just over one-fifth, even though, the percentage of debit and normal transfer transactions had a minimal share of less than one-tenth.

# In[22]:


values = df['isFraud'].value_counts().values
labels = ['Not Fraud','Fraud']
explode = (0.1,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()


# Just a tiny fraction of the total transactions in the given dataset are fraud which indicates that this is a highly imbalanced dataset.

# Let's find out the maximum transferred amount by type.

# In[23]:


max_amount_type = df.groupby('type')['amount'].max().sort_values(ascending=False).reset_index()[:10]
max_amount_type


# In[24]:


sns.barplot(x='type',y='amount',data=max_amount_type,palette='magma')


# The highest amount was transferred through normal transfer mode while the least money was transferred by payment.

# In[25]:


sns.countplot(df['isFraud'])


# This is an imbalanced dataset as almost all the samples provided in the dataset belong to the majority class label 'Not Fraud'.

# In[26]:


sns.distplot(df['amount'],bins=50)


# All the balances, either old or new, of both the sender as well as the receiver have a positively right skewed distribution so let's perform normalization of each of these variables.

# In[27]:


positive_fraud_case = df[df['isFraud']==1]
sns.distplot(positive_fraud_case['amount'],bins=50)


# In[28]:


non_fraud_case = df[df['isFraud']==0]
sns.distplot(non_fraud_case['amount'],bins=50)


# In[29]:


sns.regplot(x='oldbalanceDest',y='newbalanceDest',data=df.sample(100000))


# ## Performing Min Max Normalization of the features

# In[30]:


df['amount'].fillna(df['amount'].mean(),inplace=True)
df['oldbalanceOrg'].fillna(df['oldbalanceOrg'].mean(),inplace=True)
df['newbalanceOrig'].fillna(df['newbalanceOrig'].mean(),inplace=True)
df['oldbalanceDest'].fillna(df['oldbalanceDest'].mean(),inplace=True)
df['newbalanceDest'].fillna(df['newbalanceDest'].mean(),inplace=True)


# In[31]:


payment_types = pd.get_dummies(df['type'],prefix='type',drop_first=True)
df = pd.concat([df,payment_types],axis=1)
df.head()


# In[32]:


df.drop('type',axis=1,inplace=True)


# In[33]:


df['type_CASH_OUT'] = df['type_CASH_OUT'].astype(np.int64)
df['type_DEBIT'] = df['type_DEBIT'].astype(np.int64)
df['type_PAYMENT'] = df['type_PAYMENT'].astype(np.int64)
df['type_TRANSFER'] = df['type_TRANSFER'].astype(np.int64)


# Warning: The target variable of our machine learning models is predominantly imbalanced which may hamper the predictive accuracy of the models as the predictions may be solely made on the basis of the 'majority class', thereby completely neglecting the 'minority class' as a consequence.

# In[34]:


x = df.drop('isFraud',axis=1)
y = df['isFraud']


# In[35]:


nm = NearMiss()
x_nm, y_nm = nm.fit_resample(x,y)


# # Model Training and Evaluation

# In[36]:


X = x_nm
y = y_nm
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,stratify=y,random_state=2022)


# In[37]:


X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)


# In[38]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
print('ROC AUC Score:',roc_auc_score(y_test,lr_pred))
print('F1 Score:',f1_score(y_test,lr_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,lr_pred))
print('Classification Report:\n',classification_report(y_test,lr_pred))
print('Accuracy Score:',accuracy_score(y_test,lr_pred))


# In[39]:


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


# In[40]:


rfc_pred = rfc.predict(X_test)
rfc_pred


# In[41]:


print("Confusion Matrix:\n",confusion_matrix(y_test,rfc_pred))
print("Classification Report:\n",classification_report(y_test,rfc_pred))
print("ROC AUC Score:",roc_auc_score(y_test,rfc_pred))
print("F1 Score:",f1_score(y_test,rfc_pred))
print('Accuracy Score:',accuracy_score(y_test,rfc_pred))


# In[42]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[43]:


dtree_pred = dtree.predict(X_test)
dtree_pred


# In[44]:


print("ROC AUC Score:",roc_auc_score(y_test,dtree_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,dtree_pred))
print("F1 Score:",f1_score(y_test,dtree_pred))
print("Classification Report:\n",classification_report(y_test,dtree_pred))
print("Accuracy Score:",accuracy_score(y_test,dtree_pred))


# In[45]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[46]:


gnb_pred = gnb.predict(X_test)
gnb_pred


# In[47]:


print("ROC AUC Score:",roc_auc_score(y_test,gnb_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,gnb_pred))
print("F1 Score:",f1_score(y_test,gnb_pred))
print("Classification Report:\n",classification_report(y_test,gnb_pred))
print("Accuracy Score:",accuracy_score(y_test,gnb_pred))


# In[48]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)


# In[49]:


knn_pred = knn.predict(X_test)
knn_pred


# In[50]:


print("ROC AUC Score:",roc_auc_score(y_test,knn_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,knn_pred))
print("F1 Score:",f1_score(y_test,knn_pred))
print("Classification Report:\n",classification_report(y_test,knn_pred))
print("Accuracy Score:",accuracy_score(y_test,knn_pred))


# In[51]:


svm = SVC()
svm.fit(X_train,y_train)


# In[52]:


svm_pred = svm.predict(X_test)
svm_pred


# In[53]:


print("ROC AUC Score:",roc_auc_score(y_test,svm_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,svm_pred))
print("F1 Score:",f1_score(y_test,svm_pred))
print("Classification Report:\n",classification_report(y_test,svm_pred))
print("Accuracy Score:",accuracy_score(y_test,svm_pred))


# In[54]:


xgb = XGBClassifier()
xgb.fit(X_train,y_train)


# In[55]:


xgb_pred = xgb.predict(X_test)
xgb_pred


# In[56]:


print("ROC AUC Score:",roc_auc_score(y_test,xgb_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,xgb_pred))
print("F1 Score:",f1_score(y_test,xgb_pred))
print("Classification Report:\n",classification_report(y_test,xgb_pred))
print("Accuracy Score:",accuracy_score(y_test,xgb_pred))


# ## Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV

# In[57]:


param_grid = {'C': [1,10,100,1000,10000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}


# In[58]:


grid_search_svm = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid_search_svm.fit(X_train,y_train)


# In[59]:


grid_search_svm.best_estimator_


# In[60]:


svm = SVC(C=10000,gamma=1)
svm.fit(X_train,y_train)


# In[61]:


svm_pred = svm.predict(X_test)
svm_pred


# In[62]:


print("ROC AUC Score:",roc_auc_score(y_test,svm_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,svm_pred))
print("F1 Score:",f1_score(y_test,svm_pred))
print("Classification Report:\n",classification_report(y_test,svm_pred))
print("Accuracy Score:",accuracy_score(y_test,svm_pred))


# In[63]:


param_grid = {'n_estimators': [100,200,300,400,500],
              'criterion': ['gini','entropy'],
              'class_weight': ['balanced','balanced_subsample']
             }


# In[64]:


random_search_rfc = RandomizedSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=3)
random_search_rfc.fit(X_train,y_train)


# In[65]:


rfc = RandomForestClassifier(n_estimators=300,criterion='entropy',class_weight='balanced')
rfc.fit(X_train,y_train)


# In[66]:


rfc_pred = rfc.predict(X_test)
rfc_pred


# In[67]:


print("ROC AUC Score:",roc_auc_score(y_test,rfc_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,rfc_pred))
print("F1 Score:",f1_score(y_test,rfc_pred))
print("Classification Report:\n",classification_report(y_test,rfc_pred))
print("Accuracy Score:",accuracy_score(y_test,rfc_pred))


# In[68]:


param_grid = {'C': [1.0,2.0,3.0,4.0,5.0], 
              'solver': ['liblinear','sag','saga'],
              'class_weight': ['balanced']}


# In[69]:


random_search_lr = RandomizedSearchCV(LogisticRegression(),param_grid,refit=True,verbose=3)
random_search_lr.fit(X_train,y_train)


# In[70]:


lr = LogisticRegression(C=5.0,solver='saga',class_weight='balanced')
lr.fit(X_train,y_train)


# In[71]:


lr_pred = lr.predict(X_test)
lr_pred


# In[72]:


print("ROC AUC Score:",roc_auc_score(y_test,lr_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,lr_pred))
print("F1 Score:",f1_score(y_test,lr_pred))
print("Classification Report:\n",classification_report(y_test,lr_pred))
print("Accuracy Score:",accuracy_score(y_test,lr_pred))


# In[73]:


param_grid = {'criterion': ['gini','entropy'], 'splitter': ['best','random']}


# In[74]:


random_search_dtree = RandomizedSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=3)
random_search_dtree.fit(X_train,y_train)


# In[75]:


dtree = DecisionTreeClassifier(criterion='gini',splitter='random')
dtree.fit(X_train,y_train)


# In[76]:


dtree_pred = dtree.predict(X_test)
dtree_pred


# In[77]:


print("ROC AUC Score:",roc_auc_score(y_test,dtree_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,dtree_pred))
print("F1 Score:",f1_score(y_test,dtree_pred))
print("Classification Report:\n",classification_report(y_test,dtree_pred))
print("Accuracy Score:",accuracy_score(y_test,dtree_pred))


# In[78]:


param_grid = {'n_neighbors': [1,2,3,4,5],
             'weights': ['uniform','distance'],
             'algorithm': ['auto','ball_tree','kd_tree','brute'],
             'p': [1,2]}


# In[79]:


random_search_knn = RandomizedSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=3)
random_search_knn.fit(X_train,y_train)


# In[80]:


knn = KNeighborsClassifier(n_neighbors=2,algorithm='auto',p=2,weights='distance')
knn.fit(X_train,y_train)


# In[86]:


knn_pred = knn.predict(X_test)
knn_pred


# In[87]:


print("ROC AUC Score:",roc_auc_score(y_test,knn_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,knn_pred))
print("F1 Score:",f1_score(y_test,knn_pred))
print("Classification Report:\n",classification_report(y_test,knn_pred))
print("Accuracy Score:",accuracy_score(y_test,knn_pred))


# ## Conclusion

# In[83]:


print("Performance of ML Models:")
print('Predictive Accuracy of Logistic Regression:',str(np.round(accuracy_score(y_test,lr_pred)*100,2)) + '%')
print('Predictive Accuracy of K Neighbors Classifier:',str(np.round(accuracy_score(y_test,knn_pred)*100,2)) + '%')
print('Predictive Accuracy of Support Vector Classifier:',str(np.round(accuracy_score(y_test,svm_pred)*100,2)) + '%')
print('Predictive Accuracy of Decision Tree Classifier:',str(np.round(accuracy_score(y_test,dtree_pred)*100,2)) + '%')
print('Predictive Accuracy of Random Forest Classifier:',str(np.round(accuracy_score(y_test,rfc_pred)*100,2)) + '%')
print('Predictive Accuracy of Gaussian Naive Bayes:',str(np.round(accuracy_score(y_test,gnb_pred)*100,2)) + '%')
print('Predictive Accuracy of XGBoost Classifier:',str(np.round(accuracy_score(y_test,xgb_pred)*100,2)) + '%')


# ### K Nearest Neighbors Classifier is the best performing model with a prediction accuracy of approximately 99%.

# ### Gaussian Naive Bayes is the worst performing model having the least prediction accuracy of just 85.81%.
