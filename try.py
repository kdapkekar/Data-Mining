# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:43:55 2020

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:26:40 2020

@author: Dell
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv("D:\SPRING 2020\ICSI531\Project\Canvas_Network\CNPC_1401-1509_DI_v1_1_2016-03-01.csv",nrows = 37080)


to_drop=['course_id_DI','discipline','userid_DI','explored','grade_reqs','course_reqs','final_cc_cname_DI','LoE_DI','age_DI','gender','start_time_DI','course_start','course_end','last_event_DI','ncontent']
dataset.drop(to_drop, inplace=True, axis=1)
dataset=pd.DataFrame(dataset)
dataset['learner_type']=dataset['learner_type'].fillna('Missing')
dataset['primary_reason']=dataset['primary_reason'].fillna(0)
dataset['expected_hours_week']= dataset['expected_hours_week'].replace({'Between 4 and 6 hours':5,'Between 2 and 4 hours':3,'Between 1 and 2 hours':2,'Less than 1 hour':1,'Between 6 and 8 hours':7,'More than 8 hours per week':10,'Missing':0})
dataset['expected_hours_week']= dataset['expected_hours_week'].fillna(0)
dataset['completed_%']= dataset['completed_%'].fillna(0)
dataset['nforum_posts']=dataset['nforum_posts'].fillna(0)
dataset['grade'] = dataset['grade'].fillna(0)
#print dataset.head()
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(0,50),1)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(50,100),2)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100,500),3)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500,1000),4)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(1000,5000),5)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(5000,10000),6)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(10000,100000),7)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100000,500000),8)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500000,1000000),9)
dataset['nevents']=dataset['nevents'].fillna(0)
dataset['ndays_act']=dataset['ndays_act'].fillna(0)
#print np.unique(dataset['ndays_act'])
print(dataset.head())

#writer = pd.ExcelWriter('D:\SPRING 2020\ICSI531\Project\Canvas_Network\output.xlsx')
#dataset.to_excel(writer)
#writer.save()
#print('DataFrame is written successfully to Excel File.')

X = dataset
y = dataset.iloc[:,2].values
print(X.head())

y = pd.DataFrame(y)
y = y.fillna(0)
#print(y)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#X['learner_type'] = le.fit_transform(X['learner_type'])
X['primary_reason'] = le.fit_transform(X['primary_reason'].astype('str'))

#Splitting into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,shuffle=False)
dataset_pred = pd.DataFrame()
dataset_pred['learner_type'] = X_test['learner_type']
dataset_pred['y_test'] = y_test
X_train['learner_type']=X_train['learner_type'].replace({'Missing':0,'Passive participant':4,'Passive':3,'Active participant':6,'Active':5,'Drop-in':2,'Observer':1})
X_test['learner_type']=X_test['learner_type'].replace({'Missing':0,'Passive participant':4,'Passive':3,'Active participant':6,'Active':5,'Drop-in':2,'Observer':1})
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(X_train)
reduced = pca.transform(X_train)
X_train_reduced = pd.DataFrame(reduced)
print(X_train_reduced)

#pca_test = PCA(n_components = 0.95)
reduced_test = pca.transform(X_test)
X_test_reduced = pd.DataFrame(reduced_test)
print(X_test_reduced)

#fitting multiple linear regression to training dataset
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train_reduced,y_train)

#predicting the test sets results
y_pred = linear_regression.predict(X_test_reduced)
print(len(y_pred))
dataset_pred['y_pred'] = y_pred

RMSE_values = []
#calculating RMSE value
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
RMSE_values.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#building model using backward elimination and forward elimination
X_train = pd.DataFrame(X_train)
X_test1 = pd.DataFrame(X_test)
print(X_train.shape)
print(y_train.head())
import statsmodels.api as sm
def stepwise_selection_fs(X1, y1, X2,y2,
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X1.columns)-set(included)) 
        print("excluded",excluded)
        print('included',included)
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            included.append(new_column)
            model_fs = sm.OLS(y1, sm.add_constant(pd.DataFrame(X1[included]))).fit()
            print(model_fs.summary())
            new_pval[new_column] = model_fs.pvalues[new_column]
            print(new_pval)
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            changed=True
            best_feature = new_pval.argmin()
            included.append(best_feature)
            print("in loop",included)
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        if not changed:
            break
    print('before prediction',included)
    y_pred_fs = model_fs.predict(X2[included])
    print('RMSE value for forward selection',np.sqrt(metrics.mean_squared_error(y2, y_pred_fs)))
    RMSE_values.append(np.sqrt(metrics.mean_squared_error(y2, y_pred_fs)))
    print(y_pred_fs)
    # dataset_pred['y_pred_fs'] = y_pred_fs
    return included
result_fs = stepwise_selection_fs(X_train,y_train,X_test1,y_test)
print('resulting features for forward selection:',result_fs)

def stepwise_selection_bs(X1, y1, X2, y2,
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(set(X1.columns))
    while True:
        changed=False
        # backward step
        print(included)
        model_bs = sm.OLS(y1, sm.add_constant(pd.DataFrame(X1[included]))).fit()
        print(model_bs.summary())
        # use all coefs except intercept
        pvalues = model_bs.pvalues.iloc[1:].values
        print(pvalues)
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    X_bs_test = sm.add_constant(pd.DataFrame(X2[included]))
    print(X_bs_test.shape)
    y_pred_bs = model_bs.predict(X_bs_test)
    print('RMSE value for backward selection',np.sqrt(metrics.mean_squared_error(y2, y_pred_bs)))
    RMSE_values.append(np.sqrt(metrics.mean_squared_error(y2, y_pred_bs)))
    print(y_pred_bs)
    #dataset_pred['y_pred_bs'] = y_pred_bs
    return included
result_bs = stepwise_selection_bs(X_train,y_train,X_test1,y_test)
print('resulting features for backward selection:',result_bs)


import matplotlib.pyplot as plt
bar_x_axis = ["Linear Regression", "Stepwise-Forward", "Stepwise-Backward"]
plt.title("RMSE vs Models")
plt.bar(bar_x_axis, RMSE_values)
plt.show()

#writer = pd.ExcelWriter('D:\SPRING 2020\ICSI531\Project\Canvas_Network\output1.xlsx')
#dataset_pred.to_excel(writer)
#writer.save()
#print('DataFrame is written successfully to Excel File.')



