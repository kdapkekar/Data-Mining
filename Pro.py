#                           CSI 531 Data Mining Project
#                           Group 6
#                           Spring 2020
#########################################################       Imports   ########################################################################################

#Dataframe
import pandas as pd
#Numeric and plotting
import numpy as np
import matplotlib.pyplot as plt
#To label the data
from sklearn.preprocessing import LabelEncoder
#utility to split data
from sklearn.model_selection import train_test_split
#utility to standardize data
from sklearn.preprocessing import StandardScaler
#Dimensionality reduction
from sklearn.decomposition import PCA
#Regression model
from sklearn.linear_model import LinearRegression
#metrics for evaluation of model
from sklearn import metrics
#Resampling data
from sklearn.utils import resample
import statsmodels.api as sm


#################################################   Data Preprocessing      #####################################################################################

#Reading the dataset
dataset=pd.read_csv("CNPC_1401-1509_DI_v1_1_2016-03-01.csv",nrows = 37080)
np.seterr(divide='ignore', invalid='ignore')
#Resampling the dataset
majority=dataset[dataset.grade!=0]
minority=dataset[dataset.grade==0]
#Decreasing the number of rows with grade 0 to 10000
resampling = resample(minority,replace=True,n_samples=10000,random_state=50)
dataset=pd.concat([majority,resampling])
#Drop all the unwanted data
to_drop=['course_id_DI','discipline','userid_DI','explored','grade_reqs','course_reqs','final_cc_cname_DI','LoE_DI','age_DI','gender','start_time_DI','course_start','course_end','last_event_DI','ncontent']
dataset.drop(to_drop, inplace=True, axis=1)
dataset=pd.DataFrame(dataset)
#preprocess the rest of data
dataset['learner_type']=dataset['learner_type'].fillna('Missing')
dataset['expected_hours_week']= dataset['expected_hours_week'].replace({'Between 4 and 6 hours':5,'Between 2 and 4 hours':3,'Between 1 and 2 hours':2,'Less than 1 hour':1,'Between 6 and 8 hours':7,'More than 8 hours per week':10,'Missing':0})
dataset=dataset.fillna(0)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(0,50),1)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(50,100),2)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100,500),3)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500,1000),4)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(1000,5000),5)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(5000,10000),6)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(10000,100000),7)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(100000,500000),8)
dataset['nevents']=dataset['nevents'].mask(dataset['nevents'].between(500000,1000000),9)


#Assiging dataset
y = dataset.iloc[:,2].values
X=dataset

#writing the data to an excel file
y = pd.DataFrame(y)
#Please give a valid path and valid excel file name here
writer = pd.ExcelWriter('D:\DM\Project\output.xlsx')
dataset.to_excel(writer)
writer.save()
print('DataFrame is written successfully to Excel File.')

#Encoding Labels
lableEncoderObj = LabelEncoder()
X['primary_reason'] = lableEncoderObj.fit_transform(X['primary_reason'].astype('str'))
X = X.drop(['grade'],axis=1)

#Splitting into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=0,shuffle=False)
dataset_pred = pd.DataFrame()
X_test=pd.DataFrame(X_test)
X_train=pd.DataFrame(X_train)
dataset_pred['learner_type'] = X_test['learner_type']
dataset_pred['Y_test'] = Y_test
X_train['learner_type']=X_train['learner_type'].replace({'Missing':0,'Passive participant':4,'Passive':3,'Active participant':6,'Active':5,'Drop-in':1,'Observer':2})
X_test['learner_type']=X_test['learner_type'].replace({'Missing':0,'Passive participant':4,'Passive':3,'Active participant':6,'Active':5,'Drop-in':1,'Observer':2})

#feature scaling
standardScalerObj = StandardScaler()
X_train = standardScalerObj.fit_transform(X_train)
X_test = standardScalerObj.transform(X_test)

##################################################################PCA  Dimensionality Reduction ###########################################################################

#Dimensionality reduction using PCA
pca = PCA(n_components = 0.95)
pca.fit(X_train)
reduced = pca.transform(X_train)
X_train_reduced = pd.DataFrame(reduced)
##print(X_train_reduced)
#
pca_test = PCA(n_components = 0.95)
reduced_test = pca.transform(X_test)
X_test_reduced = pd.DataFrame(reduced_test)

######################################################## Linear Regression model ################################################################################

#Linear Regression without PCA
#fitting multiple linear regression to training dataset
linearRegressionObj = LinearRegression()
linearRegressionObj.fit(X_train,Y_train)

#predicting the test sets results
Y_pred = linearRegressionObj.predict(X_test)
RMSE_values = []
dataset_pred['Y_pred'] = Y_pred
#calculating RMSE value
print("The RMSE using linear regression without PCA is",np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
RMSE_values.append(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

#fitting multiple linear regression to training dataset
linearRegressionObj = LinearRegression()
linearRegressionObj.fit(X_train_reduced,Y_train)

#predicting the test sets results
Y_pred = linearRegressionObj.predict(X_test_reduced)
#calculating RMSE value
print("The RMSE using linear regression with PCA is",np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
RMSE_values.append(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

#building model using backward elimination and forward elimination
X_train = pd.DataFrame(X_train)
X_test1 = pd.DataFrame(X_test)

##########################################################Forward approach ########################################################

#function implementing forward stepwise approach
def forwardStepwiseSelection(stepwise_x_train, stepwise_y_train, stepwise_x_test,stepwise_y_test):
    #Threshold value for forward selection
    innnerThreshold = 0.01
    empty_list = []
    #List to store the included vaaribales
    final_included = []
    includedColumns = list(empty_list)
    while True:
        updated=False
        #In forward approach intially all the coloumns are excluded, Then includes one by one
        excludedColumns = list(set(stepwise_x_train.columns) - set(includedColumns))
        #New P value
        updatedPValue = pd.Series(index=excludedColumns,dtype=float)
        for column in excludedColumns:
            #Include new columns
            includedColumns.append(column)
            #train the model with included columns
            forwardStepwiseModel = sm.OLS(stepwise_y_train, sm.add_constant(pd.DataFrame(stepwise_x_train[includedColumns]))).fit()
            #updated p value for every column
            updatedPValue[column] = forwardStepwiseModel.pvalues[column]
        #best p value so far
        bestPValue = updatedPValue.min()
        #if this values is less than inner threshold, the newly included feature is considered best feature
        if bestPValue < innnerThreshold:
            bestFeature = updatedPValue.argmin()
            #including the best feature
            final_included.append(bestFeature)
            updated=True
        if not updated:
            break
    #print the values before prediction
    print('before prediction',includedColumns)
    #print the final included columns
    print(final_included)
    #training the model with final included columns
    forwardStepwiseModel = sm.OLS(stepwise_y_train, sm.add_constant(pd.DataFrame(stepwise_x_train[final_included]))).fit()
    #print summary of the model
    print(forwardStepwiseModel.summary())
    X_fs_test = sm.add_constant(pd.DataFrame(stepwise_x_test[final_included]))
    #Pass the Test data and get predicted values
    Y_pred_forwardStepwise = forwardStepwiseModel.predict(X_fs_test)
    #Compute the RMSE with obtained predictions
    print('RMSE value for forward selection',np.sqrt(metrics.mean_squared_error(stepwise_y_test, Y_pred_forwardStepwise)))
    #adding to RMSE_values to plot
    RMSE_values.append(np.sqrt(metrics.mean_squared_error(stepwise_y_test, Y_pred_forwardStepwise)))
    dataset_pred['Y_pred_forwardStepwise'] = Y_pred_forwardStepwise
    return final_included

#Call the forward stepwise approach function
result_fs = forwardStepwiseSelection(X_train,Y_train,X_test1,Y_test)
print('resulting features for forward selection:',result_fs)


################################################### Backward Approach #########################################################################

#function to implement backward stepwise approach
def backwardstepwiseselection(stepwise_x_train, stepwise_y_train, stepwise_x_test, stepwise_y_test):
    #Threshold value for backward approach
    outterThreshold = 0.05
    prev_worst_feature=20
    #All the columns are included in the begining
    includedColumns = list(set(stepwise_x_train.columns))
    while True:
        updated=False
        print(includedColumns)
        #Train the models with included columns
        backwardStepwiseModel = sm.OLS(stepwise_y_train, sm.add_constant(pd.DataFrame(stepwise_x_train[includedColumns]))).fit()
        #pvalues of the model so far
        PValues = backwardStepwiseModel.pvalues.iloc[1:].values
        print(PValues)
        #columns with higher pValue are worse
        worstPValue = PValues.max() # null if PValues is empty
        #if the value is greater than threshold, include it in worse features and remove it.
        if worstPValue > outterThreshold and PValues.argmax()!= prev_worst_feature:
            prev_worst_feature = PValues.argmax()
            worstFeature = PValues.argmax()
            includedColumns.remove(worstFeature)
            updated=True
        if not updated:
            break
    #print the summary of backward approach
    print(backwardStepwiseModel.summary())
    X_bs_test = sm.add_constant(pd.DataFrame(stepwise_x_test[includedColumns]))
    #obtain predictions of the test data
    Y_pred_bs = backwardStepwiseModel.predict(X_bs_test)
    #compute and print Root Mean Square Error
    print('RMSE value for backward selection',np.sqrt(metrics.mean_squared_error(stepwise_y_test, Y_pred_bs)))
    #Append the RMSE values for plotting
    RMSE_values.append(np.sqrt(metrics.mean_squared_error(stepwise_y_test, Y_pred_bs)))
    #include the predictions in dataset
    dataset_pred['Y_pred_backwardStepwise'] = Y_pred_bs
    return includedColumns

#call the backward function and print the included columns at end
result_bs = backwardstepwiseselection(X_train,Y_train,X_test1,Y_test)
print('resulting features for backward selection:',result_bs)

########################################################  Data visualization #######################################################

bar_x_axis = ["Linear Regression without PCA","Linear Regression with PCA", "Stepwise-Forward", "Stepwise-Backward"]
plt.title("RMSE vs Models")
plt.bar(bar_x_axis, RMSE_values)
plt.show()

#writing the dataset with predicted values into an Excel file
#Please give a valid path and valid excel file name here
writer = pd.ExcelWriter('D:\DM\Project\output1.xlsx')
dataset_pred.to_excel(writer)
writer.save()
print('DataFrame is written successfully to Excel File.')
