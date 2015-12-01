
import pandas as pd
#import pydot#
import numpy as np
#import graphviz
import math
from sklearn.externals.six import StringIO
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from os import system
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn import linear_model
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime

import matplotlib.pyplot as plt



#Reading the training and Testing Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

ids = test['Id']
y = train['Hazard']
train = train.drop(['Hazard', 'Id'], axis=1)
test = test.drop(['Id'], axis=1)

#get the categorical columns
fact_cols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11','T1_V12', 'T1_V15','T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12','T2_V13']

#time evalution
start_time = datetime.now()


#Preprocessing of the data using Label Encoder
lbl = preprocessing.LabelEncoder()
for column in fact_cols:
    train[column+'_new'] = lbl.fit_transform(train[column])
    test[column+'_new'] = lbl.fit_transform(test[column])
train_data = train.drop(fact_cols,axis=1)
test_data = test.drop(fact_cols,axis=1)
train_data = train_data.astype(float)
test_data = test_data.astype(float)

train_data.drop('T2_V10', axis=1, inplace=True)
train_data.drop('T2_V7', axis=1, inplace=True)
train_data.drop('T1_V13', axis=1, inplace=True)
train_data.drop('T1_V10', axis=1, inplace=True)

test_data.drop('T2_V10', axis=1, inplace=True)
test_data.drop('T2_V7', axis=1, inplace=True)
test_data.drop('T1_V13', axis=1, inplace=True)
test_data.drop('T1_V10', axis=1, inplace=True)

############ PCA REDUCTION ##########
#pca=PCA(n_components=2)
#pca.fit(train_data)
#pca.fit(test_data)
#plt.plot(train_data,'ro')
#plt.show()

#put the numerical as matrix

train_data1 = np.array(train_data.as_matrix(columns = None), dtype=object).astype(np.int)
test_data1 = np.array(test_data.as_matrix(columns=None), dtype=object).astype(np.int)




########### RANDOM FOREST REGRESSOR ##########
n_neighbors=8

rf = ensemble.RandomForestRegressor(n_estimators=200, max_depth=15)
rf.fit(train_data1, y)
pred1 = rf.predict(test_data1)

########### DECISION TREE REGRESSOR ##########
#dt =tree.DecisionTreeRegressor(max_depth=15)
#dt.fit(train_data1, y)

########### K NEIGHBOUR REGRESSOR ##########
#pred2 = dt.predict(test_data1)
#knn =KNeighborsRegressor(n_neighbors=9)
#knn.fit(train_data1, y)
#pred3 = knn.predict(test_data1)

########### GRADIENT BOOSTING REGRESSOR ##########
gb = ensemble.GradientBoostingRegressor(n_estimators=150, max_depth=6)
gb.fit(train_data1, y)
pred4 = gb.predict(test_data1)

########### SVM MODEL ##########
#svm1 = svm.SVR()
#svm1.fit(train_data1, y)
#pred5 = svm1.predict(test_data1)

###########XGBOOST###########################
def xgboost_pred(train,labels,test):
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.005
	params["min_child_weight"] = 6
	params["subsample"] = 0.7
	params["colsample_bytree"] = 0.7
	params["scale_pos_weight"] = 1
	params["silent"] = 1
	params["max_depth"] = 9
    
    
	listOfParameters = list(params.items())
 
	offset = 4000

	num_rounds = 10000
	xgtest = xgb.DMatrix(test)

	#create a train and validation dmatrices 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#predicition
	evallist = [(xgtrain, 'train'),(xgval, 'val')]
	modelXGB = xgb.train(listOfParameters, xgtrain, num_rounds, evallist, early_stopping_rounds=10)
	preds1 = modelXGB.predict(xgtest,ntree_limit=modelXGB.best_iteration)


	#reverse train and labels 
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	evallist = [(xgtrain, 'train'),(xgval, 'val')]
	modelXGB = xgb.train(listOfParameters, xgtrain, num_rounds, evallist, early_stopping_rounds=120)
	preds2 = modelXGB.predict(xgtest,ntree_limit=modelXGB.best_iteration)


	#combine predictions
	preds = preds1*1.4 + preds2*8.6
	return preds
########### XGB MODEL ##########
pred7=xgboost_pred(train_data1,y,test_data1)


########### NEURAL NETWORK MODEL ##########
log = linear_model.LogisticRegression()
NN_BRBM_model_1 = BernoulliRBM(n_components=3, learning_rate = 0.1)
cls1 = Pipeline(steps=[('rbm', NN_BRBM_model_1), ('logistic', log)]).fit(train_data1,y)


########### CROSS VALIDATING NN MODEL ##########

KF_NN1 = cross_validation.KFold(len(train_data1), n_folds=10, shuffle=True, random_state=4)
Score_NN1 = cross_validation.cross_val_score(cls1, train_data1, y, cv=KF_NN1, n_jobs=1)

print "Neural Network Accuracy : "+str(Score_NN1.mean())

pred6=cls1.predict(test_data1)

######## Comination of Predicitions #############
pred=(pred1*0.05)+(pred4*0.15)+(pred6*0.5)+(pred7*0.3)

preds = pd.DataFrame({"Id": ids, "Hazard": pred})

preds = preds[['Id', 'Hazard']]

preds.to_csv('result_random_with_time.csv', index=False)

end_time = datetime.now()
time_taken = (end_time - start_time)
print time_taken



