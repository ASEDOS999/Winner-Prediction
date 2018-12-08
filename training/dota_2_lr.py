import time
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as CBC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as cvs
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt


def preprocessing_data(data):
	number_of_heroes = 112
	data = data.fillna(value = 0)
	
	scaler = StandardScaler()
	
	#Drop categorial features
	new = data.copy()
	new.drop(['r%d_hero'%(i) for i in range(1, 6, 1)], axis = 1, inplace = True)
	new.drop(['d%d_hero'%(i) for i in range(1, 6, 1)], axis = 1, inplace = True)
	new.drop('lobby_type', axis = 1, inplace = True)
	scaler.fit(new)
	new = scaler.transform(new)
	
	#Word bag
	X_pick = np.zeros((data.shape[0], number_of_heroes))
	
	for i, match_id in enumerate(data.index):
		for p in range(5):
			X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
			X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
	data_categ = np.hstack((X_pick, new))
	return data_categ

def number_of_heroes(data):
	#Number of heroes
	ser = pd.Series([])
	for i in range(1, 6, 1):
		ser = pd.concat([ser, data['r%d_hero'%(i)]], axis = 0)
		ser = pd.concat([ser, data['d%d_hero'%(i)]], axis = 0)
	num_of_heroes = len(ser.unique())
	print('Number of unique heroes', num_of_heroes)
	N_max = ser.max()
	print('Max id of heroes', N_max)

def logis_regr(data):
	est = KFold(shuffle = True, n_splits = 5, random_state = 241)
	
	#Finding new best parameter
	test = [i for i in range(0, 5, 1)] 
	acc = []
	for p in test:
		clf = LR(penalty = 'l2', C = (4 + p/2)/100)
		acc.append(np.mean(cvs(clf, data_categ, Y, cv = est, scoring = 'roc_auc')))
	plt.plot(test, acc, 'ro')
	plt.plot(test, acc, 'r')
	plt.title('New best parameter')
	plt.show()


data = pd.read_csv("feat_new.csv", index_col = 'match_id')
Y = data['radiant_win']
data.drop(data.columns[-2:], axis = 1, inplace = True)

est = KFold(shuffle = True, n_splits = 5, random_state = 241)
data_categ = preprocessing_data(data.copy())

#From test of logistic regression
p_best = 5.0/100

clf = LR(penalty = 'l2', C = p_best)
acc_lr_categ = ( np.mean(cvs(clf, data_categ, Y, cv = est, scoring = 'roc_auc')))
acc_lr_categ_ac = ( np.mean(cvs(clf, data_categ, Y, cv = est, scoring = 'accuracy')))
print('Quality of LR with categorial features using word bag(roc_auc)', acc_lr_categ, '\n')
print('Quality of LR with categorial features using word bag(accuracy)', acc_lr_categ_ac, '\n')
