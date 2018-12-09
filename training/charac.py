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
