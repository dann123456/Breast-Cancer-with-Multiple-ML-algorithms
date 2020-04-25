# Load libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.impute import SimpleImputer 

def preproc(path_to_file):
	df = pd.read_csv(path_to_file)

	# convert to int
	df['class'].replace('class1', 0, inplace=True)
	df['class'].replace('class2', 1, inplace=True)

	# convert the ? character to np.nan
	df.replace("?", np.nan, inplace=True)

	# Create x column's values as floats
	x = df.loc[:, df.columns[0]:df.columns[-2]].values.astype(float)
	# replace np.nan with mean of cols
	
	imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
	imputer = imputer.fit(x)
	x = imputer.transform(x)

	# Create a minimum and maximum processor object
	min_max_scaler = preprocessing.MinMaxScaler()
	# Create an object to transform the data to fit minmax processor
	x_scaled = min_max_scaler.fit_transform(x)
	# Run the normalizer on the dataframe
	df_normalized = pd.DataFrame(x_scaled)

	df_normalized = df_normalized.apply(lambda x: round(x, 4))

	return pd.concat([df_normalized, df.loc[:, df.columns[-1] ] ], axis=1) 

def kNNClassifier(X, y, K):
    name, model = ("K-Nearest Neighbor", KNeighborsClassifier(n_neighbors = K)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def logregClassifier(X, y):
    name, model = ("Logistic Regression", LogisticRegression(random_state=0)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def nbClassifier(X, y):
    name, model = ("Naive Bayes", GaussianNB()) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def dtClassifier(X, y):
    name, model = ("Decision Tree", DecisionTreeClassifier(criterion='entropy', random_state=0)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    name, model = ("Bagging", BaggingClassifier(DecisionTreeClassifier(criterion='entropy', max_depth = max_depth, random_state = 0), n_estimators = n_estimators, max_samples = max_samples, random_state=0)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    name, model = ("Ada Boost", AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth = max_depth, random_state = 0), n_estimators = n_estimators, learning_rate = learning_rate, random_state=0)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

def gbClassifier(X, y, n_estimators, learning_rate):
    name, model = ("Gradient Boosting", GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = learning_rate, random_state=0)) 
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    msg = "%.4f" % scores.mean()
    print(msg, end="")
    
    return

warnings.filterwarnings('ignore')

def bestLinCLassifier(X_train, X_test, Y_train, Y_test):

    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

    param_grid = dict(C=c_values, gamma=gamma_values)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    grid_search = GridSearchCV(SVC(kernel="linear", random_state=0), param_grid, cv=kfold, return_train_score=True)
    grid_search.fit(X_train, Y_train)

    # You can also show these results:
    print("{}".format(grid_search.best_params_["C"]))
    print("{}".format(grid_search.best_params_["gamma"]))
    print("{:.4f}".format(grid_search.best_score_))
    # Accuracy on test set of the model with selected best parameters:
    print("{:.4f}".format(grid_search.score(X_test, Y_test)), end="")
    
    return

def bestRFClassifier(X_train, X_test, Y_train, Y_test):
    n_values = [10, 20, 50, 100]
    max_ft = ['auto', 'sqrt', 'log2']
    max_leaf = [10, 20, 30]
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = dict(n_estimators=n_values, max_features=max_ft, max_leaf_nodes = max_leaf)

    grid_search = GridSearchCV(RandomForestClassifier(criterion="entropy", random_state=0), param_grid, cv=kfold, return_train_score=True)
    grid_search.fit(X_train, Y_train)

    # You can also show these results:
    print("{}".format(grid_search.best_params_["n_estimators"]))
    print("{}".format(grid_search.best_params_["max_features"]))
    print("{}".format(grid_search.best_params_["max_leaf_nodes"]))
    print("{:.4f}".format(grid_search.best_score_))
    # Accuracy on test set of the model with selected best parameters:
    print("{:.4f}".format(grid_search.score(X_test, Y_test)), end="")
    
    return

# below function is to process clean files only ---------------------------------------------------------------

def get_train_test_split(path_to_file_csv):
	df = pd.read_csv(path_to_file_csv)

	data = df.loc[:, df.columns[0]:df.columns[-2]]
	label = df.loc[:, df.columns[-1] ]

	X_train, X_test, Y_train, Y_test = train_test_split(data, label, stratify = label, random_state=0)

	return X_train, X_test, Y_train, Y_test

def get_data(path_to_file_csv):
	df = pd.read_csv(path_to_file_csv)
	X = df.loc[:, df.columns[0]:df.columns[-2]]
	y = df.loc[:, df.columns[-1] ]
	return X, y

def get_param(path_to_file_param):
	df = pd.read_csv(path_to_file_param)
	return df

#===============================================================================================================

from sys import argv

if len(argv) == 3: # inclusive of this .py file
	#arg[0] is the .py file
	#arg[1] is the csv file
	#arg[2] is the choice of algo
	
	if argv[2] == "P": # user want to pre proc the input data
		df = preproc(argv[1])
		temp = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
		lines = [','.join(ele.split()) for ele in temp]
		for line in lines:
			print(line)

	elif argv[2] == "LR":
		X, y = get_data(argv[1])
		res = logregClassifier(X, y)

	elif argv[2] == "NB":
		X, y = get_data(argv[1])
		res = nbClassifier(X, y)

	elif argv[2] == "DT":
		X, y = get_data(argv[1])
		res = dtClassifier(X, y)

	elif argv[2] == "RF":
		X, x, Y, y = get_train_test_split(argv[1])
		res = bestRFClassifier(X, x, Y, y)

	elif argv[2] == "SVM":
		X, x, Y, y = get_train_test_split(argv[1])
		res = bestLinCLassifier(X, x, Y, y)
	else:
		print("no option")

elif len(argv) == 4:
	#arg[0] is the .py file
	#arg[1] is the csv file
	#arg[2] is the choice of algo 
	#arg[3] is the param csv
	
	if argv[2] == "NN":
		X, y = get_data(argv[1])
		K = get_param(argv[3])["K"].values[0]
		
		res = kNNClassifier(X, y, int(K))

	elif argv[2] == "BAG":
		X, y = get_data(argv[1])
		n = get_param(argv[3])["n_estimators"].values[0]
		lr = get_param(argv[3])["max_samples"].values[0]
		max_d = get_param(argv[3])["max_depth"].values[0]

		res = bagDTClassifier(X, y, n, lr, max_d)

	elif argv[2] == "ADA":
		X, y = get_data(argv[1])
		n = get_param(argv[3])["n_estimators"].values[0]
		lr = get_param(argv[3])["learning_rate"].values[0]
		max_d = get_param(argv[3])["max_depth"].values[0]

		res = adaDTClassifier(X, y, n, lr, max_d)

	elif argv[2] == "GB":
		X, y = get_data(argv[1])
		n = get_param(argv[3])["n_estimators"].values[0]
		lr = get_param(argv[3])["learning_rate"].values[0]

		res = gbClassifier(X, y, n, lr)