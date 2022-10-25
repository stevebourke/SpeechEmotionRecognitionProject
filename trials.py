import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from audio_NeuralNetworks import create_multi_model
from sklearn.model_selection import GridSearchCV


df_ravdess = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')
df_tess = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_tess.json')
df_mine = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_mine.csv')
df_crema = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_crema.json')


######   Decision Trees   ######


# Target variable is emotion - convert df columns to numpy arrays
X,y = df_crema['processed_audio'], np.array(df_crema['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

# Set seed to prevent randomness of results
seed = 123
np.random.seed(seed)
    

# Split into test and train data
x_train, x_test, y_train, y_test = train_test_split(xxl,y, test_size=.2, random_state=32)
    


from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier()
DT_model = DT_model.fit(x_train, y_train)

y_pred = DT_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)


# Accuracy is approx 40% on test data with ravdess
# Accuracy is approx 93% on test data with tess
# Accuracy is approx 34% on test data with crema
cross_tab = pd.crosstab(y_test, y_pred, dropna=False)

print(cross_tab)

print ("Accuracy score", accuracy)



# Try the model on my data
df1 = df_mine.loc[:, df_mine.columns!='emotion']

# This collects each row of values into a numpy array
xx = df1.values
y_mine = np.array(df_mine['emotion'])

y_pred_mine = DT_model.predict(xx)
accuracy_mine = accuracy_score(y_mine, y_pred_mine)

cross_tab = pd.crosstab(y_mine, y_pred_mine, dropna=False)

print(cross_tab)

print ("Accuracy score", accuracy_mine)

# Accuracy is approx 25% on my data with ravdess
# Accuracy is approx 14% on my data with tess
# Accuracy is approx 18% on my data with crema





##############################################################



#######   S V M   #######

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

x_train_SVC = scale.transform(x_train)
x_test_SVC = scale.transform(x_test)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_train_SVC, y_train)

svm_pred = clf.predict(x_test_SVC)

accuracy_svm = accuracy_score(y_test, svm_pred)

cross_tab = pd.crosstab(y_test, svm_pred, dropna=False)

print(cross_tab)

print ("Accuracy score", accuracy_svm)
# Accuracy is approx 41% on test data with ravdess
# Accuracy is approx 98.6% on test data with tess
# Accuracy is approx 42.5% on test data with crema



### Try the model on my data ###

# Scale it first
xx_SVM = scale.transform(xx)

svm_pred_mine = clf.predict(xx_SVM)
accuracy_mine_svm = accuracy_score(y_mine, svm_pred_mine)

cross_tab = pd.crosstab(y_mine, svm_pred_mine, dropna=False)

print(cross_tab)

print ("Accuracy score", accuracy_mine_svm)
# Accuracy is approx 10% on my data with ravdess
# Accuracy is approx 10.7% on my data with tess
# Accuracy is approx 29% on my data with crema



##############################################################


########   SVM with multi-models   ########


### Ravdess-Crema ###

df_crema_ravdess = create_multi_model([df_ravdess, df_crema])

X,y = df_crema_ravdess['processed_audio'], np.array(df_crema_ravdess['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

# Set seed to prevent randomness of results
seed = 123
np.random.seed(seed)
    

# Split into test and train data
x_train, x_test, y_train, y_test = train_test_split(xxl,y, test_size=.2, random_state=32)
    
# Scale the data
scale = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

x_train_SVC = scale.transform(x_train)
x_test_SVC = scale.transform(x_test)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_train_SVC, y_train)

svm_pred = clf.predict(x_test_SVC)

accuracy_svm = accuracy_score(y_test, svm_pred)

print ("Accuracy score", accuracy_svm)

cross_tab = pd.crosstab(y_test, svm_pred, dropna=False)

print(cross_tab)

print(sum(np.diagonal(cross_tab))/np.sum(cross_tab.values))
# Accuracy is 42.3% on test data


# Scale it first
xx_SVM = scale.transform(xx)

svm_pred_mine = clf.predict(xx_SVM)
accuracy_mine_svm = accuracy_score(y_mine, svm_pred_mine)

print ("Accuracy score", accuracy_mine_svm)

cross_tab = pd.crosstab(y_mine, svm_pred_mine, dropna=False)

print(cross_tab)

print(sum(np.diagonal(cross_tab))/np.sum(cross_tab.values))
# Accuracy is 25% on my data





##############################################################



########   Random Forest   #########

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

# Target variable is emotion - convert df columns to numpy arrays
X,y = df_crema['processed_audio'], np.array(df_crema['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

# Set seed to prevent randomness of results
seed = 123
np.random.seed(seed)
    

# Split into test and train data
x_train, x_test, y_train, y_test = train_test_split(xxl,y, test_size=.2, random_state=32)
    

# Using GridSearch
model = RandomForestClassifier()

param_search = { 
    'n_estimators': [20, 50, 80],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [i for i in range(6,8)]
}

# tscv = TimeSeriesSplit(n_splits=10)
# gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = 'accuracy')

# gsearch.fit(x_train, y_train)
# best_score = gsearch.best_score_
# best_model = gsearch.best_estimator_

#print(best_model)


# Apply best model

#y_pred_mine = best_model.predict(xx)

print("accuracy", accuracy_score(y_mine, y_pred_mine))
# CREMA
# 43.2% accuracy on test data
# 25% accuracy on my data



###########################################################



##############   S V M w/ GridSearch   ################

# Was taking too long!!!

X,y = df_ravdess['processed_audio'], np.array(df_ravdess['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

# Set seed to prevent randomness of results
seed = 123
np.random.seed(seed)
    

# Split into test and train data
x_train, x_test, y_train, y_test = train_test_split(xxl,y, test_size=.2, random_state=32)
    
# Scale the data
scale = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
model = svm.SVC()

param_search = {
    'C' : [1, 10.0],
    'gamma' : [1, 10.0],
    'degree' : [1,2],
    'kernel' : ['linear', 'poly']
    }

tscv = TimeSeriesSplit(n_splits=10)
gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = 'accuracy')

gsearch.fit(x_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

print(best_model)


# Apply best model

y_pred_mine = best_model.predict(xx)

print("accuracy", accuracy_score(y_mine, y_pred_mine))