import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready.json')

# Put processed audio column into a list
list_all = df['processed_audio'].to_list()

# Create a separate column for each value from librosa
# 1440 columns of 160 values each to try out different ML algorithms
# list_ready = list(zip(*list_all))
df1 = pd.DataFrame(list_all)

# Target variable is emotion
X,y = df1, df['emotion']

# Split into test and train data
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=32)


# Copied from pythoncode...just to see if my data will work
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}

model = MLPClassifier(**model_params)

model.fit(x_train, y_train)

yPred= model.predict(x_test)



accuracy = accuracy_score(y_true=y_test, y_pred=yPred)