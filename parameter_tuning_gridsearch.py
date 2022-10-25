import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier

# Import dataset
df = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')

def create_model():

    model = Sequential()
    model.add(Dense(128, input_shape=(160,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))
    	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

# Load dataset
X,y = df['processed_audio'], np.array(df['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

# Get list of the emotions
emotion_list = np.unique(y)

# Get dummy values for emotion - data needs to be numerical for NN
y_dum = pd.get_dummies(y)

# create model
model = KerasClassifier(model=create_model, verbose=0)

# define the grid search parameters
batch_size = [20]
epochs = [10, 200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
grid_result = grid.fit(xxl, y_dum)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))