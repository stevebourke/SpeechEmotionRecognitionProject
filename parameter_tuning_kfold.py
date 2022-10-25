import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

df = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')


# Target variable is emotion - convert df columns to numpy arrays
X,y = df['processed_audio'], np.array(df['emotion'])

# Convert input variable to list and then to numpy array
Xl = list(X)
xxl = np.array(Xl)

    
# Get dummy values for emotion - data needs to be numerical for NN
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y_dum = np_utils.to_categorical(encoded_Y)
    

# Set seed to prevent randomness of results
seed = 123

tf.random.set_seed(seed)

# Set up k-fold parameters
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)


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


estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, xxl, y_dum, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Baseline: 54.65% (3.68%) with 200 epochs
