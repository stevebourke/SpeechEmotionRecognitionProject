import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt


df_ravdess = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')
df_tess = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_tess.json')
df_mine = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_mine.csv')
df_mine_180 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_mine_180.csv')
df_crema = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_crema.json')
df_crema_180 = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_crema_180.json')
df_emo_db = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_emo_db.json')
df_x4ntho55 = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_train_custom.json')
df_savee = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_savee.json')


df_ravdess_mfcc = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\mfcc_audio_ready_ravdess.json')
df_ravdess_mfcc2 = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\mfcc2_audio_ready_ravdess.json')


def create_NN_model(df, name):
    """
    Input a dataframe containing the processed audio and create a neural network
    based on this audio data
    

    Parameters
    ----------
    df : A dataframe containing information about the specific dataset
    as well as the processed audio ready for use
    
    name : A string indicating which dataset is being entered to function

    Returns
    -------
    None but the neural network model is saved

    """
    import os
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    # Drop neutral (from ravdess)
    #df = df.loc[df['emotion'] != "neutral"]

    # Target variable is emotion - convert df columns to numpy arrays
    X,y = df['processed_audio'], np.array(df['emotion'])
    
    #Convert input variable to list and then to numpy array
    Xl = list(X)
    xxl = np.array(Xl)
    xxl = np.array(xxl)
    
    
    # Split into test and train data
    x_train, x_test, y_train, y_test = train_test_split(xxl,y, test_size=.2, random_state=32)
    
    # Reshape the input for the RNN
    #x_train = x_train.reshape((2599, 160, 1))
    #x_test = x_test.reshape((650, 160, 1))

    
    # Get dummy values for emotion - data needs to be numerical for NN
    lb = LabelEncoder()

    y_train_ohe = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test_ohe = np_utils.to_categorical(lb.fit_transform(y_test))
    
    
    # Specify number of output layers, depending on dataset being used
    output_layers = 8
    #if "ravdess" in name:
     #   output_layers = 8
        
    #if "ravdess_" in name:
     #   output_layers = 7   # No 'neutral' with strong intensity 
        
    # if "tess" in name:
    #     output_layers = 7
        
    # if "crema" in name:
    #     output_layers = 6
        
    # if "subset" in name:
    #     output_layers = 4
    


    # Set seed to prevent randomness of results
    tf.random.set_seed(123)
        
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, Flatten, LSTM
 
    
 
    ###   C N N   ###
    
    # model = Sequential()
    # model.add(Conv1D(256, 5,padding='same', input_shape=(160,1))) #1
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same')) #2
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(MaxPooling1D(pool_size=(8)))
    # model.add(Conv1D(128, 5,padding='same')) #3
    # model.add(Activation('relu'))
    # #model.add(Conv1D(128, 5,padding='same')) #4
    # #model.add(Activation('relu'))
    # #model.add(Conv1D(128, 5,padding='same')) #5
    # #model.add(Activation('relu'))
    # #model.add(Dropout(0.2))
    # model.add(Conv1D(128, 5,padding='same')) #6
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(7)) #7
    # model.add(Activation('softmax'))
    
    
    ###   R N N   ###
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(160, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    
  

    model = Sequential([
        Dense(128, input_shape=(160,), activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),        
        Dense(output_layers, activation='softmax')
    ])
    
    print(model.summary())
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    history = model.fit(x_train, y_train_ohe, epochs=500, verbose=0, validation_split = 0.1)
     
    
    df = pd.DataFrame(history.history)
    df['epoch'] = history.epoch
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    
    loss, accuracy = model.evaluate(x_test, y_test_ohe, verbose=1)
    
    # Get predicted labels for the test data
    y_pred = model.predict(x_test)
    
    # Convert these to the index of the max value, i.e. the emotion that best fits
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    predictions = (lb.inverse_transform((y_pred_classes)))
    
    # y_test = y_test.to_numpy()
    
    # # Find the value of 1 from the test labels (value is 1 as a result of OHEncoding)
    # y_test_classes = [np.where(r==1)[0][0] for r in y_test]
    
    # y_test_classes = np.asarray(y_test_classes)
    
    
    cross_tab = pd.crosstab(y_test, predictions)
    
    print(cross_tab)
    
    print(sum(np.diagonal(cross_tab))/np.sum(cross_tab.values))
    
    #Save the model using the name input to label it
    model_file = name + "_model.h5"
    
    if not os.path.isfile(model_file):
        model.save(model_file)
        
    else:
        os.remove(model_file)   # Maybe not the most elegant; remove model
        model.save(model_file)    # Replace with new version





def create_multi_model(df_list):
    """
    Combine two or more input datasets together to allow us to
    create a NN model based on this combination

    Returns
    -------
    The dataframe containing the two columns used in creating model

    """
    
    df_multi = pd.DataFrame()
    
    # Concatenate all emotion & audio columns from each datafram in the list
    # and put them in our multi df; axis=0 stacks them on top of each other
    df_multi['emotion'] = pd.concat([df_list[i]['emotion'] for i in range(0, len(df_list))], axis=0)
    df_multi['processed_audio'] = pd.concat([df_list[i]['processed_audio'] for i in range(0, len(df_list))], axis=0)

    # Drop the 'calm' rows (from ravdess) to keep the emotions balanced
    # Also drop 'surprised' to cater for crema dataset
    df_multi = df_multi[~df_multi['emotion'].isin(["calm", "boredom", "surprised", "fear"])]

    return df_multi
    

def test_model(model_NN, df):
    """
    

    Parameters
    ----------
    model : A h5 file containing the NN model we want to use
    df : A dataframe containing the data we wish to analyse


    Returns
    -------
    None.

    """

    from tensorflow.keras.models import load_model
    
    model = load_model(model_NN)
    
    # We need to get the list of emotions that correspond to the predicted classes
    emotion_list_pred = ()
    
    # Import the required dataframe - allows us to access the emotion list
    # used by the model; Crema has the least number of emotions, check for this
    # first, will also catch multi models
    if "crema" in model_NN:
        df_model = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_crema.json')

    elif "ravdess_tess_emo_db" in model_NN:
        df_model = create_multi_model([df_ravdess, df_tess, df_emo_db])

    elif "tess" in model_NN:
        df_model = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_tess.json')

    elif "emo_db" in model_NN:
        df_model = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_emo_db.json')
            
    elif "LSTM" in model_NN:
        df_model = create_multi_model([df_ravdess, df_tess, df_emo_db])

    elif "savee" in model_NN:
        df_model = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_savee.json')     
        
        
    elif "ravdess" in model_NN:
        df_model = pd.read_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')
        
        if "ravdess_no" in model_NN: # Remove 'neutral' if strong intensity
            df_model = df_model[df_model['emotion'] != "neutral"]
            
            

            
    # Create a list of the emotions that were in the model's dataset    
    emotion_list_pred = df_model["emotion"]
    
    # If using the 'subset' model reduce the emotions accordingly
    if "subset" in model_NN:
        if "ADHS" in model_NN:
            emotion_list_pred = ["happy", "angry", "sad", "disgust"]
        elif "AFHS" in model_NN:
            emotion_list_pred = ["happy", "angry", "sad", "fear"]
            
        elif "AHNPS" in model_NN:
            emotion_list_pred = ["angry","happy", "neutral", "surprised", "sad"]
    
    # Check for this model here...ps = surprised
    elif "AHNPS" in model_NN:
        emotion_list_pred = ["angry","happy", "neutral", "surprised", "sad"]

    emotion_list_pred = list(pd.unique(emotion_list_pred))
    
    # Drop rows from df where the emotion is not found in the emotion list of model
    df = df[df['emotion'].isin(emotion_list_pred)]
    
    
    # Add the emotions to a dictionary, thus giving them a corresponding index number
    emotion_dict_pred = {}
    
    # Sort the list as dummy values seem to be sorted!
    for i in range(0, len(emotion_list_pred)):
        emotion_dict_pred[i] = sorted(emotion_list_pred)[i]
        
    
    # A repeat of the code in the create_NN_model function - get the input
    # and output variables ready for feeding into the model
    # Target variable is emotion - convert df columns to numpy arrays
    y = np.array(df['emotion'])
    
    # If a downloaded dataset then the processed audio will be in one column
    try:
    
        X = df['processed_audio']

    # My own audio data was in csv as it is easier to append, so the librosa
    # results are spread across columns
    except:
        
        df1 = df.loc[:, df.columns!='emotion']
        
        # This collects each row of values into a numpy array
        X = df1.values
    
    # Convert input to list and then to numpy array
    Xl = list(X)
    xxl = np.array(Xl)
    set_length = len(xxl)
    # Reshape to fit the RNN model
    xxl = xxl.reshape((set_length, 1, len(xxl[0])))
    
    # Get predicted labels for the input data
    y_pred = model.predict(xxl)
    # Revert to 2D array
    y_pred = y_pred.reshape((set_length, len(emotion_list_pred)))
    
    
    # Convert these to the index of the max value, i.e. the emotion that best fits
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    
    # Change these numeric values to strings, i.e. the emotions
    y_pred_classes = list(y_pred_classes)
    y_pred_emos = []
    
    for emo in y_pred_classes:
        y_pred_emos.append(emotion_dict_pred[emo])
        
    # Convert to numpy array
    y_pred_emos = np.array(y_pred_emos)
    
    # Ensure that the full emotions are included in crosstab by setting them as categories
    y_pred_emos = pd.Categorical(y_pred_emos, categories=sorted(emotion_list_pred))
    

    # dropna=False ensures that all emotion categories are included even if empty
    cross_tab = pd.crosstab(y, y_pred_emos, dropna=False)
    
    print(cross_tab)
    
    print(sum(np.diagonal(cross_tab))/np.sum(cross_tab.values))
    




###################################################################



# Run the function passing in the datasets
#create_NN_model(df_ravdess, "ravdess")

# create_NN_model(df_tess, "tess")

# create_NN_model(df_crema, "crema")

#create_NN_model(df_emo_db, "emo_db")

#create_NN_model(df_savee, "savee")


#test_model('AHNPS-c-LSTM-layers-2-2-units-128-128-dropout-0.3_0.3_0.3_0.3.h5', df_mine_180)
#test_model('ravdess_tess_emo_db_RNN_model.h5', df_mine)
#test_model('ravdess_emotion_subset_ADHS_model.h5', df_mine)


# RAVDESS
# 59.4% accuracy on training data ... 65.3% with 1000 epochs
# 35.7% accuracy on my data ... but only 25% on mine with 1000

# Drop neutral emotion as it is performing poorly
# 64.3% accuracy on training data
# 32% on my data

# 62.45% with the CNN
# Only 13% on my own data

# Only 42.4% with the RNN !!
# Only 21.2% with RNN and complete MFCC (not the mean) !!!!

# 56.6% when just using mfcc alone

# 58.7% with the 180 markers
# 62.8% with the 180 markers and neutral dropped


# TESS
# 99.5% accuracy on training data
# 12% accuracy on my data


# CREMA
# 47.4% accuracy on training data!?!
# 15% accuracy on my data


# EMO_DB
# 72% accuracy on training data
# 12% accuracy on my data


# SAVEE
# 68.75% on training data
# 12% on my data


###############################################
#### Trying out different subsets of the input data ####



### Ravdess by gender ###

# 61.8% accuracy on training data
# 28.6% accuracy on my data
df_ravdess_male = df_ravdess[df_ravdess['gender'] == "Male"]
# 68% accuracy on training data
# 17.6% accuracy on my data
df_ravdess_female = df_ravdess[df_ravdess['gender'] == "Female"]
# create_NN_model(df_ravdess_female, "ravdess_female")



### Separate out the different emotional intensity sets in ravdess ###

# 69.6% accuracy on training data
# 24% on my data!?!
df_ravdess_strong_intensity = df_ravdess[df_ravdess['emotion_intensity'] == "strong"]
# 50.65% accuracy on training data
# 32.1% accuracy on my data
df_ravdess_normal_intensity = df_ravdess[df_ravdess['emotion_intensity'] == "normal"]
#create_NN_model(df_ravdess_normal_intensity, "ravdess_normal_intensity")



### Only use some of the emotions ###

df_ravdess_emotion_subset = df_ravdess[df_ravdess['emotion'].isin(["happy", "angry", "sad", "neutral", "surprised"])]
# Accuracy 63.6% on training data with AFHS
# Accuracy 72.7% on training data with ADHS
# Accuracy 72.25% on training data with AHNPS
#create_NN_model(df_ravdess_emotion_subset, "ravdess_emotion_subset_AHNPS")


df_tess_emotion_subset = df_tess[df_tess['emotion'].isin(["happy", "angry", "sad", "fear"])]
# Accuracy 100% on training data with AFHS!
# Accuracy 99.7% on training data with ADHS
#create_NN_model(df_tess_emotion_subset, "tess_emotion_subset_AFHS")


df_crema_emotion_subset = df_crema[df_crema['emotion'].isin(["happy", "angry", "sad", "fear"])]
# Accuracy 59.4% on training data with AFHS
# Accuracy 60% on training data with ADHS
#create_NN_model(df_crema_emotion_subset, "crema_emotion_subset_AFHS")

test_model('ravdess_emotion_subset_ADHS_model.h5', df_crema)
# Ravdess w/ ADHS on my data - 50% accuracy
# Ravdess w/ AFHS on my data - 50% accuracy

# Tess w/ ADHS on my data - 21.4% accuracy
# Tess w/ AFHS on my data - 43% accuracy

# Crema w/ ADHS on my data - 43% accuracy
# Crema w/ AFHS on my data - 35.7% accuracy


########################################################

## Combinations of datasets ##


# Create a model from ravdess and crema combined
# 46.5% accuracy on training data
# 14.3% accuracy on my data, exactly the same as crema on its own
df_crema_ravdess = create_multi_model([df_ravdess, df_crema])
# create_NN_model(df_crema_ravdess, "crema_ravdess")



# Create a model from ravdess and tess combined
# 86.7% accuracy on training data
# 11.8% accuracy on my data
df_tess_ravdess = create_multi_model([df_tess, df_ravdess])
# create_NN_model(df_tess_ravdess, "tess_ravdess")



# Create a model from crema and tess combined
# 57.5% accuracy on training data
# 15% accuracy on my data
df_tess_crema = create_multi_model([df_tess, df_crema])
# create_NN_model(df_tess_crema, "tess_crema")



# Create a model from ravdess, tess and crema combined
# 58.8% accuracy on training data
# Again 14.3% accuracy on my data
df_crema_tess_ravdess = create_multi_model([df_ravdess, df_crema, df_tess])
# create_NN_model(df_crema_tess_ravdess, "crema_tess_ravdess")



# Ravdess, Tess & Emo-DB
# 87.7% on training data
# 23.5% on mine - almost everything of mine being labelled as 'disgust'
df_ravdess_tess_emo_db = create_multi_model([df_ravdess, df_tess, df_emo_db])
#create_NN_model(df_ravdess_tess_emo_db, "ravdess_tess_emo_db_RNN")

# 77.2% on training using five emotions, same as x4ntho55 on github
# 35.3% on my data
# 36% on Crema
# Beware - took 4+ hours to train the model


# I ran x4ntho55 code - 77.2% accuracy on training, same as mine.
# Using the model it created gave 32% accuracy on my data
# Just 35.5% on crema datset