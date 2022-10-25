import librosa
import pandas as pd
import numpy as np
import os
import glob
import re
import soundfile



def convert_audio(audio_path, dataset_name):
    """
    Convert all audio files to mono if needed and change to 16kHz...
    

    Parameters
    ----------
    audio_path : file path of file to be altered
    dataset_name : identifies which audio data is being converted

    Returns
    -------
    The updated file in a different folder

    """

    base = os.path.basename(audio_path)    
        
    # Set names for new directory and files therewithin
    target_dir = dataset_name + "_converted_speech_files"
    target_path = dataset_name + "_converted_speech_files/" + base  
    
    # If directory does not already exist create one
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    
        
    # Only convert the file it is not already in 'converted' folder
    if not os.path.isfile(target_path):
    
        os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")


# Tips taken from https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn


# First reduce all files to mono and 16k Hz

#for file in glob.glob("ravdess_speech_files/*.wav"):    
    
#    convert_audio(file, "ravdess")


def extract_audio_features(audio_file, **kwargs):
    """   
    Feed in a sound file and use librosa to convert
    and extract the desired processing as indicated by the arguments

    Parameters
    ----------
    audio_file : the audio file to be processed
    **kwargs: plug in whichever librosa processing is required

    Returns
    -------
    An array of values for each metric for each file

    """
    
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    # Alternative mfcc...
    mfcc2 = kwargs.get("mfcc2")
    
    result = np.array([])
    
    with soundfile.SoundFile(audio_file) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
    
    # Try out these parameters when loading the audio file
    # Set duration to 2.5, do not strip silent parts - keep files same length
    if mfcc2:
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast', duration=2.5, sr=22050*2,offset=0.5)
        

    # Remove silence from start and end of each file
    X = librosa.effects.trim(X, top_db=10)
    x = X[0]
    
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))

    # MFCC
    if mfcc:
        mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))
    
    # Chroma
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    
    # Mel Spectrogram
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
    # Contrast
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
        
    # Tonnetz
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
        
    # MFCC 2
    if mfcc2:
        result = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        #result = np.hstack((result, mfcc2))
    
    return result





def create_ravdess_files():
    """
    A specific function for processing the ravdess dataset.
    The ravdess files are fed into the function for extracting audio features.
    The name of each file gives information about the content of the file,
    e.g. gender of actor, emotion being expressed, and this info is also
    retrieved and saved in a json file along with the librosa-created sound features.

    Returns
    -------
    Nothing is returned, but a json file is created

    """

    # All possible emotions and their corresponding file markers in ravdess files
    emotion_dict = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprised"
    }
    
    emotion_intensity_dict = {
        "01": "normal",
        "02": "strong"
        }
    
    statement_dict = {
        "01": "kids",
        "02": "dogs"
        }
    
    repetition_dict = {
        "01": "first",
        "02": "second"
        }
    
    # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotions_audio = []
    data_details = []
    
    # Create a list for each audio file detail
    emotion_list = []
    emotion_intensity_list = []
    statement_list = []
    repetition_list = []
    actor_list = []
    gender_list = []
    length_list = []
        
    # Get the details of each file using the file name
    for file in glob.glob("ravdess_converted_speech_files/*.wav"):   
        
        
        basename = os.path.basename(file)
        
        emotion = emotion_dict[basename.split("-")[2]]
        
        emotion_intensity = emotion_intensity_dict[basename.split("-")[3]]
        
        statement = statement_dict[basename.split("-")[4]]
        
        repetition = repetition_dict[basename.split("-")[5]]
        
        actor_group = re.search("-(\d{2}).wav", basename)
        
        actor = int(actor_group.group(1))
        
        gender = "Male"
        
        if (actor % 2 == 0):
            
            gender = "Female"
            
        with soundfile.SoundFile(file) as sound_file:
            X = sound_file.read(dtype="float32")
            audio_length = len(X)

            
        # Add all info to lists
        emotion_list.append(emotion)
        emotion_intensity_list.append(emotion_intensity)
        statement_list.append(statement)
        repetition_list.append(repetition)
        actor_list.append(actor)
        gender_list.append(gender)
        length_list.append(audio_length)
        
        
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
        
        
        # Add the results to a list
        emotions_audio.append(processed_audio)
        
    # Add each bit of info about the file to my list
    data_details = [emotion_list, emotion_intensity_list, statement_list, repetition_list, actor_list, gender_list, length_list, emotions_audio]    
             

    # Create a dataframe and add in the file details from above
    df = pd.DataFrame(data_details).T
    
    df.columns=('emotion', 'emotion_intensity', 'statement', 'repetition',
                        'actor', 'gender', 'length', 'processed_audio')


    # Save dataframe to json file
    df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_ravdess.json')


#create_ravdess_files()


  ####################################################################




def create_tess_files():
    """
    
    Again due to the different features of tess as opposed to ravdess, it is as easy
    to create this specific function for processing the tess dataset.
    The tess files are fed into the function for extracting audio features.
    The name of each file gives information about the content of the file,
    e.g. emotion being expressed, and this info is also
    retrieved and saved in a json file along with the librosa-created sound features.

    Returns
    -------
    Nothing is returned, but a json file is created

    """
    
    # File name indicates age of actress
    actor_dict = {"OAF" : "Older",
                  "YAF" : "Younger"}
    
    
    # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotions_audio = []
    data_details = []
    
    # Create a list for each audio file detail
    emotion_list = []
    statement_list = []
    actor_list = []
    gender_list = []
    
    # Get the details of each file using the file name
    for file in glob.glob("tess_converted_speech_files/*.wav"):
        
        basename = os.path.basename(file)
        
        actor = actor_dict[basename.split("_")[0]]
        
        statement = basename.split("_")[1]
        
        emotion_group = re.search("_([A-Za-z]+).wav", basename)
        
        emotion = emotion_group.group(1)
        
        # Change 'ps' to 'surprised'
        if emotion == "ps":
            emotion = "surprised"
        
        gender = "Female"
        
        
        # Add each detail to its corresponding list
        actor_list.append(actor)
        statement_list.append(statement)
        emotion_list.append(emotion)
        gender_list.append(gender)
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
        emotions_audio.append(processed_audio)   
        
    # Add each bit of info about the file to my list
    data_details = [actor_list, statement_list, emotion_list, gender_list, emotions_audio]
             
    
    # Create a dataframe and add in the file details from above
    # The .T means that the lists will be added as columns
    df = pd.DataFrame(data_details).T
    
    # Naming the columns
    df.columns=('actor', 'statement', 'emotion',
                         'gender', 'processed_audio')


    # Save dataframe to json file
    df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_tess.json')
    

#create_tess_files()  


 ###################################################################



def create_crema_files():
    """
    A similar function to the others, which will specifically deal with
    the Crema dataset.
    The files are fed into the function for extracting audio features.
    The name of each file gives information about the content of the file,
    e.g. emotion being expressed, and this info is also
    retrieved and saved in a json file along with the librosa-created sound features.

    Returns
    -------
    Nothing is returned, but a json file is created

    """
    
    # All possible emotions and their corresponding file markers in crema files
    emotion_dict = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fear",
        "DIS": "disgust",
    }
    
    emotion_intensity_dict = {
        "LO": "low",
        "MD": "medium",
        "HI": "high",
        "XX": "unspecified"
        }
    
    # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotions_audio = []
    data_details = []
    
    
    # Create a list for each audio file detail
    actor_list = []
    emotion_list = []
    emotion_intensity_list = []
    
    
    # Get the details of each file using the file name
    for file in glob.glob("crema_converted_speech_files/*.wav"):
            
        basename = os.path.basename(file)
            
        # Convert to int to sync better with actor csv file below
        actor = int(basename.split("_")[0])
            
        emotion = emotion_dict[basename.split("_")[2]]
        
        emotion_intensity_group = re.search("_([A-Za-z]+).wav", basename)
        
        try: # Some filenames were incorrectly labelled, catch them here
        
            emotion_intensity = emotion_intensity_dict[emotion_intensity_group.group(1)]

        except:
            
            emotion_intensity = "missing"
        
            
        
        # Add each detail to its corresponding list
        actor_list.append(actor)
        emotion_list.append(emotion)
        emotion_intensity_list.append(emotion_intensity)
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc2=True)
        emotions_audio.append(processed_audio) 
        
        
    # Add each bit of info about the file to my list
    data_details = [actor_list, emotion_list, emotion_intensity_list, emotions_audio]
          
    
    # Create a dataframe and add in the file details from above
    # The .T means that the lists will be added as columns
    df = pd.DataFrame(data_details).T
    
    # Naming the columns
    df.columns=('actor', 'emotion', 'emotion_intensity', 'processed_audio')
    
    
    # Import a csv file containing more info about the actors
    df_actor_details = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\crema_actors.csv')
    
    merged_df = pd.merge(left=df, right=df_actor_details, left_on='actor', right_on='ActorID')
    
    
    # Save dataframe to json file
    merged_df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_crema_216.json')
    
# for file in glob.glob("crema_speech_files/*.wav"):    
    
#     convert_audio(file, "crema")

#create_crema_files()

 #####################################################################


def create_emo_db_files():
    
    # All possible emotions and their corresponding file markers in crema files
    emotion_dict = {
        "N": "neutral",
        "F": "happy",
        "T": "sad",
        "W": "angry",
        "A": "fear",
        "E": "disgust",
        "L": "boredom"
    }
    
        # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotions_audio = []
    data_details = []
    
    
    # Create a list for each audio file detail
    emotion_list = []
    
    # Get the details of each file using the file name
    for file in glob.glob("emo_db_speech_files/wav/*.wav"):
            
        basename = os.path.basename(file)
        
        emotion = emotion_dict[basename[5]]
        
        emotion_list.append(emotion)
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
        emotions_audio.append(processed_audio) 
        
        
    # Add each bit of info about the file to my list
    data_details = [emotion_list, emotions_audio]
          
    
    # Create a dataframe and add in the file details from above
    # The .T means that the lists will be added as columns
    df = pd.DataFrame(data_details).T
    
    # Naming the columns
    df.columns=('emotion', 'processed_audio')
    

    # Save dataframe to json file
    df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_emo_db.json')
    

# Emo-db are already in mono and at 16k Hz...
#create_emo_db_files()



def create_savee_files():
    
    # All possible emotions and their corresponding file markers in crema files
    emotion_dict = {
        "n": "neutral",
        "h": "happy",
        "sa": "sad",
        "a": "angry",
        "f": "fear",
        "d": "disgust",
        "su": "surprised"
    }
    
    # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotion_list = []
    emotions_audio = []
    data_details = []
    
    # Get the details of each file using the file name
    for file in glob.glob("savee_speech/*.wav"):
        
        basename = os.path.basename(file)      
        
        emotion_group = re.search("_([a-z]+)[0-9]", basename)
        
        emotion = emotion_dict[emotion_group.group(1)]
        
        emotion_list.append(emotion)
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
        emotions_audio.append(processed_audio) 
        
        
    # Add each bit of info about the file to my list
    data_details = [emotion_list, emotions_audio]
    
    # Create a dataframe and add in the file details from above
    # The .T means that the lists will be added as columns
    df = pd.DataFrame(data_details).T
    
    # Naming the columns
    df.columns=('emotion', 'processed_audio')
    

    # Save dataframe to json file
    df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_savee.json')
    
    
#create_savee_files()


def create_x4ntho55_files():
    
    # Empty lists...one for containing the processed audio and another
    # for containing the details about each file
    emotion_list = []
    emotions_audio = []
    data_details = []
    
    # Get the details of each file using the file name
    for file in glob.glob("train-custom/*.wav"):
        
        basename = os.path.basename(file)      
        
        emotion_group = re.search("_([A-Za-z]+).wav", basename)
        
        emotion = emotion_group.group(1)
        
        emotion_list.append(emotion)
        
        # Process each wav file using librosa
        processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
        emotions_audio.append(processed_audio) 
        
        
    # Add each bit of info about the file to my list
    data_details = [emotion_list, emotions_audio]
    
    # Create a dataframe and add in the file details from above
    # The .T means that the lists will be added as columns
    df = pd.DataFrame(data_details).T
    
    # Naming the columns
    df.columns=('emotion', 'processed_audio')
    

    # Save dataframe to json file
    df.to_json('C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_train_custom.json')
    
    
#create_x4ntho55_files()


# Process some test audio

def test_my_audio(audio_file):
    
    
    # Create a list for my emotion labels and processed audio
    emotion_list = []
    emotions_audio = []
    
    # Move processed audio files to a different directory
    for file in glob.glob(audio_file):
        
        basename = os.path.basename(file)
        # Alter name of file so that we do not process files twice
        file_dest = "processed_" + file
            
        # Give the old and new files their full file names
        working_dir = os.getcwd()
        old_file = os.path.join(working_dir, file)
        new_file = os.path.join(working_dir, file_dest)
        
        
        # Screen out any files already processed - check if 'new' file already exists
        if not os.path.isfile(new_file):
    
            # Extract emotion label from file name
            
            emotion_group = re.search("_([A-Za-z]+).wav", basename)
            
            emotion = emotion_group.group(1)
            
            emotion_list.append(emotion) # Add to list
            
            
            # Process each wav file using librosa
            processed_audio = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
            emotions_audio.append(processed_audio)
            
            
            # Change directory of file just processed
            os.rename(old_file, new_file)
            
          
    
    df = pd.DataFrame(emotions_audio)
    df['emotion'] = emotion_list
    
    # Save dataframe to csv file - easier to append to than a json file
    output_path = 'C:\\Users\Admin\\Desktop\\AIT 2021\\Semester3\\SER_Project\\audio_ready_mine.csv'
    # Use append mode - if it is first time it will create new csv file
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    
    

# # Convert my own recordings
#for file in glob.glob("mine_speech_files/*.wav"):    

#   convert_audio(file, "mine")


# # Run function using my own recordings
#my_processed_audio = "mine_converted_speech_files/*.wav"
#test_my_audio(my_processed_audio)

    