from django.shortcuts import render, redirect
from .forms import UploadAudioForm
import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler


model_file_path = 'D:/Afrobeat_project/afrobeat_music_hit_prediction/afrobeat_predict/models/radomF_Mod.pkl'
# Create your views here.
model = joblib.load(filename=model_file_path)
# print(model)
def home_page(request):
    feature_list = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'liveness', 'loudness', 'duration', 'instrumentalness',
                    'key', 'mode', 'speechiness', 'time_signature']
    if request.method == 'POST':
        danceability = float(request.POST.get('danceability'))
        energy = float(request.POST.get('energy'))
        tempo = float(request.POST.get('tempo'))
        valence = float(request.POST.get('valence'))
        acousticness = float(request.POST.get('acousticness'))
        liveness = float(request.POST.get('liveness'))
        loudness = float(request.POST.get('loudness'))
        duration = int(request.POST.get('duration'))
        instrumentalness = float(request.POST.get('instrumentalness'))
        key = int(request.POST.get('key'))
        mode = int(request.POST.get('mode'))
        speechiness = float(request.POST.get('speechiness'))
        time_signature = int(request.POST.get('time_signature'))
        
        features = {
        'danceability': danceability,
        'energy': energy,
        'tempo': tempo,
        'valence': valence,
        'acousticness': acousticness,
        'liveness': liveness,
        'loudness': loudness,
        'duration_ms': duration,
        'instrumentalness': instrumentalness,
        'key': key,
        'mode': mode,
        'speechiness': speechiness,
        'time_signature': time_signature,
        }
        # df = pd.DataFrame(features, index=[0])
        data = []
        for keys, values in features.items():
            data.append(values)
        # print(data)
        # data_ = np.asarray(data).reshape(1, -1)
        # std = StandardScaler()
        # std.fit(data_)
        # scaler_data = std.transform(data_)
        # print(scaler_data[0][0])
        
        data_asarray = np.asarray(data)
        # print(data_asarray.shape)
        reshape_data = data_asarray.reshape(1, -1)
        print(reshape_data)
        preds = model.predict(reshape_data)
        print(preds)
        pred = None
        # pred = dataprocessing(X_test, model)
        if preds == [1]:
            pred = 'This will be a hit!'
        else:
            pred = "This will likely not be a hit!"
        return redirect('result', pred=pred)
    return render(request, 'myapp/home.html', {'feature_list': feature_list})

def upload_audio(request):
    form = UploadAudioForm()
    if request.method == 'POST':
        form = UploadAudioForm(request.POST, request.FILES)
        if form.is_valid():
            print("Form is valid. Processing...")
            title = form.cleaned_data['title']
            audio_file = form.cleaned_data['audio_file']
            
            extracted_features = get_audio_features(audio_file)
            # print(extracted_features)
            prediction = dataprocessing(extracted_features, model)
            if prediction == [0]:
                print("It won't hit")
                prediction = "Your AfroBeat is Likely not going to hit In UK"
            else:
                print("It will hit")
                prediction = "Your AfroBeat Will shake the world of United Kingdom!"
            return render(request, 'myapp/result.html', {'prediction':prediction, 'title':title})   
    return render(request, 'myapp/index.html', {'form': form})


def get_audio_features(audio_file):
    # Load an audio file
    # Replace with the path to your audio file

    # Extract audio features
    y, sr = librosa.load(audio_file)

    # Extract the features you specified
    danceability = librosa.feature.chroma_cens(y=y)
    energy = librosa.feature.rms(y=y)
    tempo = librosa.beat.tempogram(y=y, sr=sr)
    valence = librosa.feature.chroma_cqt(y=y)
    acousticness = librosa.feature.spectral_centroid(y=y, sr=sr)
    liveness = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    loudness = librosa.feature.rms(y=y)
    duration = librosa.get_duration(y=y, sr=sr)
    instrumentalness = librosa.feature.spectral_rolloff(y=y, sr=sr)
    key = librosa.feature.chroma_stft(y=y, sr=sr)
    mode = librosa.feature.spectral_flatness(y=y)
    speechiness = librosa.feature.spectral_contrast(y=y, sr=sr)
    time_signature = librosa.feature.tempogram(y=y, sr=sr)
    # print(len(danceability))
    # Create a dictionary to store the extracted features
    features = {
        'danceability': danceability.mean(),
        'energy': energy.mean(),
        'tempo': tempo.argmax(),
        'valence': valence.mean(),
        'acousticness': acousticness.mean(),
        'liveness': liveness.mean(),
        'loudness': loudness.mean(),
        'duration_ms': duration * 1000,
        'instrumentalness': instrumentalness.mean(),
        'key': key[0].mean(),
        'mode': mode[0].mean(),
        'speechiness': speechiness[0].mean(),
        'time_signature': int(time_signature.mean()),
    }
    # print(features)
    df = pd.DataFrame(features, index=[0])
    # Convert the dictionary to a Pandas DataFrame
    
    return df

def dataprocessing(dataset, model):
    data_asarray = dataset.to_numpy()
    # data = []
    # dataset.values()
    
    # data_asarray = [-0.76388986, -1.48164056,  0.00970622, -1.28755332, -1.25529427,
    #    -0.92752355, -1.46640386,  2.10746685,  1.35767957, -1.30846547,
        # 1.35080544, -0.44918279, -0.02737592]
    # data_asarray = np.asarray(data)
    # print(data_asarray)
    # data_asarray = np.asarray(data_asarray)
    print(data_asarray)
    reshaped_X = data_asarray.reshape(1, -1)
    # print(reshaped_X, type(reshaped_X))
    pred = model.predict(reshaped_X)
    
    return pred

def data_extraction_from_user(request):
    feature_list = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'liveness', 'loudness', 'duration', 'instrumentalness',
                    'key', 'mode', 'speechiness', 'time_signature']
    
    if request.method == 'POST':
        danceability = float(request.POST.get('danceability'))
        energy = float(request.POST.get('energy'))
        tempo = float(request.POST.get('tempo'))
        valence = float(request.POST.get('valence'))
        acousticness = float(request.POST.get('acousticness'))
        liveness = float(request.POST.get('liveness'))
        loudness = float(request.POST.get('loudness'))
        duration = int(request.POST.get('duration'))
        instrumentalness = float(request.POST.get('instrumentalness'))
        key = int(request.POST.get('key'))
        mode = int(request.POST.get('mode'))
        speechiness = float(request.POST.get('speechiness'))
        time_signature = int(request.POST.get('time_signature'))
        
        features = {
        'danceability': danceability,
        'energy': energy,
        'tempo': tempo,
        'valence': valence,
        'acousticness': acousticness,
        'liveness': liveness,
        'loudness': loudness,
        'duration_ms': duration,
        'instrumentalness': instrumentalness,
        'key': key,
        'mode': mode,
        'speechiness': speechiness,
        'time_signature': time_signature,
    }
        # df = pd.DataFrame(features, index=[0])
        data = []
        for keys, values in features.items():
            data.append(values)
        # print(data)
        # data_ = np.asarray(data).reshape(1, -1)
        # std = StandardScaler()
        # std.fit(data_)
        # scaler_data = std.transform(data_)
        # print(scaler_data[0][0])
        
        data_asarray = np.asarray(data)
        # print(data_asarray.shape)
        reshape_data = data_asarray.reshape(1, -1)
        print(reshape_data)
        preds = model.predict(reshape_data)
        print(preds)
        pred = None
        # pred = dataprocessing(X_test, model)
        if preds == [1]:
            pred = 'This will be a hit!'
        else:
            pred = "This will likely not be a hit!"
        return render(request,'myapp/form.html', {'pred':pred})
    return render(request, 'myapp/form.html', {'feature_list': feature_list})


def result(request, pred):
    return render(request, 'myapp/result.html', {'pred':pred})


def audio_result(request, prediction, title):
    return render(request, 'myapp/result.html', {'title':title, 'prediction':prediction})