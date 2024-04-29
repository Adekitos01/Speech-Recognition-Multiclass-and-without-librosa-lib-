import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")


train_audio_path = 'C:/Users/Adekitos/Desktop/audio-dataset/'
samples, sample_rate = librosa.load(train_audio_path +'yes/0a7c2a8d_nohash_0.wav', sr = 16000)
fig = plt.figure(figsize=(14, 8))

#fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + 'C:/Users/Adekitos/Desktop/audiodataset/yes/0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples) #basically plots the object in memory
plt.show() #to show the plot object


#Let us now look at the sampling rate of the audio signals:
ipd.Audio(samples, rate=sample_rate)
print(sample_rate) #show the sample rate


#From the above, we can understand that the sampling rate of the signal is 16,000 Hz.
#Let us re-sample it to 8000 Hz since most of the speech-related frequencies are present at 8000 Hz:
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)

labels = my_list = os.listdir(train_audio_path) #get all the subdirectories as labels
#Now, let’s understand the number of recordings for each voice command:
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

#plot it
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()

#labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

folder = 'C:/Users/Adekitos/Desktop/audio-dataset' #path to dataset labels
labels = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
print (labels) #print the labels (subfolder names)

#Lets take a look at the distribution of the duration of recordings:
#takes too much time, might want to dodge this bullet
duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
plt.hist(np.array(duration_of_recordings)) #plot the histogram


#preprocessing step
#Resampling
#Removing shorter commands of less than 1 second
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)




#Convert the output labels to integer encoded:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


#Now, convert the integer encoded labels
#to a one-hot vector since it is a multi-classification problem:
from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))


#Reshape the 2D array to 3D since the input to the conv1d must be a 3D array:
#But first get the size of the all_wave data...............

all_wave_len = print(len(all_wave))
all_wave_initial = all_wave #for trouble shooting
all_wave = np.array(all_wave).reshape(-1,8000,1) #lists only have len and all_wave is a list i used 8000 instead of all_wave_len to see


#Split into train and validation set
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)




#CREATE THE MODEL 1D
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()


#Define the loss function to be categorical cross-entropy since it is a multi-classification problem:
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Early stopping and model checkpoints are the callbacks to stop training the neural network
#at the right time and to save the best model after every epoch:
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


#Let us train the model on a batch size of 32 and evaluate the performance on the holdout set:
history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
