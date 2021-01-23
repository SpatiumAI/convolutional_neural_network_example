import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('train.csv')


train.shape


train.head(3)


test = pd.read_csv('test.csv')


test.shape


test.head(3)


y_train = train['label']


X_train = train.drop(labels='label', axis=1)


y_train.value_counts()


plt.figure(figsize=(16,9))
sns.countplot(y_train, palette='icefire')
plt.show()


img = X_train.iloc[67].values.reshape((28,28))


plt.imshow(img,cmap='gray')
plt.title(train.iloc[67,0])
plt.axis('off')
plt.show()


X_train = X_train / 255.0
test = test / 255.0


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


X_train.shape # -1:sample, 28x28:img_res., 1:color_channel


from keras.utils.np_utils import to_categorical # convert to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


X_train.shape


from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


model = Sequential()
## 1st conv
model.add(Conv2D(filters = 16, kernel_size = (7,7), padding = 'Same',
         activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
## 2nd conv
model.add(Conv2D(filters = 24, kernel_size = (5,5), padding = 'Same',
         activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
## 3nd conv
model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same',
         activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))
# fully connected NN
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))


# Adam is 'adaptive momentum optimizer'
optimizer = RMSprop()


model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


epochs = 10
batch_size = 250


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=30,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.3, # Randomly zoom image 10%
        width_shift_range=0.2,  # randomly shift images horizontally 10%
        height_shift_range=0.2,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


epoch_list = np.arange(1,21)


batch_list = np.arange(50,550,50)


for b in batch_list:
    history = model.fit_generator(datagen.flow(X_train,y_train, batch_size = b),
                                  epochs = epochs, validation_data = (X_val,y_val), 
                                  steps_per_epoch = X_train.shape[0] // b)


# history = model.fit_generator(datagen.flow(X_train,y_train, batch_size = batch_size),
#                              epochs = epochs, validation_data = (X_val,y_val), steps_per_epoch = X_train.shape[0] // batch_size)


plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
y_true = model.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



