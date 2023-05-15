from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

classes = {
    0: 'alarm clock',
    1: 'anvil',
    2: 'bicycle',
    3: 'crown',
    4: 'grapes',
    5: 'octopus',
    6: 'panda',
    7: 'pizza',
    8: 'snowflake',
    9: 'star'
}


# load data from the dataset folder
ac = np.load('dataset/full_numpy_bitmap_alarm clock.npy')
an = np.load('dataset/full_numpy_bitmap_anvil.npy')
bic = np.load('dataset/full_numpy_bitmap_bicycle.npy')
cro = np.load('dataset/full_numpy_bitmap_crown.npy')
gra = np.load('dataset/full_numpy_bitmap_grapes.npy')
octo = np.load('dataset/full_numpy_bitmap_octopus.npy')
pan = np.load('dataset/full_numpy_bitmap_panda.npy')
piz = np.load('dataset/full_numpy_bitmap_pizza.npy')
sno = np.load('dataset/full_numpy_bitmap_snowflake.npy')
sta = np.load('dataset/full_numpy_bitmap_star.npy')


# select 2000 images from each class
ac = ac[:2000]
an = an[:2000]
bic = bic[:2000]
cro = cro[:2000]
gra = gra[:2000]
octo = octo[:2000]
pan = pan[:2000]
piz = piz[:2000]
sno = sno[:2000]
sta = sta[:2000]


# create labels
ac_label = np.full((ac.shape[0], 1), 0)
an_label = np.full((an.shape[0], 1), 1)
bic_label = np.full((bic.shape[0], 1), 2)
cro_label = np.full((cro.shape[0], 1), 3)
gra_label = np.full((gra.shape[0], 1), 4)
octo_label = np.full((octo.shape[0], 1), 5)
pan_label = np.full((pan.shape[0], 1), 6)
piz_label = np.full((piz.shape[0], 1), 7)
sno_label = np.full((sno.shape[0], 1), 8)
sta_label = np.full((sta.shape[0], 1), 9)


# show one sample image from each class with its label
fig, axs = plt.subplots(2, 5)
axs[0, 0].imshow(ac[0].reshape(28, 28))
axs[0, 0].set_title('alarm clock')
axs[0, 1].imshow(an[0].reshape(28, 28))
axs[0, 1].set_title('anvil')
axs[0, 2].imshow(bic[0].reshape(28, 28))
axs[0, 2].set_title('bicycle')
axs[0, 3].imshow(cro[0].reshape(28, 28))
axs[0, 3].set_title('crown')
axs[0, 4].imshow(gra[0].reshape(28, 28))
axs[0, 4].set_title('grapes')
axs[1, 0].imshow(octo[0].reshape(28, 28))
axs[1, 0].set_title('octopus')
axs[1, 1].imshow(pan[0].reshape(28, 28))
axs[1, 1].set_title('panda')
axs[1, 2].imshow(piz[0].reshape(28, 28))
axs[1, 2].set_title('pizza')
axs[1, 3].imshow(sno[0].reshape(28, 28))
axs[1, 3].set_title('snowflake')
axs[1, 4].imshow(sta[0].reshape(28, 28))
axs[1, 4].set_title('star')
plt.show()


# combine all images and labels
X = np.concatenate((ac, an, bic, cro, gra, octo, pan, piz, sno, sta))
y = np.concatenate((ac_label, an_label, bic_label, cro_label, gra_label,
                    octo_label, pan_label, piz_label, sno_label, sta_label))


# shuffle data
X, y = shuffle(X, y)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# normalize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # 1 for grayscale
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# one-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))  # dropout layer to prevent overfitting
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# save model
model.save('quickdraw_model.h5')

# evaluate model
model.evaluate(X_test, y_test)
