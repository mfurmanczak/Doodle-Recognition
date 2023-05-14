from tensorflow.keras.models import load_model
from random import randint
import matplotlib.pyplot as plt
import numpy as np

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


# select random 2000 images from each class in index 2000, *.shape[0]
ac = ac[2000:10000]
an = an[2000:10000]
bic = bic[2000:10000]
cro = cro[2000:10000]
gra = gra[2000:10000]
octo = octo[2000:10000]
pan = pan[2000:10000]
piz = piz[2000:10000]
sno = sno[2000:10000]
sta = sta[2000:10000]
# print(ac.shape, an.shape, bic.shape, cro.shape, gra.shape, octo.shape, pan.shape, piz.shape, sno.shape, sta.shape)


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


X = np.concatenate((ac, an, bic, cro, gra, octo, pan, piz, sno, sta))
y = np.concatenate((ac_label, an_label, bic_label, cro_label, gra_label,
                   octo_label, pan_label, piz_label, sno_label, sta_label))

# reshape the data
X = X.reshape(X.shape[0], 28, 28, 1)
y = y.reshape(y.shape[0], 1)

# normalize the data
X = X.astype('float32')
X /= 255

# load the model
model = load_model('quickdraw_model.h5')

# predict all of the images and calculate accuracy
pred = model.predict(X)
pred = np.argmax(pred, axis=1)
y = y.reshape(y.shape[0])
acc = np.mean(pred == y)
print('Accuracy:', acc)

# create a confusion matrix of predictions against actual values (label the axes as well)
# plot it as a heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y, pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='icefire', cbar=False,
            xticklabels=classes.values(), yticklabels=classes.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# print accuracies of each class
from sklearn.metrics import classification_report
print(classification_report(y, pred, target_names=classes.values())) 
