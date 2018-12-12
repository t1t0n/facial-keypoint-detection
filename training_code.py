import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten, Dropout, Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FIdLookup = 'data/IdLookupTable.csv'


def load(test=False, cols=None):
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    re = load(test, cols)

    X = re[0].reshape(-1, 96, 96, 1)
    y = re[1]

    return X, y


X, y = load2d()


def CNN(with_dropout=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.1))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(500))
    model.add(Activation('relu'))
    if with_dropout:
        model.add(Dropout(0.1))

    model.add(Dense(500))
    model.add(Activation('relu'))
    if with_dropout:
        model.add(Dropout(0.1))

    model.add(Dense(30))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="mean_squared_error", optimizer=sgd)
    return model



if __name__=='main':
    model = CNN()

    filepath="/home/dato/Projects/facial_keypoints_detection/checkpoints/facial-keypoints-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)

    history = model.fit(X, y, nb_epoch=1000, validation_split=0.2, verbose=True, callbacks=[checkpoint])
