import keras.callbacks
from keras.models import load_model
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import keras.callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageChops, ImageEnhance
from PyQt6.QtCore import pyqtProperty, QObject, QUrl, pyqtSlot, pyqtSignal
from PyQt6 import QtCore
from PyQt6 import QtGui
from threading import Thread
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
import numpy as np
import os
import random

image_size = (128, 128)

import sys

class SignalHelper(QObject):
    messageSignal = pyqtSignal(str)

class ModelOutputCallback(keras.callbacks.Callback):
    def __init__(self, sigHelper):
        super(ModelOutputCallback, self).__init__()
        self._signalHelper = sigHelper

    def on_epoch_end(self, epoch, logs=None):
        self._signalHelper.messageSignal.emit("Epoch {}, loss: {:5.3f}, accuracy: {:5.3f}, val_loss: {:5.3f}"
                                              ", val_accuracy: {:5.3f}\n".format(epoch + 1, logs["loss"], logs["accuracy"],
                                                                                 logs["val_loss"], logs["val_accuracy"]))

class_names = ['Поддельное', 'Подлинное']
class Predicter(QObject):
    imagePathChanged = pyqtSignal()
    trainingProcessChanged = pyqtSignal()
    imagePredicted = pyqtSignal(str)
    authenticFolderPathChanged = pyqtSignal()
    fakeFolderPathChanged = pyqtSignal()
    modelPreparedChanged = pyqtSignal()
    batchSizeChanged = pyqtSignal()
    epochsNumChanged = pyqtSignal()

    def __init__(self, sigHelper, parent=None):
        super().__init__(parent)
        self._path = ''
        self._model = Sequential()
        self._trainingProcess = False
        self._authenticPath = ''
        self._fakePath = ''
        self._modelPrepared = False
        self._epochs = 30
        self._batch_size = 32
        self._signalHelper = sigHelper

    #epochs property
    @pyqtProperty(int, notify=epochsNumChanged)
    def epochs(self):
        return self._epochs
    @epochs.setter
    def epochs(self, epochs):
        if (self._epochs != epochs):
            self._epochs = epochs
            self.epochsNumChanged.emit()

    #batch size property
    @pyqtProperty(int, notify=batchSizeChanged)
    def batchSize(self):
        return self._batch_size
    @batchSize.setter
    def batchSize(self, size):
        if (self._batch_size != size):
            self._batch_size = size
            self.batchSizeChanged.emit()

    #modelPrepared property
    @pyqtProperty(bool, notify=modelPreparedChanged)
    def modelPrepared(self):
        return self._modelPrepared

    #authenticPath property
    @pyqtProperty(str, notify=authenticFolderPathChanged)
    def authenticPath(self):
        return self._authenticPath
    @authenticPath.setter
    def authenticPath(self, path):
        url = QUrl(path)
        resultPath = path
        if url.isLocalFile():
            resultPath = url.toLocalFile()
        self._authenticPath = resultPath
        self.authenticFolderPathChanged.emit()

    #fakePath property
    @pyqtProperty(str, notify=fakeFolderPathChanged)
    def fakePath(self):
        return self._fakePath
    @fakePath.setter
    def fakePath(self, path):
        url = QUrl(path)
        resultPath = path
        if url.isLocalFile():
            resultPath = url.toLocalFile()
        self._fakePath = resultPath
        self.fakeFolderPathChanged.emit()

    #imageToCheck property
    @pyqtProperty(str, notify=imagePathChanged)
    def imagePath(self):
        return self._path
    @imagePath.setter
    def imagePath(self, path):
        url = QUrl(path)
        resultPath = path
        if url.isLocalFile():
            resultPath = url.toLocalFile()
        self._path = resultPath
        self.imagePathChanged.emit()

    #trainingProcess property
    @pyqtProperty(bool, notify=trainingProcessChanged)
    def trainingProcess(self):
        return self._trainingProcess

    def convert_to_ela_image(self, path, quality):
        temp_filename = 'temp_file.jpg'
        ela_filename = 'temp_ela_file.png'

        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)

        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    image_size = (128, 128)

    def prepare_image(self, image_path):
        return np.array(self.convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(128, 128, 3)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        return model

    def createAndTrainModel(self):
        self._trainingProcess = True
        self.trainingProcessChanged.emit()
        X = []  # ELA converted images
        Y = []  # 0 for fake, 1 for real

        for dirname, _, filenames in os.walk(self._authenticPath):
            for filename in filenames:
                #         count+=1
                #         if count < 1000:
                #             pass
                if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
                    full_path = os.path.join(dirname, filename)
                    X.append(self.prepare_image(full_path))
                    Y.append(1)
            #  if len(Y) % 1000 == 0:
            #     print(f'Processing {len(Y)} images')
            #    break
                #if len(Y) % 2100 == 0:
                #    break

        random.shuffle(X)
        X = X[:2100]
        Y = Y[:2100]
        print(len(X), len(Y))

        for dirname, _, filenames in os.walk(self._fakePath):
            for filename in filenames:
                if filename.endswith('jpg') or filename.endswith('bmp'):
                    full_path = os.path.join(dirname, filename)
                    X.append(self.prepare_image(full_path))
                    Y.append(0)
                #if len(Y) % 2100 == 0:
                #    break

        print(len(X), len(Y))

        # for i in range(10):
        #    X, Y = shuffle(X, Y, random_state=i)

        X = np.array(X)
        Y = to_categorical(Y, 2)
        X = X.reshape(-1, 128, 128, 3)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=14)
        X = X.reshape(-1, 1, 1, 1)
        print(len(X_train), len(Y_train))
        print(len(X_val), len(Y_val))
        model = self.build_model()
        model.summary()

        init_lr = 1e-4
        optimizer = Adam(learning_rate=init_lr, decay=init_lr / 50)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_train, Y_train,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_data=(X_val, Y_val),
            verbose=2,
            callbacks=[ModelOutputCallback(self._signalHelper)],)
        #model.save("imgTmpModel2")
        self._model = model
        self._trainingProcess = False
        self.trainingProcessChanged.emit()
        self._modelPrepared = True
        self.modelPreparedChanged.emit()

    @pyqtSlot(str)
    def loadModel(self, path):
        url = QUrl(path)
        resultPath = path
        if url.isLocalFile():
            resultPath = url.toLocalFile()
        dir_path = os.path.dirname(os.path.realpath(resultPath))
        print(dir_path)
        self._model = load_model(dir_path)
        self._modelPrepared = True
        self.modelPreparedChanged.emit()

    @pyqtSlot(str)
    def saveModel(self, path):
        url = QUrl(path)
        resultPath = path
        if url.isLocalFile():
            resultPath = url.toLocalFile()
        self._model.save(resultPath)

    @pyqtSlot()
    def runTraining(self):
        thread = Thread(target = self.createAndTrainModel)
        thread.start()

    @pyqtSlot(str)
    def predictImage(self, path):
        print(path)
        url = QUrl(path)
        if url.isLocalFile():
            path = url.toLocalFile()
        print(path)
        imageCheck = self.prepare_image(path)
        imageCheck = imageCheck.reshape(-1, 128, 128, 3)
        y_pred = self._model.predict(imageCheck)
        y_pred_class = np.argmax(y_pred, axis=1)[0]
        message = (f'Изображение: {path}\nКласс : {class_names[y_pred_class]}, вероятность: {np.amax(y_pred) * 100:0.2f}%\n\n')
        self.imagePredicted.emit(message)
        print(y_pred)
        print(f'Class : {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

def qt_message_handler(mode, context, message):
     if mode == QtCore.QtMsgType.QtInfoMsg:
         mode = 'Info'
     elif mode == QtCore.QtMsgType.QtWarningMsg:
         mode = 'Warning'
     elif mode == QtCore.QtMsgType.QtCriticalMsg:
         mode = 'critical'
     elif mode == QtCore.QtMsgType.QtFatalMsg:
         mode = 'fatal'
     else:
         mode = 'Debug'
     print("%s: %s (%s:%d, %s)" % (mode, message, context.file, context.line, context.file))

if __name__ == '__main__':
    QtCore.qInstallMessageHandler(qt_message_handler)
    app = QGuiApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('neural-network.png'))
    engine = QQmlApplicationEngine()
    #engine.quit.connect(app.quit)
    sigHelper = SignalHelper()
    predicter = Predicter(sigHelper)
    context = engine.rootContext()
    context.setContextProperty("predicter", predicter)
    context.setContextProperty("signalHelper", sigHelper)
    engine.load('main.qml')
    sys.exit(app.exec())