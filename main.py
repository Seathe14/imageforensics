#import keras
import tensorflow.python.keras.models
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
#from scipy import stats
#from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import random


def convert_to_ela_image(path, quality):
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

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0

def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    return model

def createAndTrainModel():
    X = []  # ELA converted images
    Y = []  # 0 for fake, 1 for real

    count = 0
    path = 'C:/Users/imynn/Downloads/CASIA2/Au'
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            #         count+=1
            #         if count < 1000:
            #             pass
            if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(1)
          #  if len(Y) % 1000 == 0:
           #     print(f'Processing {len(Y)} images')
            #    break
        #if len(Y) % 1000 == 0:
        #    break

    random.shuffle(X)
    X = X[:2100]
    Y = Y[:2100]
    print(len(X), len(Y))

    path = 'C:/Users/imynn/Downloads/CASIA2/Tp'
    count = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            #         count += 1
            #         if count < 1000:
            #             pass
            if filename.endswith('jpg') or filename.endswith('png'):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(0)
            #if len(Y) % 2000 == 0:
             #   print(f'Processing {len(Y)} images')
              #  break
       # if len(Y) % 1700 == 0:
          #  break

    print(len(X), len(Y))

   #for i in range(10):
    #    X, Y = shuffle(X, Y, random_state=i)

    X = np.array(X)
    Y = to_categorical(Y, 2)
    X = X.reshape(-1, 128, 128, 3)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=14)
    X = X.reshape(-1, 1, 1, 1)
    print(len(X_train), len(Y_train))
    print(len(X_val), len(Y_val))
    model = build_model()
    model.summary()

    init_lr = 1e-4
    optimizer = Adam(learning_rate = init_lr, decay = init_lr/50)

    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    batch_size = 32
    epochs = 20

    history = model.fit(
        X_train, Y_train,
       epochs=epochs,
      batch_size = batch_size,
     validation_data=(X_val, Y_val),
        verbose=2)
    model.save("imgTmpModel2")
    return model

imgTmpModel = 'C:\\Users\\imynn\\PycharmProjects\\diplom\\imgTmpModel2'

class_names = ['fake', 'real']
def predict_image(path):
    imageCheck = prepare_image(path)
    imageCheck = imageCheck.reshape(-1, 128, 128, 3)
    y_pred = model.predict(imageCheck)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    print(f'Class : {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

if __name__ == '__main__':
    if os.path.isdir(imgTmpModel):
        print('load')
        model = tensorflow.keras.models.load_model(imgTmpModel)
    else:
        print('train')
        model = createAndTrainModel()
    #model = createAndTrainModel()
    #print("AA")
    #real_image_path = 'G:\\tp1.jpg'
    #Image.open(real_image_path)
    #imageCheck = prepare_image('C:\\Users\\imynn\\Downloads\\CASIA2\\Tp\\Tp_D_CRN_S_N_cha00029_art00013_00501.tif')
    #imageCheck = imageCheck.reshape(-1,128,128,3)
    #y_pred = model.predict(imageCheck)
    #print(y_pred)
    #imageCheck = prepare_image('C:\\Users\\imynn\\Downloads\\CASIA2\\Au\\Au_ani_101899.jpg')
    #imageCheck = imageCheck.reshape(-1,128,128,3)
    #y_pred = model.predict(imageCheck)
    #print(y_pred)
    #ab = convert_to_ela_image(real_image_path, 85)
    #ab.save("abc.jpg", 'JPEG')
    #predict_image('C:\\Users\\imynn\\Downloads\\CASIA2\\Tp\\Tp_D_CNN_M_B_nat10139_nat00097_11948.jpg')
    predict_image('C:\\Users\\imynn\\Downloads\\CASIA2\\Au\\Au_ani_101899.jpg')
    #path = 'C:/Users/imynn/Downloads/CASIA2/Au'
    #correct_r = 0
   # total_r = 0
    #for dirname, _, filenames in os.walk(path):
    #    for filename in filenames:
    #        #         count+=1
    #        #         if count < 1000:
    #        #             pass
    #        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
    #            imageCheck = prepare_image(path + '/' + filename)
    #            imageCheck = imageCheck.reshape(-1, 128, 128, 3)
    #            y_pred = model.predict(imageCheck)
    #            y_pred_class = np.argmax(y_pred, axis=1)[0]
    #            total_r += 1
    #            if y_pred_class == 1:
    #                correct_r +=1
        #  if len(Y) % 1000 == 0:
        #     print(f'Processing {len(Y)} images')
        #    break
        # if len(Y) % 1000 == 0:
        #    break
    #correct += correct_r
    #total += total_r
    #print(f'Total: {total_r}, Correct: {correct_r}, Acc: {correct_r / total_r * 100.0}')
    #print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/