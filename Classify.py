import os
import os.path
import random
import numpy as np
import h5py
from PIL import Image
import tensorflow as tf

def data_set(Test_Perc, Val_Perc):
    global count, Set
    Data_Dir = '/home/chris/Documents/repos/Concrete Crack Images'
    Categories = ['Negative', 'Positive']
    for Category in Categories:
        path = os.path.join(Data_Dir,Category)
        class_bin = Categories.index(Category)
        for img in os.listdir(path):
            if count == 100:
                h5py_append(Test_Perc, Val_Perc)
            img_file = Image.open(os.path.join(path, img))
            img_array = np.asarray(img_file)
            Set.append([img_array, class_bin])
#             Size = [img_array.shape]
#             if Size not in Sizes:
#                 Sizes.append(Size)
            img_file.close()
            count += 1
    if len(Set) != 0:
        h5py_append(Test_Perc, Val_Perc)

def h5py_append(Test_Perc, Val_Perc):
    global X_train, y_train, X_val, y_val, X_test, y_test, Set, count
    Test_Length  = int(len(Set) * Test_Perc)
    Val_Length = int(len(Set) * Val_Perc)
    random.shuffle(Set)
    for index in range(Test_Length):
        Set_Test = Set.pop(0)
        X_test.append(Set_Test[0])
        y_test.append(Set_Test[1])
    for index in range(Val_Length):
        Set_Val = Set.pop(0)
        X_val.append(Set_Val[0])
        y_val.append(Set_Val[1])
    Train_Length = len(Set)
    for index in range(Train_Length):
        Set_Train = Set.pop(0)
        X_train.append(Set_Train[0])
        y_train.append(Set_Train[1])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train, X_val, X_test =  X_train / 255.0, X_val / 255.0, X_test / 255.0
    if os.path.isfile('ANN_Dataset.hdf5'):
        print('File found')
        with h5py.File('ANN_Dataset.hdf5', 'a') as hf:
            for i in range(Train_Length):
                hf['X_train'].resize((hf['X_train'].shape[0] + 1), axis = 0)
                hf['X_train'][-X_train.shape[0]:] = X_train
        
                hf['y_train'].resize((hf['y_train'].shape[0] + 1), axis = 0)
                hf['y_train'][-y_train.shape[0]:] = y_train
            
            for i in range(Val_Length):
                hf['X_val'].resize((hf['X_val'].shape[0] + 1), axis = 0)
                hf['X_val'][-X_val.shape[0]:] = X_val
                
                hf['y_val'].resize((hf['y_val'].shape[0] + 1), axis = 0)
                hf['y_val'][-y_val.shape[0]:] = y_val
            
            for i in range(Test_Length):
                hf['X_test'].resize((hf['X_test'].shape[0] + 1), axis = 0)
                hf['X_test'][-X_test.shape[0]:] = X_test
                
                hf['y_test'].resize((hf['y_test'].shape[0] + 1), axis = 0)
                hf['y_test'][-y_test.shape[0]:] = y_test
    else:
        with h5py.File('ANN_Dataset.hdf5', 'w') as hf:
            hf.create_dataset('X_train', data = X_train, maxshape = (None, 277, 277, 3))
            hf.create_dataset('y_train', data = y_train, maxshape = (None,))
            hf.create_dataset('X_val', data = X_val, maxshape = (None, 277, 277, 3))
            hf.create_dataset('y_val', data = y_val, maxshape = (None,))
            hf.create_dataset('X_test', data = X_test, maxshape = (None, 277, 277, 3))
            hf.create_dataset('y_test', data = y_test, maxshape = (None,))
    with h5py.File('ANN_Dataset.hdf5', 'r') as hf:
        print(hf.get('X_train').shape)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    count = 0
        

class generator:
    def __call__(self, feature_set, label_set):
        with h5py.File('ANN_Dataset.hdf5', 'r') as hf:
            for feature, label in zip(hf[feature_set], hf[label_set]):
                yield feature, np.array([label])
                
def data_iter(feature_name, label_name):
    ds = tf.data.Dataset.from_generator(generator(), (tf.float64, tf.int64), args=(feature_name, label_name))
    iterator = iter(ds)
    feature, label = iterator.get_next()
    feature = tf.expand_dims(feature, axis=0)
    return feature, label
    
if __name__ == '__main__':
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    Set = []
#     Sizes = []
    count = 0
#     data_set(0.15, 0.15)
#     print(Sizes)
    X_train, y_train = data_iter('X_train', 'y_train')
    X_val, y_val = data_iter('X_val', 'y_val')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1, 277, 277, 3)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
