#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, log_loss, classification_report



train_data_generator = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rescale = 1.0/255,
    zoom_range = 0.1,  
    rotation_range = 10
    )

train_flow_params = dict(
    directory='/home/aeyesafe/zhipingw/practice/datasets/train',
    class_mode='categorical',
    color_mode="grayscale",
    batch_size=128,
    target_size=(224, 224),
    shuffle=True,
    seed=21,
    save_to_dir='/home/aeyesafe/zhipingw/practice/datasets/aug',
    save_format="jpg",
    interpolation="nearest"
)



train_data_iterator = train_data_generator.flow_from_directory(
    **train_flow_params    
)



valid_data_generator = ImageDataGenerator(rescale=1./255)
valid_iterator = valid_data_generator.flow_from_directory(directory='/home/aeyesafe/zhipingw/practice/datasets/valid',
    color_mode="grayscale",target_size=(224, 224))



sample_batch_input,sample_batch_labels = train_data_iterator.next()
 
print(sample_batch_input.shape,sample_batch_labels.shape)



model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=(224,224,1)))

model.add(tf.keras.layers.Conv2D(32,3,strides=1,padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(tf.keras.layers.Conv2D(64,3,strides=1,padding="same",activation="relu"))
model.add(tf.keras.layers.Conv2D(64,3,strides=1,padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(tf.keras.layers.Conv2D(128,3,strides=1,padding="same",activation="relu"))
model.add(tf.keras.layers.Conv2D(128,3,strides=1,padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(3,activation="relu"))

model.add(tf.keras.layers.Dense(3,activation="softmax"))

#Print model information:
model.summary()



model_check = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min', restore_best_weights=True)

csv_logger = CSVLogger('train_epochs_log.csv', separator=',')


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]



model.fit(
        train_data_iterator,
        steps_per_epoch=train_data_iterator.samples/128,
        epochs=100,
        validation_data=valid_iterator,
        validation_steps=valid_iterator.samples/128,
        callbacks =[model_check,es,csv_logger])

loss, categorical_accuracy, auc = model.evaluate_generator(generator=valid_iterator,
steps=valid_iterator.samples/128)
		
print('loss: ', loss,'\n' 'categorical_accuracy: ', categorical_accuracy,'\n' 'auc: ', auc)	
# In[ ]:




