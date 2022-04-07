from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from zoobot.tensorflow.estimators import define_model

# courtesy https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

def small_cnn(input_shape=(224, 224, 1), output_dim=1):
    # model = Sequential()
    model = define_model.CustomSequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, name='dense64'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(output_dim, name='dense_output'))
    model.add(Activation('sigmoid'))

    return model