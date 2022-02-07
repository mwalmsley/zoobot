"""
AlexNet Keras Implementation
BibTeX Citation:
@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
Courtesy https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
"""

# Import necessary components to build LeNet
import tensorflow as tf
from tensorflow.keras import layers, regularizers
# from keras.layers.normalization import layers.BatchNormalization


def alexnet_model(img_shape=(224, 224, 1), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = tf.keras.Sequential()

	# Layer 1
	alexnet.add(layers.Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('relu'))
	alexnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(layers.Conv2D(256, (5, 5), padding='same'))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('relu'))
	alexnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(layers.ZeroPadding2D((1, 1)))
	alexnet.add(layers.Conv2D(512, (3, 3), padding='same'))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('relu'))
	alexnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(layers.ZeroPadding2D((1, 1)))
	alexnet.add(layers.Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('relu'))

	# Layer 5
	# alexnet.add(layers.ZeroPadding2D((1, 1)))
	# alexnet.add(layers.Conv2D(1024, (3, 3), padding='same'))
	# alexnet.add(layers.BatchNormalization())
	# alexnet.add(layers.Activation('relu'))
	# alexnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(layers.Flatten())
	alexnet.add(layers.Dense(3072))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('relu'))
	alexnet.add(layers.Dropout(0.5))

	# Layer 7
	# alexnet.add(layers.Dense(4096))
	# alexnet.add(layers.BatchNormalization())
	# alexnet.add(layers.Activation('relu'))
	# alexnet.add(layers.Dropout(0.5))

	# Layer 8
	alexnet.add(layers.Dense(n_classes))
	alexnet.add(layers.BatchNormalization())
	alexnet.add(layers.Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet
