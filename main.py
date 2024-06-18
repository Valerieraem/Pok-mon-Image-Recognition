import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
# PCA - Linear dimensionality reduction using Singular Value Decomposition 
# of the data to project it to a lower dimensional space.
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
import PIL
import PIL.Image
import os
import cv2
import pathlib

folder_path = os.path.join("C:\\", "Users", "valer", "Documents", "NewProject", "Pokemon Images DB")
testing_path = os.path.join("C:\\", "Users", "valer", "Documents", "NewProject", "Pokemon Dataset")

#training set
loaded_images = []
train_images = []
train_labels = []

#testing set
testing_images = []
test_labels = []

#loading all image paths in the folder and subfolders
for path in pathlib.Path(folder_path).rglob("*.png"):
    image_path = os.path.join(folder_path, path)
    if "_new" not in image_path :
        image = PIL.Image.open(image_path)
        if image is not None:
            # #transform image from bgr to rgb not needed just looking
            # convert = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
            # image = cv2.imwrite(image_path, convert)
        
            # loaded_images.append(image)
            loaded_images.append(np.asarray(image))
            train_labels.append(os.path.splitext(os.path.basename(image_path))[0])

        else:
            print(f"error loading image: {image_path}")


print(f"loaded {len(loaded_images)} images from the folder")
print(f"loaded {len(train_labels)} images titles from the folder")

print('loaded_images shape: ', loaded_images[0].shape)

# for images in loaded_images: 
#     for i in range(len(loaded_images)):
#         train_images.append(loaded_images[i].reshape((80,80)))

# print('loaded_images shape after resize: ', loaded_images[0].shape)

# plt.imshow(loaded_images[0])
# plt.show()

#plot the first 9 images to check if importing data worked
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(loaded_images[i], cmap = plt.get_cmap('gray'))
    print(train_labels[i])
plt.show()   

for path in pathlib.Path(testing_path).rglob("*.png"):
    image_path = os.path.join(testing_path, path)
    image = PIL.Image.open(image_path)
    if image is not None:
        # #transform image from bgr to rgb not needed just looking
        # convert = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        # image = cv2.imwrite(image_path, convert)
        
        testing_images.append(np.asarray(image))
        photo_title = os.path.splitext(os.path.basename(image_path))[0]
        test_labels.append(photo_title.split("_")[0])
    else:
        print(f"error loading image: {image_path}")

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(testing_images[i], cmap = plt.get_cmap('gray'))
    print(test_labels[i])
plt.show()   


train_dataset_tf = tf.convert_to_tensor(loaded_images, np.int32)
test_dataset_tf = tf.convert_to_tensor(testing_images, np.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_tf, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_tf, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)
