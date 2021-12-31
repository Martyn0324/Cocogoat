import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from datasetcreator import DatasetCreator

# Damn Pytorch. I DO NOT want to use DataLoader.

data = DatasetCreator.images('D:/Python/gallery-dl/Ganyu', 100, 100)

plt.imshow(data[0])
plt.show()

DatasetCreator.save_dataset(dataset=data,dataset_name='ganyu100')

data = data.astype('float32') # Necessary. Otherwise the normalization will return only black squares

#data = data/127.5 - 1.0 # Dangerous operation --> Might collapse CPU

for i in range(len(data)):
    data[i] = data[i]/127.5 - 1.0 # This one probably does the same, but in a smoother way.

X_train = data[0:1000] # 10% of a dataset seems reasonable.

classes = ("Classify", "As", "You", "Like")

for i in range(0, 1000): # Plotting the images so we can label them
    X_train[i] = (X_train[i]+1.0)*0.5 # Denormalizing
    plt.imshow(X_train[i])
    plt.title(f"image {i}")
    plt.show()

y_train = [1, 1, 1, 0, 2, 3, 2, 1] # Continue on as you wish.

y_train = np.array(y_train)
y_train = to_categorical(y_train, 4)

print(y_train[0])
print(y_train.shape)

print(len(X_train), len(y_train))

X_test = data[1000:1050]

for i in range(X_test.shape[0]):
    X_test[i] = (X_test[i]+1.0)*0.5
    plt.imshow(X_test[i])
    plt.title(f'image {i}')
    plt.show()
    
y_test = [0, 0, 1, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1, 2,
        1, 0, 2, 1, 0, 1, 1, 0, 2, 1, 0, 1, 1, 1, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 0]

y_test = np.array(y_test)
y_test = to_categorical(y_test, 4)

print(y_test[0])
print(y_test.shape)

model = Sequential()

model.add(Conv2D(6, kernel_size=5, activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(7744, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(84, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=50, epochs=10000, verbose=1)

model.save_weights('image_classifier')

#model.load_weights('image_classifier') # After a good training, simply load the weights.

y_pred = model.predict(X_test, batch_size=10, verbose=1)

# Plotting images + predictions and labels
for i in range(X_test.shape[0]):
    predicted = classes[np.argmax(y_pred[i])]
    real = classes[np.argmax(y_test[i])]
    imagem = (X_test[i]+1)*0.5
    plt.imshow(imagem)
    plt.title(f'Predicted outcome: {predicted}\nReal outcome: {real}')
    plt.show()
