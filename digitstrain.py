import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np

tensorboard = TensorBoard(log_dir='./logs')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.around(x_train, 0)
x_test = np.around(x_test, 0)

model = Sequential()

model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), shuffle=True, batch_size=128, callbacks=[tensorboard])
model.save('model.h5')

prediction = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)

for i in range(5, 10):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + str(y_test[i]))
    plt.title('Prediction: ' + str(np.argmax(prediction[i])))
    plt.show()
