import matplotlib.pyplot as plt
import keras
import numpy as np

data = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.around(x_train, 0)
x_test = np.around(x_test, 0)

model = keras.models.load_model('model.h5')

prediction = model.predict(x_test)

for i in range(100):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + str(y_test[i]))
    plt.title('Prediction: ' + str(np.argmax(prediction[i])))
    if y_test[i] != np.argmax(prediction[i]):
        plt.show()
