import cv2
import numpy as np
from matplotlib.image import imread
from keras.models import load_model

model = load_model('mnist.h5')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = cv2.resize(img, (28,28))
    #convert rgb to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

image = cv2.imread("test2.png")
print(image.shape)
digit, acc = predict_digit(image)

print("Prediction is '", digit, "' with accuracy of", acc)