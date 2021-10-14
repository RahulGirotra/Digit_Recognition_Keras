from flask import Flask, render_template, request
import numpy as np
import cv2

from keras.datasets import mnist
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,Flatten
from numpy import loadtxt
from keras.models import load_model


app = Flask(__name__)



# load model
model = load_model('model.h5')
print("+"*50, "Model is loaded")
# summarize model.
model.summary()







@app.route('/')
def get_image():
    return render_template("index.html",data="hey")

@app.route("/prediction", methods=["POST"])
def prediction():
    img=request.files['img']
    img.save("img.jpg")
    image=cv2.imread("img.jpg")
    print("x",image.shape)
    #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("y",image.shape)
    image = cv2.resize(image, (28,28))
    print("z",image.shape)

    #image = np.reshape(image, (28,28,1))
    image=image.reshape(1, 28, 28, 1)
    image=image/255
    print(image)
    pred= model.predict(image)
    pred = np.argmax(pred)

    """
    to_test = X_test[0]
    to_test = to_test.reshape(1, 28, 28, 1)
    predict_x0 = model.predict(to_test)
    classes_x0 = np.argmax(predict_x0, axis=1)
    classes_x0[0]
    """

    return render_template("prediction.html", data=pred)




if __name__ == '__main__':
    app.run(debug=True)