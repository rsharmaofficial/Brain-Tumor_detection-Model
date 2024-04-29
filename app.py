import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import webbrowser
from threading import Timer



app = Flask(__name__)

def open_browser():
      webbrowser.open_new("http://127.0.0.1:2000")


model =load_model('Braintumour10epochs.keras')
print('Model loaded.')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Tumor Found And the treatments are : Chemotherapy , Surgery , Tomotherapy and Radio therapy"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None

if __name__ == "__main__":
      Timer(1, open_browser).start()
      app.run(port=2000)
      

if __name__ == '__main__':
    app.run(debug=True)