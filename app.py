from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import keras
import cv2
import os
import numpy as np

app = Flask(__name__)
# Define settings
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = keras.models.load_model('Neumonia.h5')
model.summary()


def change_diemsions(img):
    img = cv2.imread(f'./static/uploads/{img}', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img_final = img.reshape(-1, 150, 150, 1)
    return img_final


def allowed_file(file):
    file = file.split(".")
    if file[1] in ALLOWED_EXTENSIONS:
        return True
    return False


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['uploadfile']
    filename = secure_filename(file.filename)
    if allowed_file(filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = change_diemsions(filename)
        x = model.predict(img)
        print(x)
        return render_template('predict.html',mensaje='Upload correct', predict=x)
    return "The extensions dont have support"


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
