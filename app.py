from flask import Flask, render_template, request
import requests
from keras.models import load_model
from librosa import load
from librosa.feature import mfcc
import numpy as np

app = Flask(__name__)

dic = {
    0 : "hiphop",
    1 : "rock"
}    

model = load_model("model1.h5")

def predict_class(audio_path):
    wav, sample_rate = load(audio_path)
    m = mfcc(wav, sr = sample_rate, n_mfcc=20)
    mean_scaled_feature = np.mean(m.T, axis = 0)
    l = []
    l.append(mean_scaled_feature)
    l = np.array(l)
    p = model.predict(l)
    p = [np.argmax(i) for i in p]
    return dic[p[0]]

@app.route('/', methods = ["GET"])
def index():
    return render_template("index.html")

@app.route('/submit', methods = ['GET', 'POST'])
def output():
    if request.method == 'POST':
        aud = request.files['my_audio']
        aud_path = "static/"+aud.filename
        aud.save(aud_path)

        p = predict_class(aud_path)
    
    return render_template('index.html', prediction = p)

app.run(debug=True)