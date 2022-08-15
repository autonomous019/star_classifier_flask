from flask import Flask, render_template, request
#import jsonify
#import requests
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)
#model = xgb.Booster()
#model.load_model("model.json")

model = pickle.load(open('xgbcl_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():


    return render_template('index.html')
'''
Temperature (K),Luminosity(L/Lo),Radius(R/Ro),Absolute magnitude(Mv),Star type,Star color,Spectral Class
3068,0.0024,0.17,16.12,0,Red,M
'''

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        star_temp = float(request.form['Temperature (K)'])
        star_luminosity = float(request.form['Luminosity(L/Lo)'])
        star_radius = float(request.form['Radius(R/Ro)'])
        star_magnitude = float(request.form['Absolute magnitude(Mv)'])
        print('requested temp')
        print(star_temp)
        print('requested lum')
        print(star_luminosity)
        print('requested rad')
        print(star_radius)
        print('requested mag')
        print(star_magnitude)

        params = np.array([[star_temp, star_luminosity, star_radius, star_magnitude]])
        print("Params")
        print(params)
        #requested_data = xgb.DMatrix(params, target)
        #preds = model.predict(requested_data)
        preds = model.predict(params)
        output = preds
        print("output")
        print(output)

        #output = 1
        if output<0:
            return render_template('index.html',prediction_texts="Unable to classify the type of star, use manual methods.")

        if output==1:
            return render_template('index.html',prediction_text="The Type of star is Type 1")
        elif output==2:
            return render_template('index.html',prediction_text="The Type of star is Type 2")
        elif output==3:
            return render_template('index.html',prediction_text="The Type of star is Type 3")
        elif output==4:
            return render_template('index.html',prediction_text="The Type of star is Type 4")
        elif output==5:
            return render_template('index.html',prediction_text="The Type of star is Type 5")
        elif output==6:
            return render_template('index.html',prediction_text="The Type of star is Type 6")
        elif output==7:
            return render_template('index.html',prediction_text="The Type of star is Type 7")
        else:
            return render_template('index.html',prediction_text="UThe Type of star is Type 0")
    
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)