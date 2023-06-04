from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=float(request.form.get('age')),
            gender=request.form.get('gender'),
            height=float(request.form.get('height')),
            weight=float(request.form.get('weight')),
            family_history_with_overweight=request.form.get('family_history_with_overweight'),
            favc=request.form.get('FAVC'),
            fcvc=float(request.form.get('FCVC')),
            ncp=int(request.form.get('ncp')),
            caec=request.form.get('caec'),
            smoke=request.form.get('smoke'),
            CH2O=float(request.form.get('ch2o')),
            scc=request.form.get('scc'),
            faf=float(request.form.get('faf')),
            tue=float(request.form.get('tue')),
            calc=request.form.get('calc'),
            mtrans=request.form.get('mtrans'))
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        print("after prediction")

        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)