from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__, template_folder="templates")

# Load the model
model = joblib.load('./Deploying your ML Model/models/model.p')

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text')
        # Make prediction
        df = pd.DataFrame(data.split(';'), columns=['DPD', 'term', 'seasoning', 'finance_charge', 'principal_ptd', 'down_%','LTV', 'WLTV'])
        print(df.head())
        rescaled_df = scaler.transform(df)
        pred = model.predict(rescaled_df)
        print(pred)
        return render_template('index.html', sentiment=pred['airline_sentiment_predictions'][0])
    return render_template('index.html', sentiment='')
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)