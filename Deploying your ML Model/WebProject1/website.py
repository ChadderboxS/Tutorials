from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__, template_folder="templates")

# Load the model
model = joblib.load('./models/model.p')
scaler = joblib.load('./models/scaler.p')

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        principal_balance = request.form.get('principal_balance')
        principal_ptd = request.form.get('principal_ptd')
        down = request.form.get('down')
        NADA = request.form.get('NADA')
        finance_charge = request.form.get('finance_charge')
        term = request.form.get('term')
        seasoning = request.form.get('seasoning')
        DPD = request.form.get('DPD')
        text = request.form.get('text')
        data = [principal_balance,principal_ptd,down,NADA,finance_charge,term,seasoning,DPD,text]
        input = pd.DataFrame([data], 
                             columns=['principal_balance', 'principal_ptd', 'down', 'NADA', 'finance_charge','term','seasoning', 'DPD', 'content'])
        input['LTV'] = float(input.principal_balance)/float(input.NADA)
        input['WLTV'] = input.LTV*float(input.principal_balance)
        input['down_%'] = float(input.down)/(float(input.principal_balance)+float(input.principal_ptd))
        print(input)
        df = input[['DPD', 'term', 'seasoning', 'finance_charge', 'principal_ptd', 'down_%','LTV', 'WLTV']]
        # df = pd.DataFrame(input.content.str.split().tolist(), columns=['DPD', 'term', 'seasoning', 'finance_charge', 'principal_ptd', 'down_%','LTV', 'WLTV'])
        # Make prediction
        print(df)
        rescaled_df = scaler.transform(df)
        pred = model.predict(rescaled_df)
        pred = np.round(pred, decimals=2)
        #print(preds)
        #pred = pred*100
        print(pred)
        return render_template('index.html', value=pred[0])
    return render_template('index.html', value='')
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)