from flask import Flask, request, jsonify # loading in Flask
import pandas as pd # loading pandas for reading csv
from sklearn.externals import joblib # loading in joblib

# creating a Flask application
app = Flask(__name__)

# Load the model
model = joblib.load('./models/model.p')
scaler = joblib.load('./models/scaler.p')

# creating predict url and only allowing post requests.
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # put input into Pandas df
    input = pd.DataFrame([str(data)], columns=['content'])
    print(input)
    df = pd.DataFrame(input.content.str.split().tolist(), columns=['DPD', 'term', 'seasoning', 'finance_charge', 'principal_ptd', 'down_%','LTV', 'WLTV'])
    print(df)
    # Scale data in preparation for prediction
    rescaled_df = scaler.transform(df)
    print(rescaled_df)
    # making predictions
    pred = model.predict(rescaled_df)
    print(pred)
    # returning the predictions as json
    return jsonify(pred[0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)