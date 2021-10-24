from flask import Flask, render_template, request,jsonify,url_for
from joblib import dump, load
import numpy as np

app = Flask(__name__)
file = "Newhousing_model.joblib"
model = load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features =[float(x) for x in request.form.values()]
    final_features =[np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template("index.html",predition_text="House price will be {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(fore=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
