import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_max_depth=10_n_estimators=150.bin'

f_in = open(model_file, 'rb')
dv, model = pickle.load(f_in)
f_in.close()

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    Going_to_university = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'Go_to_university': bool(Going_to_university)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
