# Question 4

# Now let's serve this model as a web service

#     Install Flask and gunicorn (or waitress, if you're on Windows)
#     Write Flask code for serving the model
#     Now score this client using requests:

# url = "YOUR_URL"
# client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
# requests.post(url, json=client).json()

# What's the probability that this client will get a credit card?

#     0.274
#     0.484
#     0.698
#     0.928

import pickle
from flask import Flask 
from flask import request
from flask import jsonify


with open('model1.bin','rb') as f_in:
    model = pickle.load(f_in)


with open('dv.bin','rb') as f_in:
    dv = pickle.load(f_in)

# url = "127.0.0.1"
# client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"} 
# request.post(url, json=client).json()


app = Flask('card')
 
@app.route('/predict', methods=['POST'])
def predict():
    
    client = request.get_json()
    
    X = dv.transform([client])
    ypred_prob = model.predict_proba(X)[0,1]
    card = (ypred_prob >= 0.5)
    
    result = {
        "card probability": float(ypred_prob),
        "card prediction" : bool(card)
    }
    
    print(result)
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=9696, host="0.0.0.0")
    
