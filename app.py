import pickle
from flask import Flask,request,app,jsonify
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    """
    For direct api call through request
    """
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)
if __name__=="__main__":
    app.run(debug=True)