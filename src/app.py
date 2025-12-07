from flask import Flask, request,jsonify
from joblib import load
from prometheus_client import Counter,Histogram
app=Flask(__name__)
req_counter=Counter('req_count','api_req')
req_inst=Histogram('req_inst','api_inst')
@app.route('/',methods=['GET'])
def home():
    return "run"
model=load('model.pkl')
@app.route("/predict",methods=['POST'])
def predict():
    req_counter.inc()
    with req_inst.time():
        data=request.get_json()
        feature=data["feature"]
        prediction=model.predict([feature])
        return {"predict":int(prediction[0])}
    

@app.route('/metrics')
def metrics():
  
    return prometheus_client.generate_latest(),200,{"content_type":"text/plain"}

if __name__ =='__main__':
    app.run(host='0.0.0.0',port=5000)