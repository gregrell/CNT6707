from flask import Flask, request, jsonify
import ExecuteModel as EM

app = Flask(__name__)



model = None
def load_model():
    global model
    model = "fuck"




@app.route("/predict", methods=['POST'])
def predict():
    echo = request.json
    print(f"{echo}")
    return jsonify(model)


load_model()
app.run()


