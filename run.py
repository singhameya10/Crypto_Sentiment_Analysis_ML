from flask import Flask, request, jsonify
from predictors.btc_ltsm import BtcLtsm

app = Flask(__name__)

btc_ltsm = BtcLtsm()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/update_dataset', methods=['GET'])
def update_dataset():
    success = btc_ltsm.update_dataset()
    if success:
        return jsonify({"message": "Dataset updated successfully"})
    else:
        return jsonify({"error": "Failed to update dataset"}), 500

@app.route('/train_model', methods=['GET'])
def train_model():
    btc_ltsm.train()
    return jsonify({"message": "Model trained successfully"})

@app.route('/test_model', methods=['GET'])
def test_model():
    btc_ltsm.load()
    result = btc_ltsm.test_model()
    return jsonify({"prediction": btc_ltsm._prediction, "accuracy" : btc_ltsm._accuracy})

@app.route('/check_model_state', methods=['GET'])
def check_model_state():
    return jsonify({"model_updated": btc_ltsm.model_updated, "model_trained": btc_ltsm.model_trained})

if __name__ == "__main__":
    app.run(debug=True)
