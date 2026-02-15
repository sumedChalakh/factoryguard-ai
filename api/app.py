from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    return jsonify({
        "received": data,
        "failure_probability": 0.12
    }), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

