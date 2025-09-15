from flask import Flask, request, jsonify
import services as svcs


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        message = data.get("text", "")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        result = svcs.predict_message(message)

        return jsonify({
            "message": message,
            "prediction": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
