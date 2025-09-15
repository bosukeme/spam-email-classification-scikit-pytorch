import pickle


with open("models/spam_model_scikit.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict_message(message: str):
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return result
