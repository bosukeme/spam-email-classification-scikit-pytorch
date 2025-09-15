from fastapi import FastAPI
from models import TextRequest
import services as svcs


app = FastAPI()


@app.post("/predict")
def predict(request: TextRequest):
    text = request.text

    result = svcs.predict_message(text)

    return {
        "prediction": result,
        "message": text
    }
