import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = FastAPI()

class EmailInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Email Classifier API is running"}

@app.post("/predict")
def predict(data: EmailInput):
    vec = vectorizer.transform([data.text])
    result = model.predict(vec)[0]
    return {"prediction": str(result)}