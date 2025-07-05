from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# Load model and dataset
with open("neet_college_predictor.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_excel("neet_cutoff.xlsx")
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df = df.dropna(subset=["Rank"])

def encode_category(category: str) -> int:
    categories = {"UNRESERVED": 0, "SC": 1, "ST": 2}
    return categories.get(category.upper(), -1)

def encode_state(state: str) -> int:
    states = {s: i for i, s in enumerate(df["state"].unique())}
    return states.get(state, -1)

class PredictionRequest(BaseModel):
    score: float | None = None
    rank: int | None = None
    category: str
    state: str

app = FastAPI()

@app.post("/predict")
def predict_colleges(req: PredictionRequest):
    if req.score is None and req.rank is None:
        raise HTTPException(status_code=400, detail="Score or Rank required.")

    category_encoded = encode_category(req.category)
    state_encoded = encode_state(req.state)

    if category_encoded == -1 or state_encoded == -1:
        raise HTTPException(status_code=400, detail="Invalid category or state.")

    score = req.score if req.score is not None else 0.0
    rank = req.rank if req.rank is not None else 0

    input_data = np.array([[score, rank, category_encoded, state_encoded]])
    predicted_rank = model.predict(input_data)[0]

    filtered_df = df[
        (df["category"].str.upper() == req.category.upper()) &
        (df["state"] == req.state) &
        (df["Rank"] >= predicted_rank)
    ]

    top_colleges = filtered_df.nsmallest(5, "Rank")
    if top_colleges.empty:
        return {"colleges": [], "message": "No suitable colleges found."}
    return {"colleges": top_colleges["college_name"].tolist()}