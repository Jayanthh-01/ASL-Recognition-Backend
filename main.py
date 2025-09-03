from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # For frontend connections
import numpy as np

app = FastAPI()

# Add CORS to allow frontend requests (important for testing)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/")
def root():
    return {"message": "Backend is ready! Model training in progress."}

# Placeholder predict function
def fake_predict(input_data):
    signs = ["Hello", "Thank you", "A", "B"]
    predicted_index = np.random.randint(0, len(signs))
    confidence = np.random.uniform(0.7, 0.95)
    return signs[predicted_index], confidence

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    if not contents:
        return {"error": "No file uploaded"}
    
    sign, confidence = fake_predict(None)
    return {"sign": sign, "confidence": confidence}
