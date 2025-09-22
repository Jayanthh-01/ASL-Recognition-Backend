# ======================================================================================
# PROFESSIONAL SIGN LANGUAGE INTERPRETATION BACKEND
# Description: This stable backend uses placeholders for all AI functions
#              and includes the /upload-video endpoint.
# ======================================================================================

import os
import shutil
import uuid
import time
import random
import openai
import cv2
from fastapi import FastAPI, UploadFile, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# --- 1. INITIAL CONFIGURATION & SETUP ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. PLACEHOLDER & HELPER FUNCTIONS ---
PLACEHOLDER_SIGNS = ['hello', 'thank you', 'I love you', 'please', 'help', 'yes', 'no', 'goodbye']

def recognize_sign_placeholder():
    """Returns a random sign to simulate a real AI model."""
    return random.choice(PLACEHOLDER_SIGNS)

def generate_sentence(sign_list):
    """Uses OpenAI to form a sentence from a list of signs."""
    if not sign_list:
        return "No signs were detected."
    # ... (generate_sentence logic remains the same)
    sign_string = ", ".join(sign_list)
    prompt = f"Convert the following sign language sequence into a natural English sentence: '{sign_string}'."
    try:
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=60)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"ERROR: Could not connect to OpenAI API. {e}")
        return "There was an issue forming a sentence."

# --- 3. API ENDPOINTS ---

@app.get("/")
def root():
    return {"status": "ok", "message": "SignFlow AI Backend is running."}

@app.websocket("/interpret")
async def interpret(websocket: WebSocket):
    # This endpoint remains the same, using the placeholder for live interpretation
    await websocket.accept()
    sign_sequence = []
    # ... (The rest of the WebSocket logic is unchanged)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                new_sign = recognize_sign_placeholder()
                if not sign_sequence or sign_sequence[-1] != new_sign:
                    sign_sequence.append(new_sign)
                    await websocket.send_json({"type": "interim", "signs": sign_sequence})
            elif "text" in message and message["text"] == '{"action": "complete"}':
                sentence = generate_sentence(sign_sequence)
                await websocket.send_json({"type": "final", "sentence": sentence, "signs": sign_sequence})
                sign_sequence = []
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/upload-video")
async def upload_video(file: UploadFile):
    """
    Handles video file uploads. It reads the video, adds placeholder captions,
    and returns a URL to the new, processed video.
    """
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error processing video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(PROCESSED_DIR, f"processed_{file_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    sign_sequence = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add a new placeholder sign every 30 frames to simulate detection
        if frame_count % 30 == 0:
            new_sign = recognize_sign_placeholder()
            if not sign_sequence or sign_sequence[-1] != new_sign:
                sign_sequence.append(new_sign)
        
        # Draw the current sequence of signs on the frame
        caption = ' '.join(sign_sequence)
        cv2.putText(frame, caption, (20, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    os.remove(video_path) # Clean up original upload

    # Generate a final sentence from the full sequence of signs
    final_sentence = generate_sentence(sign_sequence)

    return JSONResponse(content={
        "processed_video_url": f"/processed/processed_{file_id}.mp4",
        "sentence": final_sentence,
    })

@app.get("/processed/{filename}")
async def get_processed_file(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)
