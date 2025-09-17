# The updated FastAPI application code
import os
import shutil
import uuid
import cv2
import numpy as np
import mediapipe as mp
import openai
from gtts import gTTS
from fastapi import FastAPI, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configuration for OpenAI and other services
openai.api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app = FastAPI()

# Add CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def recognize_sign(hand_landmarks):
    """
    Placeholder for sign recognition logic.
    In a real application, this would be a trained machine learning model.
    """
    # This is a placeholder; a real model would process 'hand_landmarks'
    # and return a predicted sign.
    signs = ["Hello", "Thank you", "Please", "I love you", "Help"]
    return np.random.choice(signs)


def generate_sentence(sign_list):
    """
    Uses OpenAI GPT-3.5 to form a coherent sentence from a list of signs.
    """
    try:
        sign_string = ", ".join(sign_list)
        prompt = (
            f"Form a grammatically correct and coherent English sentence "
            f"using these signs: {sign_string}. "
            f"Maintain the order of the signs as much as possible."
        )
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I couldn't form a sentence."


@app.get("/")
def root():
    return {"message": "Backend is ready! Services are live."}


# --- /interpret Endpoint (for live video streams) ---
# This endpoint uses WebSockets for real-time communication.
@app.websocket("/interpret")
async def interpret(websocket: WebSocket):
    await websocket.accept()
    sign_sequence = []
    try:
        while True:
            # Receive frame data from the frontend
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

            if frame is None:
                continue

            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            detected_signs = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    sign = recognize_sign(hand_landmarks)
                    detected_signs.append(sign)

            if detected_signs:
                # Simple logic to add unique signs to the sequence
                for sign in detected_signs:
                    if not sign_sequence or sign_sequence[-1] != sign:
                        sign_sequence.append(sign)
                
                # Generate sentence and TTS audio
                sentence = generate_sentence(sign_sequence)
                tts_audio = gTTS(text=sentence, lang="en")
                audio_filename = f"{uuid.uuid4()}.mp3"
                tts_audio.save(os.path.join(PROCESSED_DIR, audio_filename))

                # Send response back to frontend
                response = {
                    "signs": sign_sequence,
                    "sentence": sentence,
                    "audio_url": f"/processed/{audio_filename}",
                }
                await websocket.send_json(response)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# --- /upload-video Endpoint (for video file uploads) ---
class VideoResponse(BaseModel):
    sentence: str
    processed_video_url: str
    audio_url: str


@app.post("/upload-video", response_model=VideoResponse)
async def upload_video(file: UploadFile):
    if not file.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only MP4, MOV, and AVI are supported."
        )

    # Save the uploaded file
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Video processing
    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video file setup
        output_video_path = os.path.join(PROCESSED_DIR, f"output_{file_id}.mp4")
        temp_video_path = os.path.join(PROCESSED_DIR, f"temp_{file_id}.avi")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
        
        sign_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hand detection on each frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Get landmarks for annotation
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    sign = recognize_sign(hand_landmarks)
                    if not sign_sequence or sign_sequence[-1] != sign:
                        sign_sequence.append(sign)

            # Overlay sign text on the frame
            if sign_sequence:
                cv2.putText(
                    frame,
                    f"Signs: {' '.join(sign_sequence)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            out.write(frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        # Generate final sentence and TTS audio
        final_sentence = generate_sentence(sign_sequence)
        if final_sentence:
            tts_audio = gTTS(text=final_sentence, lang="en")
            audio_filename = f"audio_{file_id}.mp3"
            audio_path = os.path.join(PROCESSED_DIR, audio_filename)
            tts_audio.save(audio_path)
        
        # FFmpeg command to add audio and ensure correct format
        ffmpeg_command = (
            f"ffmpeg -y -i {temp_video_path} -i {audio_path} -c:v copy "
            f"-c:a aac -strict experimental {output_video_path}"
        )
        os.system(ffmpeg_command)
        
        # Cleanup temporary files
        os.remove(temp_file_path)
        os.remove(temp_video_path)
        
        return {
            "sentence": final_sentence,
            "processed_video_url": f"/processed/{os.path.basename(output_video_path)}",
            "audio_url": f"/processed/{audio_filename}"
        }

    except Exception as e:
        print(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during video processing.")
    finally:
        # Final cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Endpoint to serve processed files ---
@app.get("/processed/{filename}")
async def get_processed_file(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)