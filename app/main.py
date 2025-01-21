from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from .face_processor import FaceProcessor
from .face_vector import FaceEmbeddingDB
import tempfile
import os

app = FastAPI()
db_params = {
    'dbname': 'face_recognition',
    'user': 'postgres',
    'password': 'password',
    'host': 'pgvector-db',
    'port': '5432'
}

# Initialize database handler
db_handler = FaceEmbeddingDB(db_params)

# Initialize face processor
face_processor = FaceProcessor("./employee_images",db_handler)

@app.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    """Process uploaded video and detect faces."""
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await video.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    try:
        results = []
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                detected_faces = face_processor.process_video_frame(frame)
                if detected_faces:
                    results.append({
                        "frame": frame_count,
                        "detected_faces": detected_faces
                    })
            
            frame_count += 1
        
        cap.release()
        
        return {
            "total_frames": frame_count,
            "processed_frames": len(results),
            "detections": results
        }
        
    finally:
        # Cleanup temporary file
        os.unlink(temp_video_path)

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.get("/getall")
async def find():
    results = db_handler.retrieve_all_data()
    return results

@app.get("/delete")
async def delete():
    db_handler.delete_tables()
    return {"message": "Table deleted"}