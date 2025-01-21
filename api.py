from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List, Dict, Any
import uvicorn
from search_engine import SearchEngine
import asyncio
from starlette.responses import JSONResponse
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the search engine instance
search_engine = None
processing_status = {"progress": 0, "status": "", "is_processing": False}

def update_progress(progress: float, status: str):
    """Update the processing status"""
    processing_status["progress"] = progress
    processing_status["status"] = status

@app.get("/status")
async def get_status():
    return processing_status

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global search_engine, processing_status

    if processing_status["is_processing"]:
        return JSONResponse(
            status_code=409,
            content={"error": "Another file is currently being processed"}
        )

    try:
        processing_status["is_processing"] = True
        processing_status["progress"] = 0
        processing_status["status"] = "Starting file processing..."

        # Process the file in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = ""
        data = []
        total_size = 0
        processed_size = 0

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break

            # Update total size on first chunk
            if total_size == 0:
                try:
                    total_size = file.size
                except:
                    total_size = len(chunk)  # Fallback if size unknown

            chunk_str = chunk.decode('utf-8')
            buffer += chunk_str
            processed_size += len(chunk)

            # Process complete lines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue

                # Update progress
                progress = min(processed_size / total_size, 0.99)
                processing_status["progress"] = progress
                processing_status["status"] = f"Processing file: {progress:.1%}"

                # Allow other tasks to run
                await asyncio.sleep(0)

        # Process any remaining data in buffer
        if buffer.strip():
            try:
                item = json.loads(buffer)
                data.append(item)
            except json.JSONDecodeError:
                pass

        # Validate data structure
        required_fields = {'concept_id', 'aliases', 'canonical_name'}
        if not all(required_fields.issubset(item.keys()) for item in data):
            processing_status["is_processing"] = False
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file format. Missing required fields."}
            )

        # Create search engine instance
        processing_status["status"] = "Creating search index..."
        search_engine = SearchEngine(
            data,
            progress_callback=lambda p, s: update_progress(p, s)
        )

        processing_status["progress"] = 1.0
        processing_status["status"] = "Complete"
        processing_status["is_processing"] = False

        return {"message": "File processed successfully", "concepts": len(data)}

    except Exception as e:
        processing_status["is_processing"] = False
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)