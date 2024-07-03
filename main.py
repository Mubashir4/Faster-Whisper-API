from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import asyncio

app = FastAPI()

# Load the model
model = WhisperModel("large-v2", device="cuda", compute_type="float16")

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        segments, info = model.transcribe(data, beam_size=5)
        transcription = "\n".join([segment.text for segment in segments])
        await websocket.send_text(transcription)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8121)
