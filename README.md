### README.md

```markdown
# Faster Whisper Live Transcription Service

This repository provides a live transcription service using the `faster-whisper` model and `FastAPI`. The service utilizes GPU acceleration for efficient real-time audio transcription and is deployable via Docker and Docker Compose. It is designed to automatically restart on system reboot using a systemd service.

## Features

- **Real-time Transcription:** Transcribe audio input in real-time using WebSockets.
- **GPU Acceleration:** Utilize GPU for fast and efficient transcription.
- **Dockerized Deployment:** Easy to deploy using Docker and Docker Compose.
- **Auto-Restart:** Automatically restarts on system reboot using a systemd service.

## Prerequisites

- Python 3.8 or greater
- Docker and Docker Compose
- NVIDIA GPU with CUDA support

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Faster-Whisper-API.git
   cd Faster-Whisper-API
   ```

2. **Install Docker and Docker Compose:**

   Follow the instructions to install Docker [here](https://docs.docker.com/get-docker/) and Docker Compose [here](https://docs.docker.com/compose/install/).

3. **Build and Run the Docker Containers:**

   ```bash
   docker-compose up --build -d
   ```

## Usage

### WebSocket Endpoint

The WebSocket endpoint `/transcribe` accepts audio data and returns the transcription in real-time.

#### Example Client Script

```python
import websockets
import asyncio

async def test_transcription():
    uri = "ws://localhost:8121/transcribe"
    async with websockets.connect(uri) as websocket:
        # Simulate sending audio data here
        audio_data = open('path_to_audio_file.wav', 'rb').read()
        await websocket.send(audio_data)
        transcription = await websocket.recv()
        print(transcription)

asyncio.run(test_transcription())
```

### Systemd Service for Auto-Restart

To ensure the service runs automatically on system reboot, create a systemd service file.

1. **Create the Service File:**

   ```bash
   sudo nano /etc/systemd/system/faster-whisper.service
   ```

   Add the following content:

   ```ini
   [Unit]
   Description=Faster Whisper Transcription Service
   After=docker.service
   Requires=docker.service

   [Service]
   Restart=always
   WorkingDirectory=/home/ubuntu/Faster-Whisper-API
   ExecStart=/usr/local/bin/docker-compose up
   ExecStop=/usr/local/bin/docker-compose down
   ExecReload=/usr/local/bin/docker-compose restart
   TimeoutSec=30
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and Start the Service:**

   ```bash
   sudo systemctl enable faster-whisper
   sudo systemctl start faster-whisper
   ```

## Files

- `main.py`: FastAPI server code handling WebSocket connections for real-time transcription.
- `Dockerfile`: Docker configuration for building the application image.
- `requirements.txt`: Python dependencies.
- `docker-compose.yaml`: Docker Compose configuration for deploying the service.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests for any features, bug fixes, or enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Repository Title

**Faster Whisper Live Transcription Service**

You can create your GitHub repository with this title and include the provided `README.md` to help users understand how to set up and use your live transcription service.
