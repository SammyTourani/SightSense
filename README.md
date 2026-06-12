# SightSense

SightSense is an accessibility prototype that turns a live camera feed into spoken
feedback for visually impaired users. You speak a request, the app captures a frame,
a computer-vision backend processes it, and the result is read back to you.

This is a hackathon / research prototype, not a finished product. It is hardcoded for
a single local-network dev setup and has known limitations (see below).

## How it works

SightSense has two halves that talk over HTTP on a local network:

1. **iOS app (SwiftUI)** — captures camera frames, listens for voice commands using
   on-device speech recognition, sends the request and image to the backend, and reads
   the response back with `AVSpeechSynthesizer` text-to-speech. It has a voice-driven
   onboarding flow (welcome, permissions, tutorial).
2. **Python backend (FastAPI)** — runs the computer-vision pipeline and returns text
   for the app to speak.

The app voice command is matched (via sentence-embedding similarity) to one of three
actions, and the backend runs the matching pipeline:

- **Read the text** — EasyOCR reads text in the frame.
- **Describe what I am viewing** — GPT-4o produces a short scene description.
- **Identify object location** — YOLOv8 object detection + Depth-Anything depth
  estimation + MediaPipe hand tracking compute a direction ("Left", "Up-Right",
  "go forward", "object within reach", ...) to guide a user's hand toward a named object.

## Repository layout

```
.
├── main.py                         # FastAPI server: /speech (intent) and /process-image (pipeline)
├── backendcodeforobjectgrabber.py  # Standalone webcam demo of the hand-to-object finder
├── image-description.py            # Standalone script: capture a frame and describe it with GPT-4o
├── image-to-tts.py                 # Standalone script: OCR a frame and speak it with gTTS
├── requirements.txt                # Python dependencies (unpinned)
├── .env.example                    # Template for the OPENAI_API_KEY env var
└── Sightsense/                     # Xcode project for the SwiftUI iOS app
    └── Sightsense/
        ├── SightsenseApp.swift     # App entry point
        ├── ContentView.swift       # App flow state machine (loading → welcome → permissions → tutorial → home)
        ├── HomeView.swift          # Main screen: voice command + backend request
        ├── CameraManager.swift     # AVCaptureSession frame capture + image upload
        ├── SpeechRecognizer.swift  # On-device speech recognition
        ├── WelcomeView.swift, PermissionsView.swift, TutorialView.swift, LoadingView.swift
        └── Assets.xcassets         # App icon and images
```

The standalone Python scripts (`backendcodeforobjectgrabber.py`, `image-description.py`,
`image-to-tts.py`) were development experiments; `main.py` is the FastAPI server the
iOS app actually talks to.

## Tech stack

- **iOS app:** Swift, SwiftUI, AVFoundation, Speech (on-device recognition),
  AVSpeechSynthesizer (text-to-speech).
- **Backend:** Python, FastAPI, Uvicorn.
- **Computer vision / ML:** Ultralytics YOLOv8 (object detection),
  Depth-Anything-V2 via Hugging Face Transformers (depth estimation),
  MediaPipe Hands (hand tracking), EasyOCR (text reading),
  sentence-transformers (intent matching), OpenAI GPT-4o (scene description).
- **Audio:** gTTS (in the standalone scripts), AVSpeechSynthesizer (in the app).

## Setup

### Backend

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key. Copy the example file and fill it in:
   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=...
   ```
4. Run the server (bind to your machine's LAN IP so the phone can reach it):
   ```bash
   python -m uvicorn main:app --host <your-lan-ip> --port 8000
   ```
   YOLO weights (`yolov8l.pt`) and the Depth-Anything model download automatically on
   first run.

### iOS app

1. Open `Sightsense/Sightsense.xcodeproj` in Xcode.
2. Update the backend URL to point at your machine (see "Known limitations").
3. Build and run on a physical device (camera and microphone are required).

## Known limitations

This is a prototype wired for one specific dev environment. Notably:

- **Hardcoded LAN IPs.** The app and backend talk over a hardcoded local-network
  address (`172.18.179.5` in `HomeView.swift` / `CameraManager.swift`, and
  `172.18.51.126` in `Info.plist`'s ATS exception). To run it elsewhere you must
  edit these by hand — they are not configurable yet.
- **CUDA required.** The depth-estimation pipeline is pinned to `device='cuda'`, so
  the object-location feature needs an NVIDIA GPU. There is no CPU fallback.
- **Model weights are not committed.** The YOLO `.pt` files are large auto-downloaded
  artifacts and are gitignored; ultralytics fetches them on first run.

## License

MIT — see [LICENSE](LICENSE).
