# DemoDay Backend Service (FastAPI)

Autonomous backend engine that powers the DemoDay Launch Studio:
- **Repo Ingest**: Scrapes GitHub README, route trees, and package metadata.
- **Planner**: Pydantic AI & Gemini 2.0 Flash structured storyboard generator.
- **Audio**: fal.ai TTS synthesis with exact duration sync.
- **Browser Automation**: Playwright headless recording with deterministic user flows.
- **Video Assembly**: FFmpeg muxing and clip concatenation into 60s `demo.mp4`.
- **API**: FastAPI with SSE streaming endpoints for live frontend progress updates.

---

## Quickstart (Local)

### 1. Requirements
- Python 3.10+
- FFmpeg installed locally (`brew install ffmpeg` or `apt-get install ffmpeg`)

### 2. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Environment Variables
Create `.env`:
```bash
cp .env.example .env
```
Fill in:
- `GEMINI_API_KEY`: Google AI Studio API key
- `FAL_KEY`: fal.ai API key (from hackathon credits)

### 4. Run Server
```bash
uvicorn app.main:app --reload --port 8000
```
API Documentation will be live at `http://localhost:8000/docs`.

---

## Docker Deployment

Build and run in a self-contained container with all OS dependencies and Chromium pre-installed:

```bash
docker build -t demoday-backend .
docker run -p 8000:8000 --env-file .env demoday-backend
```
