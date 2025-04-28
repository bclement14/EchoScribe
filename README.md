# 🎲 EchoScribe — Turn Your D&D Sessions into Living Stories 🎤 ➔ 📝

> Capture the magic of your tabletop adventures — from chaotic voices to polished, readable narratives.

---

## 🧙‍♂️ What is EchoScribe?

**EchoScribe** is a modular Python pipeline that transforms your **multi-speaker D&D session recordings** into structured, readable stories.

From messy audio files to a clean script, EchoScribe helps you:

- Chunk long multi-speaker audio recordings (one track per speaker),
- Transcribe chunks using WhisperX (via Docker),
- Treat transcription outputs to compute a readable script.

> **Bonus:** Designed to easily extend into **LLM-based summaries** and story generation (coming soon 🚀).

---

## 🎧 Context and Origin

EchoScribe was born from real-world D&D sessions recorded over **Discord** using the **Craig bot**🐻.

- Craig provides **individual speaker tracks** (one audio file per speaker) and optionally a **merged track**.
- Early experiments tried using **merged tracks** + **WhisperX diarization** (speaker separation).
- However, **speaker-specific tracks** produced significantly **better transcription quality**, even with long silences.

As a result, EchoScribe is optimized to work best with **one audio track per speaker**, chunked with silence detection, and transcribed individually for clarity.

> 📣 While diarization can be useful, this project currently favors the more accurate per-speaker track approach.

---

## 📦 Key Features

- 🎧 **Audio Chunking** with silence detection
- ✍️ **Automatic Transcription** with WhisperX (Docker-integrated)
- 🛠️ **Correction and Formatting** of transcription outputs
- 🔀 **Dialogue Merging** into a coherent script
- 🧹 **Low RAM Mode** for lightweight machines
- 🗂️ **Session-based Folder Structure** (no working directory pollution)

---

## 🚀 Quickstart

### 1. Install the pipeline

```bash
git clone https://github.com/yourusername/echoscribe.git
cd echoscribe
pip install -e .
```

Make sure you have **Docker installed** and access to a **GPU** for WhisperX.

---

### 2. Prepare your session folder

Structure:

```
your_session/
└── tracks/    # one audio file per speaker
```

---

### 3. Run the pipeline

```python
from echoscribe.pipeline import run_pipeline

run_pipeline(
    base_path=".", 
    input_audio_folder="tracks",
    whisperx_output_folder="wx_output",
    corrected_json_folder="json_files",
    srt_folder="srt_files",
    final_folder="final_outputs",
    use_low_ram=True,
    run_whisperx=True
)
```

---

## 🧹 Full Pipeline Overview

| Step | Action |
|:-----|:-------|
| 1 | Detect silences and chunk long audio files |
| 2 | Transcribe audio with WhisperX in Docker |
| 3 | Correct and clean JSON transcription |
| 4 | Generate timestamped transcripts |
| 5 | Merge transcripts across chunks |
| 6 | Merge successive dialogue lines |
| 7 | Output a polished final script |

---

## 🐳 WhisperX + Docker Integration

No need to manually run Docker commands — the pipeline handles:

- Container image building
- Input/output volume mounting
- GPU access for faster transcription

Build the Docker image once:

```bash
docker build -t echoscribe-whisperx .
```
If you want to launch the command mannualy with your audio files in "chunked_tracks/": 
```bash
docker docker run --gpus all -it -v "$(pwd):/app" --entrypoint whisperx echoscribe-whisperx --model large-v3 --language fr --output_dir wx_output --output_format json chunked_tracks/*.flac 
```

---

## 📂 Example Folder Structure

```
your_session/
├── tracks/                 
├── chunked_tracks/                  
├── wx_output/                  
├── json_files/                 
├── srt_files                  
└── final_outputs/
    ├── merged_audio.flac       
    ├── cut_points.txt          
    ├── merged_transcript.srt   
    ├── cleaned_transcript.srt  
    └── final_script.txt        
```

---

## 🧪 Future Directions

- 🧐 **LLM-based Summarization** (create rich summaries automatically)
- 🛠 **Alternative models** beyond WhisperX
- 🔥 **Real-time audio chunking and processing**

---

## 🧑‍💻 License

This project is open-source under the **MIT License**.

---

# 📣 Join the Adventure!

If you love D&D storytelling, AI-assisted narration, or just making life easier for your players and DMs — **EchoScribe is for you.**

Contributions are welcome! ⚔️📜
