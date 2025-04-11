# app.py

from flask import Flask, request, jsonify, render_template
import os
import wave
import tempfile
import datetime
from faster_whisper import WhisperModel

app = Flask(__name__)

# より高精度なモデルを使用（容量に注意）
model = WhisperModel("medium", compute_type="auto")  # 例: "medium" や "large-v2"

def save_wav(data, path, sample_rate=44100, channels=1):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)

def transcribe_audio(path):
    # 言語を明示的に日本語に
    segments, _ = model.transcribe(path, language="ja")
    results = []
    for segment in segments:
        ts = str(datetime.timedelta(seconds=int(segment.start)))
        results.append(f"[{ts}] {segment.text}")
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        return jsonify({"error": "音声ファイルが見つかりませんでした"}), 400

    file = request.files['audio_data']
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    file.save(path)

    try:
        transcription = transcribe_audio(path)
        return jsonify({"transcription": transcription})
    finally:
        os.remove(path)

if __name__ == '__main__':
    app.run(debug=True)
