import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import predictor

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_AUDIO = {"wav"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text        = request.form.get("text", "").strip() or None
    audio_path  = None
    video_path  = None

    audio_file = request.files.get("audio")
    if audio_file and audio_file.filename and _ext(audio_file.filename) in ALLOWED_AUDIO:
        fname      = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
        audio_path = os.path.join(UPLOAD_FOLDER, fname)
        audio_file.save(audio_path)

    video_file = request.files.get("video")
    if video_file and video_file.filename and _ext(video_file.filename) in ALLOWED_VIDEO:
        fname      = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
        video_path = os.path.join(UPLOAD_FOLDER, fname)
        video_file.save(video_path)

    if not text and not audio_path and not video_path:
        return jsonify({"error": "Provide at least one input — text, audio (.wav), or video."}), 400

    result = predictor.predict_fused(text, audio_path, video_path)

    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    return jsonify(result)


if __name__ == "__main__":
    predictor.load_models()
    app.run(debug=True, host="0.0.0.0", port=5000)
