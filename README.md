# Stress Analysis Using Multimodal Evaluation

## Project Overview

Stress Analysis Using Multimodal Evaluation is an AI-based project that detects stress by analyzing multiple types of human expression such as text, audio, and video. Instead of depending on only one input source, this system combines information from all three modalities to produce a more reliable stress prediction.

The system analyzes:

* Text input using NLP-based feature extraction
* Audio input using acoustic speech features
* Video input using facial expression analysis
* Final prediction using weighted late fusion

The final output classifies the user as either:

* Stressed
* Not Stressed

This project is mainly designed to support early stress awareness and can be useful in areas like mental health monitoring, student well-being, workplace wellness, online counseling, and human-computer interaction.

## Problem Statement

Stress is not always expressed through a single form of communication. A person may write normal text but sound stressed in their voice, or they may speak normally while showing stress through facial expressions.

Most existing stress detection systems use only one modality such as text, audio, or video. This can lead to incomplete or inaccurate predictions. To solve this problem, this project combines text, audio, and video-based analysis to detect stress more accurately.

## Objectives

* To build a multimodal stress detection system using text, audio, and video.
* To extract meaningful features from each modality.
* To train individual models for text, audio, and video-based stress detection.
* To combine predictions using weighted late fusion.
* To improve reliability compared to single-modality models.
* To provide a simple final output that can be easily understood by users.

## Features

* Text-based stress detection
* Audio-based stress detection
* Video-based stress detection
* Feature extraction from multiple modalities
* Weighted late fusion for final prediction
* Simple stressed / not stressed output
* Supports real-life multimodal emotional analysis
* Designed with explainability and usability in mind

## Technologies Used

### Programming Language

* Python

### Machine Learning and Deep Learning

* Scikit-learn
* TensorFlow / PyTorch
* CNN / ResNet
* LinearSVC

### Natural Language Processing

* TF-IDF Vectorization
* Text preprocessing
* Tokenization
* Noise removal

### Audio Processing

* Librosa
* MFCC
* RMS Energy
* Zero Crossing Rate

### Computer Vision

* OpenCV
* Frame extraction
* Facial expression analysis

### Fusion Technique

* Weighted Late Fusion

## Datasets Used

### Text Dataset

* Reddit-based stress dataset / Dreaddit dataset

### Audio Dataset

* RAVDESS dataset

### Video Dataset

* Facial expression dataset

## System Architecture

The project follows three separate pipelines.

### 1. Text Pipeline

Text data is cleaned and converted into numerical form using TF-IDF. A machine learning model is trained to identify stress-related language patterns.

### 2. Audio Pipeline

Audio files are processed to extract acoustic features such as MFCC, RMS energy, and Zero Crossing Rate. These features help detect emotional changes in speech.

### 3. Video Pipeline

Video data is divided into frames. Facial expressions are analyzed using deep learning models such as CNN or ResNet to identify stress-related visual cues.

### 4. Fusion Layer

The individual predictions from text, audio, and video models are combined using weighted late fusion. This final fusion improves the reliability of the stress prediction.

## Workflow

```text
Input Data
   |
   |-- Text Input  -> Preprocessing -> TF-IDF -> Text Model
   |
   |-- Audio Input -> Feature Extraction -> MFCC/RMS/ZCR -> Audio Model
   |
   |-- Video Input -> Frame Extraction -> CNN/ResNet -> Video Model
   |
   -> Weighted Late Fusion
   |
Final Prediction: Stressed / Not Stressed
```

## Model Configuration

| Modality   | Technique Used                        |
| ---------- | ------------------------------------- |
| Text       | TF-IDF + LinearSVC                    |
| Audio      | MFCC, RMS, ZCR                        |
| Video      | CNN / ResNet                          |
| Fusion     | Weighted Late Fusion                  |
| Evaluation | Accuracy, Precision, Recall, F1-score |

## Results

The multimodal model performed better than individual single-modality models because it combined different stress indicators from text, speech, and facial expressions.

| Metric    | Value       |
| --------- | ----------- |
| Accuracy  | 0.84 - 0.85 |
| Precision | 0.83 - 0.86 |
| Recall    | 0.82 - 0.85 |
| F1-Score  | 0.83 - 0.85 |

## Project Structure

```text
Stress-Analysis-Using-Multimodal-Evaluation/
│
├── datasets/
│   ├── text/
│   ├── audio/
│   └── video/
│
├── models/
│   ├── text_model.pkl
│   ├── audio_model.pkl
│   └── video_model.h5
│
├── src/
│   ├── text_processing.py
│   ├── audio_processing.py
│   ├── video_processing.py
│   ├── fusion.py
│   └── prediction.py
│
├── app/
│   └── main.py
│
├── requirements.txt
├── README.md
└── FinalReport.pdf
```

## Installation

Clone the repository:

```bash
git clone GitHub: https://github.com/amritanshu009
cd Multi-model-stress-analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python app/main.py
```

Or run individual modules:

```bash
python src/text_processing.py
python src/audio_processing.py
python src/video_processing.py
python src/fusion.py
```

## Sample Output

```text
Text Model Prediction: Stressed
Audio Model Prediction: Stressed
Video Model Prediction: Not Stressed

Final Fused Prediction: Stressed
Confidence Score: 0.85
```

## Applications

* Mental health monitoring
* Student stress detection
* Workplace wellness systems
* Online counseling platforms
* Human-computer interaction
* Telemedicine support systems
* Early stress awareness tools

## Sustainable Development Goal

This project supports the United Nations Sustainable Development Goal 3: Good Health and Well-being.

By detecting stress indicators early, the system can help promote mental health awareness and support timely action before stress becomes more serious.

## Limitations

* The system is not a medical diagnosis tool.
* Prediction accuracy depends on the quality of text, audio, and video input.
* Background noise can affect audio-based prediction.
* Poor lighting or unclear faces can affect video-based prediction.
* The model may need more diverse datasets for real-world deployment.

## Future Enhancements

* Improve model accuracy using larger and more diverse datasets.
* Add real-time webcam and microphone-based stress detection.
* Integrate explainable AI for better prediction transparency.
* Build a full web application with a user-friendly interface.
* Add dashboard support for stress history and analytics.
* Deploy the system using cloud services.
* Improve privacy and security for sensitive user data.

## Disclaimer

This project is created for academic and research purposes. It is not intended to replace professional medical or psychological diagnosis. The system only provides stress prediction based on machine learning models and should be used as a supportive awareness tool.

## License

This project is for academic use. You may modify and use it for learning, research, and educational purposes.
