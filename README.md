Stress Analysis Using Multimodal Evaluation
Overview

Stress Analysis Using Multimodal Evaluation is an AI-based project that detects stress by analyzing text, audio, and video inputs. The main idea of this project is to improve stress detection accuracy by combining multiple human expression signals instead of depending on only one source of data.

The system analyzes written text, voice features, and facial expressions, then combines the individual predictions using weighted late fusion to generate a final stress prediction.

The final output is:

Stressed
or
Not Stressed

Problem Statement

Stress can be expressed in many different ways. A person may write normal text but show stress through their voice tone or facial expression. Similarly, a person may speak normally but show stress through facial movements. Most existing systems use only one input type, such as text, audio, or video. This can make the prediction incomplete or less reliable.

This project solves that problem by using a multimodal approach. It combines text, audio, and video-based stress detection to provide a more balanced and accurate prediction.

Objectives

The main objectives of this project are:

To detect stress using text, audio, and video inputs.
To extract useful features from each input type.
To train separate models for text, audio, and video.
To combine the predictions using weighted late fusion.
To improve prediction reliability compared to single-modality systems.
To provide a simple and understandable final output.
To support early stress awareness in students, employees, and general users.
Features
Text-based stress detection
Audio-based stress detection
Video-based stress detection
TF-IDF-based text feature extraction
MFCC, RMS, and Zero Crossing Rate-based audio feature extraction
CNN / ResNet-based facial expression analysis
Weighted late fusion for final prediction
Final output as Stressed or Not Stressed
Simple and user-friendly prediction flow
Useful for mental health awareness and stress monitoring
Technologies Used
Programming Language

Python

Machine Learning and Deep Learning

Scikit-learn
TensorFlow / PyTorch
CNN
ResNet
LinearSVC

Natural Language Processing

TF-IDF Vectorization
Text Cleaning
Tokenization
Noise Removal

Audio Processing

Librosa
MFCC
RMS Energy
Zero Crossing Rate

Computer Vision

OpenCV
Frame Extraction
Facial Expression Recognition

Fusion Technique

Weighted Late Fusion

Datasets Used
Text Dataset

Reddit-based stress dataset / Dreaddit dataset

Audio Dataset

RAVDESS dataset

Video Dataset

Facial expression dataset

System Architecture

The system follows three separate pipelines:

1. Text Pipeline

The text input is cleaned and converted into numerical form using TF-IDF vectorization. A machine learning classifier is then used to identify stress-related patterns in the text.

2. Audio Pipeline

The audio input is processed using acoustic feature extraction techniques. Features such as MFCC, RMS energy, and Zero Crossing Rate are extracted to understand changes in tone, pitch, and speech energy.

3. Video Pipeline

The video input is divided into frames. Facial expressions are analyzed using deep learning models such as CNN or ResNet to detect visual stress indicators.

4. Fusion Layer

The predictions from the text, audio, and video models are combined using weighted late fusion. This helps the system make a more reliable final prediction.

Workflow

Input Data
│
├── Text Input
│ └── Text Cleaning
│ └── TF-IDF Vectorization
│ └── Text Model Prediction
│
├── Audio Input
│ └── Audio Feature Extraction
│ └── MFCC, RMS, ZCR
│ └── Audio Model Prediction
│
├── Video Input
│ └── Frame Extraction
│ └── Facial Expression Analysis
│ └── Video Model Prediction
│
└── Weighted Late Fusion
└── Final Prediction
└── Stressed / Not Stressed

Model Configuration
Modality	Technique Used
Text	TF-IDF + LinearSVC
Audio	MFCC, RMS, Zero Crossing Rate
Video	CNN / ResNet
Fusion	Weighted Late Fusion
Evaluation Metrics	Accuracy, Precision, Recall, F1-score
Results

The multimodal model gives better performance compared to individual single-modality models because it combines information from text, audio, and video.

Approximate performance values:

Metric	Value
Accuracy	0.84 - 0.85
Precision	0.83 - 0.86
Recall	0.82 - 0.85
F1-Score	0.83 - 0.85
Sample Output

Text Model Prediction: Stressed
Audio Model Prediction: Stressed
Video Model Prediction: Not Stressed

Final Fused Prediction: Stressed
Confidence Score: 0.85

Project Structure

Stress-Analysis-Using-Multimodal-Evaluation/
│
├── datasets/
│ ├── text/
│ ├── audio/
│ └── video/
│
├── models/
│ ├── text_model.pkl
│ ├── audio_model.pkl
│ └── video_model.h5
│
├── src/
│ ├── text_processing.py
│ ├── audio_processing.py
│ ├── video_processing.py
│ ├── fusion.py
│ └── prediction.py
│
├── app/
│ └── main.py
│
├── requirements.txt
├── README.md
└── FinalReport.pdf

Installation

Clone the repository:

git clone https://github.com/your-username/stress-analysis-using-multimodal-evaluation.git
cd stress-analysis-using-multimodal-evaluation

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

For Windows:

venv\Scripts\activate

For macOS/Linux:

source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt
Usage

Run the main application:

python app/main.py

Or run individual modules:

python src/text_processing.py
python src/audio_processing.py
python src/video_processing.py
python src/fusion.py
Applications

This project can be useful in:

Mental health monitoring
Student stress detection
Workplace wellness systems
Online counseling platforms
Human-computer interaction
Telemedicine support systems
Early stress awareness tools
Sustainable Development Goal

This project supports the United Nations Sustainable Development Goal 3: Good Health and Well-being.

The system helps in early stress awareness by detecting stress indicators from text, audio, and video inputs. It can support mental health monitoring and encourage timely action before stress becomes more serious.

Limitations
This system is not a medical diagnosis tool.
Prediction accuracy depends on the quality of text, audio, and video input.
Noisy audio can affect audio prediction.
Poor lighting or unclear faces can affect video prediction.
The model may need more diverse datasets for real-world deployment.
The system should be used only as a supportive awareness tool.
Future Enhancements
Improve accuracy using larger and more diverse datasets.
Add real-time webcam-based stress detection.
Add real-time microphone-based stress detection.
Add explainable AI for better prediction transparency.
Build a complete web application with a better user interface.
Add dashboard support for stress history and analytics.
Deploy the project on cloud platforms.
Improve privacy and security for sensitive user data.
Disclaimer

This project is created for academic and research purposes only. It is not intended to replace professional medical or psychological diagnosis. The system only provides stress prediction based on machine learning models and should be used as a supportive stress awareness tool.

Authors

Amritanshu Shiwanshi
Nakka Satya Dinesh

Guide

Dr. M. Murali
Professor
Department of Computing Technologies
SRM Institute of Science and Technology

License

This project is for academic and educational use. You may modify and use it for learning, research, and project development purposes.
