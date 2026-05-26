# AI Semester Project - Image Classification

## Student Information
- University: SZABIST University Larkana
- Course: CSC4101 Artificial Intelligence
- Instructor: Abdul Muqtadar

## Project Overview
Image Classification using ResNet18 on Intel Image Classification Dataset.

## Dataset
- Name: Intel Image Classification Dataset
- Classes: buildings, forest, glacier, mountain, sea, street
- Training images: 14,034
- Testing images: 3,000

## Model
- Model: ResNet18 (Pretrained)
- Epochs: 3
- Accuracy: 84.63%
- F1 Score: 0.8457

## Files
- train.py - Training script
- inference.py - Inference script
- app.py - Flask API
- Dockerfile - Docker configuration
- requirements.txt - Dependencies

## How to Run
### Train
python train.py

### Inference
python inference.py

### Flask API
python app.py