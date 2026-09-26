# Gesture Drive Controller

A real-time hand-gesture driving controller that uses a webcam to control PC driving games through a virtual Xbox 360 controller.

The project combines **computer vision, machine learning, gesture recognition, and virtual gamepad control**. It started as a rule-based prototype and was later upgraded to use a custom-trained machine-learning model with temporal smoothing.

## Demo

> Demo video/GIF coming soon.

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hands
   ↓
21 Hand Landmarks
   ↓
63 X/Y/Z Features
   ↓
Landmark Normalization
   ↓
Random Forest Classifier
   ↓
5-Frame Temporal Smoothing
   ↓
Gesture + Steering Logic
   ↓
Virtual Xbox 360 Controller
   ↓
PC Driving Game