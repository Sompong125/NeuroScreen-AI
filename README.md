# NeuroScreen AI

**An AI-assisted EEG screening research prototype for ADHD risk assessment.**

## Overview

NeuroScreen AI is a research project exploring the use of artificial intelligence to analyze electroencephalogram (EEG) signals and support preliminary ADHD risk assessment.

The system is intended to help organize analysis results into an understandable format for review by qualified professionals.

## Project Goals

- Process and validate uploaded EEG data.
- Analyze EEG-derived features using a trained AI model.
- Present the analysis results in a user-friendly interface.
- Support expert review rather than replace professional assessment.

## Project Structure

```text
NeuroScreen-AI/
├── app.py
├── src/
│   ├── day10_02_inference_pipeline.py
│   └── ui_components.py
├── tests/
│   ├── day10_02_pipeline_test.py
│   └── day10_03_app_test.py
├── docs/
├── assets/
├── models/
└── .streamlit/
    └── config.toml
```

## Current Status

This repository contains the application code and test files prepared for the Day 10 product development stage.

Model files and research datasets may need to be configured separately before the application can run.

## Important Notice

NeuroScreen AI is a research prototype, not a medical diagnostic tool. Its outputs must not be used as a substitute for diagnosis or professional medical advice.

## Setup

Python dependencies and model configuration must be confirmed before running the application. Setup instructions will be updated after the required packages and model files have been verified.