# Real-Time Learner Profiler and a State-Aware Socratic Math Tutor

(An AI-Powered Socratic Tutor for Personalized Math Education) 

A project submitted for the Minor in AI program at IIT Ropar.

1. Project Overview
   This is an end-to-end prototype of an adaptive learning system designed to address the limitations of one-size-fits-all education. The system leverages a dual-AI architecture to create a personalized and interactive mathematics quiz experience for middle school students (Grades 6-8).
The core innovation is the integration of two custom-trained neural networks:
 * A Learner Profiler (GRU Model) that analyzes a student's behavior in real-time to infer their cognitive-affective state (e.g., CONCENTRATING, BORED, GAMING).
 * A Socratic Tutor (Fine-Tuned LLM) that takes the profiler's inference as input and generates intelligent, pedagogical feedback, guiding students to discover knowledge for themselves rather than simply providing answers.
This project demonstrates a complete cycle of data analysis, model training, fine-tuning, and deployment, culminating in a functional web application that serves as a powerful proof-of-concept for the future of AI in personalized education.
2. System Architecture
The application is built on a two-component AI architecture that works in a continuous, real-time feedback loop.
[Student Interaction -> Learner Profiler -> Inferred State -> Socratic Tutor -> Socratic Feedback -> Student Interaction]
Component 1: The Learner Profiler
 * Model: A Gated Recurrent Unit (GRU) sequential model, built with PyTorch.
 * Training Data: The 2017 ASSISTments dataset, containing millions of real-world student interactions.
 * Function: The GRU model processes a sequence of a student's actions (e.g., timeTaken, attemptCount, correct, hintCount) to predict their dominant cognitive-affective state at any given moment. This model serves as the "eyes" of the system, providing the crucial context of the student's level of engagement.
Component 2: The Socratic Tutor
 * Model: The powerful microsoft/Phi-3-mini-4k-instruct Small Language Model (SLM), fine-tuned using the Unsloth library for maximum performance and memory efficiency.
 * Training Data: A custom-generated, high-quality dataset of over 1,200 examples of Socratic dialogue. This dataset was specifically designed to teach the model how to respond to different student states identified by the Learner Profiler, including providing Socratic guidance for incorrect answers and Socratic enrichment for correct answers.
 * Function: This model acts as the "brain" and "voice" of the system. It takes the state inferred by the Learner Profiler as input and generates a context-aware, pedagogically effective response in the Socratic style.
3. Key Features
 * Real-Time Student Profiling: Accurately infers student states like CONCENTRATING, BORED, and GAMING based on their quiz-taking behavior.
 * Adaptive Socratic Feedback: Provides tailored guidance to struggling students and offers enrichment questions to successful students, moving beyond a simple right/wrong feedback model.
 * End-to-End AI Integration: Demonstrates a complete, functional pipeline from raw student interaction to intelligent, generative AI feedback.
 * Interactive Web Interface: A user-friendly quiz application built with Streamlit, featuring user authentication and a real-time dialogue and analysis panel.
4. Technology Stack
 * AI & Machine Learning: Python, PyTorch, Hugging Face (Transformers, PEFT, TRL), Unsloth
 * Web Application: Streamlit, Streamlit Authenticator
 * Data Science: Pandas, NumPy, Scikit-learn
 * Training Environment: Google Colab (for SLM fine-tuning) & Local/Kaggle (for GRU training)
5. Setup and Deployment
This application is designed to be deployed on Streamlit Community Cloud. The trained AI models are hosted on the Hugging Face Hub to manage their large file sizes and ensure the application is lightweight.
Step 1: Clone the Repository
git clone https://github.com/aiotsir/AI_Personalized_Learning.git
cd your-repository-name

Step 2: Set Up Your AI Models on Hugging Face Hub
This application requires two custom-trained models. The training notebooks are included in this repository, but the final, trained model artifacts must be hosted on the Hugging Face Hub.
 * Create a free account on huggingface.co.
 * Create a new, public model repository (e.g., YourUsername/socratic-tutor-models).
 * Upload your trained model files to this repository:
   * learner_profiler_gru.pth
   * scalers.pkl
   * mappings.pkl
   * The entire contents of the socratic_tutor_phi3_unsloth_final folder.
Step 3: Update the Application Code
In the app.py script, find the HF_REPO_ID variable and update it with your Hugging Face repository path:
# In app.py
HF_REPO_ID = "YourUsername/YourModelRepoName"

Step 4: Local Execution (Requires NVIDIA GPU)
To run this application locally, you must have a compatible NVIDIA GPU and the CUDA toolkit installed.
# Install all required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py ( Learner profiler component alone in Web App - this runs without GPU) 

Step 5: Recommended - Deploy on Streamlit Community Cloud
This is the best way to run and share the application.
 * Push your final code (app.py, config.yaml, requirements.txt) to your GitHub repository.
 * Go to share.streamlit.io and link your GitHub account.
 * Deploy a new app from your repository. The necessary unsloth and torch libraries will be installed on the cloud server automatically from your requirements.txt file.
6. Future Work
This project serves as a robust foundation for a complete adaptive learning ecosystem. Future development could include:
 * Full Socratic Dialogue: Expanding the Socratic Tutor from a single-turn response to a multi-turn, conversational dialogue.
 * Advanced Learner Profiling: Integrating more complex student behaviors and tracking mastery of specific skills over time.
 * Multimodal Input: Incorporating facial emotion detection via a privacy-preserving framework like Federated Learning to provide the Learner Profiler with an even richer set of input signals.

This project represents a significant step towards creating AI that acts not just as a tool, but as a genuine partner in the learning process.
