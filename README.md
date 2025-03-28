# ContextVision: AI-Powered Visual Assistant for Accessibility

🚀 **ContextVision** is an AI-powered **real-time scene understanding assistant** that helps visually impaired individuals interpret their surroundings through live video analysis, speech interaction, and AI-driven insights.

## 🛠 Features
✅ **Real-Time Scene Description** using OpenCV & BLIP-2  
✅ **Voice-Activated Q&A** using Whisper (STT) and GPT-4  
✅ **Text Recognition (OCR)** for reading signs and documents  
✅ **Object Detection & Tracking** using YOLOv8  
✅ **Multimodal AI** for **gesture-based sign language recognition**  

## 🎯 Use Case
- Assists **visually impaired individuals** in understanding their environment
- Provides **AI-powered interactive Q&A** for enhanced accessibility
- Supports **OCR for sign reading** and **gesture recognition** for sign language users

## 🏗 Tech Stack
- **Computer Vision:** OpenCV, BLIP-2
- **Natural Language Processing:** GPT-4, LangChain
- **Speech Processing:** OpenAI Whisper
- **Object Detection:** YOLOv8
- **Frameworks:** Streamlit, Gradio
- **Deployment:** ONNX, TensorRT, Raspberry Pi

## 🚀 How It Works
1. **Live Video Capture:** Uses OpenCV to process real-time frames.
2. **Scene Understanding:** BLIP-2 generates descriptions of the scene.
3. **Interactive Q&A:** Users ask questions via speech, and GPT-4 provides answers.
4. **Real-Time Speech Processing:** Whisper converts voice to text.
5. **OCR & Object Detection:** Detects and reads text in the environment.

## 📌 Installation & Setup
```bash
# Install Dependencies
pip install opencv-python transformers torch torchvision torchaudio sentence-transformers openai langchain gtts onnxruntime peft accelerate bitsandbytes streamlit gradio ultralytics

🚀 Run the Application

python contextvision.py  # For real-time scene description
python app.py  # To launch the Gradio web interface

📽️ Demo
[
](Insert Deployment Video Link)

📌 Roadmap
🔥 Edge Device Optimization (Jetson, Raspberry Pi 5)

🌍 Multilingual Support

🦾 Enhanced Sign Language Recognition
