# 🚀 NASA Astronaut Facial Recognition & Emotion Analysis System

A comprehensive real-time facial recognition and emotion analysis system designed for astronaut monitoring during space missions. This system combines advanced computer vision techniques with emotional state detection to provide continuous health and wellness monitoring.

## 🌟 Features

### 🎯 Core Capabilities
- **Real-time Facial Recognition**: Identifies known astronauts using face_recognition library
- **Emotion Analysis**: DeepFace-powered emotional state detection (happy, sad, angry, fear, surprise, disgust, neutral)
- **Eye State Monitoring**: MediaPipe-based eye aspect ratio (EAR) calculation for drowsiness detection
- **Sentiment Tracking**: 10-second interval sentiment analysis with statistical reporting
- **API Integration**: RESTful API data transmission for backend monitoring systems

### 📊 Monitoring States
- **🟢 OPTIMAL**: Happy/cheerful emotional state with open eyes
- **🟡 STRESSED**: Anxious/stressed emotional state requiring attention
- **🔴 CRITICAL**: Sad/angry emotional state or potential fainting (closed eyes)

### 🎮 Interactive Controls
- **'q'** - Exit application
- **'m'** - Toggle face mask overlay
- **'s'** - Toggle sentiment analysis
- **'a'** - Toggle API data transmission
- **'r'** - Reset sentiment analysis
- **'l'** - Toggle facial landmarks (future feature)

## 🛠️ Technical Stack

### Core Libraries
- **OpenCV**: Computer vision and camera handling
- **face_recognition**: Facial recognition and encoding
- **DeepFace**: Emotion analysis and facial expression detection
- **MediaPipe**: Face mesh and eye tracking
- **NumPy**: Numerical computations
- **Requests**: API communication

### System Requirements
- **Python**: 3.8+
- **Operating System**: Windows 10/11 (tested), Linux, macOS
- **Camera**: USB webcam or built-in camera
- **RAM**: 4GB+ recommended
- **GPU**: Optional (CUDA support for faster processing)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/nasa-astronaut-recognition.git
cd nasa-astronaut-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements_reconocimiento.txt
```

### 4. Setup Known Faces
1. Create a `known_faces` folder in the project root
2. Add photos of astronauts you want to recognize
3. Name files as: `astronaut_name.jpg` (e.g., `antonio.jpg`)

## 🚀 Usage

### Basic Usage
```bash
python reconocimiento_antonio.py
```

### Advanced Options
```bash
# Specify camera index
python reconocimiento_antonio.py --cam 1

# Set resolution and FPS
python reconocimiento_antonio.py --width 1280 --height 720 --fps 30

# Disable mirror mode
python reconocimiento_antonio.py --no_mirror
```

### Command Line Arguments
- `--cam`: Camera index (default: 0)
- `--width`: Video width (default: 640)
- `--height`: Video height (default: 480)
- `--fps`: Frames per second (default: 30)
- `--no_mirror`: Disable horizontal flip

## 📁 Project Structure

```
nasa-astronaut-recognition/
├── 📁 backup/                    # Backup files and previous versions
├── 📁 fotos/                     # Sample photos and documents
├── 📁 known_faces/              # Astronaut photos for recognition
│   └── antonio.jpg              # Example astronaut photo
├── 📁 venv/                     # Python virtual environment
├── 📄 reconocimiento_antonio.py # Main application
├── 📄 test_camera.py            # Camera testing utility
├── 📄 test_monitoring_integration.py # API integration tests
├── 📄 debug_reconocimiento.py   # Debug utilities
├── 📄 requirements_reconocimiento.txt # Python dependencies
├── 📄 README_MONITORING.md      # Monitoring system documentation
└── 📄 README.md                 # This file
```

## 🔧 Configuration

### API Configuration
The system is configured to send monitoring data to a REST API endpoint:

```python
API_URL = "http://localhost:5000/astronauts/monitoring/data"
```

### Recognition Parameters
```python
# Recognition settings
RECOGNITION_TOLERANCE = 0.5  # Face recognition tolerance
EAR_THRESHOLD_CLOSED = 0.15  # Eye closed threshold
EAR_THRESHOLD_DROWSY = 0.25  # Drowsy threshold
EMOTION_CONFIDENCE_LOW = 0.3 # Minimum emotion confidence
```

## 📊 Data Output

### Real-time Display
- **Main State**: Current emotional/health state (OPTIMAL/STRESSED/CRITICAL)
- **Recognition Info**: Astronaut name and confidence level
- **Technical Data**: EAR value, emotion confidence, FPS
- **Sentiment Analysis**: Real-time counters and 10-second interval analysis

### API Data Structure
```json
{
  "astronautId": "antonio",
  "astronautName": "Antonio",
  "timestamp": "2025-01-05T17:10:45.683106Z",
  "faceDetected": true,
  "faceConfidence": 0.95,
  "eyeOpening": 85,
  "tensionExpressions": 15,
  "pallor": 10,
  "focus": 90,
  "concentration": 85,
  "dominantEmotion": "happy",
  "emotionalState": "OPTIMAL",
  "overallState": "OPTIMAL",
  "alertLevel": "NORMAL",
  "sentimentCounts": {
    "OPTIMO": 45,
    "ESTRESADO": 12,
    "CRITICO": 3
  }
}
```

## 🧪 Testing

### Camera Test
```bash
python test_camera.py
```

### API Integration Test
```bash
python test_monitoring_integration.py
```

### Debug Mode
```bash
python debug_reconocimiento.py
```

## 🔍 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Try different camera indices
python reconocimiento_antonio.py --cam 1
python reconocimiento_antonio.py --cam 2
```

#### Low Performance
- Reduce video resolution: `--width 640 --height 480`
- Lower FPS: `--fps 15`
- Close other applications using the camera

#### Recognition Issues
- Ensure photos in `known_faces/` are clear and well-lit
- Use high-quality photos with good lighting
- Check that faces are clearly visible and not at extreme angles

### Error Messages
- **"No se pudo abrir la cámara"**: Camera not accessible, try different camera index
- **"No se encontraron fotos"**: No photos in `known_faces/` folder
- **"No se encontró cara"**: No face detected in uploaded photo

## 🚀 Performance Optimization

### Recommended Settings
- **Resolution**: 640x480 for optimal performance
- **FPS**: 30 for real-time analysis
- **Recognition Frequency**: Every 10 frames
- **Emotion Analysis**: Every 8 frames

### Hardware Recommendations
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 8GB+ for smooth operation
- **Camera**: 720p+ USB webcam
- **GPU**: NVIDIA GTX 1060+ (optional, for faster processing)

## 📈 Future Enhancements

### Planned Features
- [ ] Multi-person recognition
- [ ] Voice emotion analysis
- [ ] Heart rate detection via camera
- [ ] Machine learning model improvements
- [ ] Mobile app integration
- [ ] Cloud deployment support

### Advanced Monitoring
- [ ] Stress level quantification
- [ ] Fatigue detection algorithms
- [ ] Health trend analysis
- [ ] Emergency alert system

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Antonio** - *Initial work* - [YourGitHub](https://github.com/yourusername)
- **NASA Team** - *Project supervision*

## 🙏 Acknowledgments

- **OpenCV** community for computer vision tools
- **MediaPipe** team for face mesh technology
- **DeepFace** developers for emotion analysis
- **NASA** for mission-critical requirements
- **Open source community** for continuous improvements

## 📞 Support

For support and questions:
- 📧 Email: your.email@nasa.gov
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/nasa-astronaut-recognition/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/nasa-astronaut-recognition/wiki)

---

**Made with ❤️ for NASA Space Missions**

*Ensuring astronaut safety and wellness through advanced AI monitoring*
# cosmotec-emotions
