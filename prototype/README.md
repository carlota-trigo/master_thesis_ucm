# Simple Dermatology App Prototype - README

## 🚀 Quick Start

This is a **simple prototype** of your dermatology mobile app. It demonstrates the core functionality in a web-based format that works on mobile devices.

### 📱 What This Prototype Does:
- Upload skin lesion photos
- Get AI predictions (fine-grained + coarse classification)
- Show confidence scores
- Display medical recommendations
- Works on mobile browsers (responsive design)

### 🛠️ Setup (2 minutes):

1. **Install dependencies:**
```bash
cd prototype
pip install -r requirements.txt
```

2. **Optimize your model:**
```bash
python optimize_model.py
```

3. **Run the prototype:**
```bash
streamlit run app.py
```

4. **Open in browser:** http://localhost:8501

### 📱 Mobile Testing:
- Open the same URL on your phone
- The interface is responsive and mobile-friendly
- Test camera upload functionality

### 🎯 Key Features:

#### **Model Optimization:**
- Converts your Keras model to TensorFlow Lite
- Reduces model size by ~75% (float16 quantization)
- Optimized for mobile inference

#### **Simple UI:**
- Upload image → Get prediction → See results
- Shows both fine-grained and coarse classifications
- Displays confidence scores
- Provides basic medical guidance

#### **Mobile-Ready:**
- Responsive design works on phones
- Fast inference (local processing)
- Privacy-focused (no data upload)

### 🔧 How It Works:

1. **Model Conversion:** Your trained model → TensorFlow Lite
2. **Web Interface:** Streamlit app with mobile-responsive design
3. **Local Processing:** All inference happens locally
4. **Simple Results:** Clear, actionable output

### 📊 Model Requirements:
- Your model should have two outputs (fine + coarse classification)
- Model file should be named with "best" in the filename
- Should be saved as .keras format

### 🚀 Next Steps for Full Mobile App:

1. **Native App Development:**
   - Convert to React Native or Flutter
   - Use TensorFlow Lite directly in mobile app
   - Add camera integration

2. **App Store Deployment:**
   - iOS: Core ML conversion
   - Android: TensorFlow Lite integration
   - App store compliance

3. **Production Features:**
   - User accounts
   - History tracking
   - Medical disclaimers
   - Professional integration

### ⚠️ Important Notes:

- **This is a prototype** for demonstration purposes
- **Not for medical use** - always consult healthcare professionals
- **Educational/research purposes** only
- **Privacy-focused** - no data leaves the device

### 🏥 Medical Disclaimer:
This prototype is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

**Ready to test?** Run the setup commands above and start with your best model!



