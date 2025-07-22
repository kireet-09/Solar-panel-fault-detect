import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create uploads folder inside static for serving files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('solar_fault_classifier.h5')

# Use the exact classes you trained on (case-sensitive)
classes = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

efficiency_map = {
    'Clean': 100,
    'Snow-Covered': 69,
    'Dusty': 60,
    'Bird-drop': 40,
    'Electrical-damage': 40,
    'Physical-Damage': 40,
}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            # Save uploaded image securely
            filename = secure_filename(img_file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(save_path)

            # Open, resize, and preprocess image
            img = Image.open(save_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img).reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            # Predict
            pred = model.predict(img_array)
            predicted_index = np.argmax(pred)
            predicted_class = classes[predicted_index]
            confidence = np.max(pred)
            efficiency = efficiency_map.get(predicted_class, 0)

            
            print("Raw model output:", pred)
            print("Predicted index:", predicted_index)
            print("Predicted class:", predicted_class)
            print(f"Confidence: {confidence*100:.2f}%")

            return render_template('result.html', 
                                   predicted_class=predicted_class,
                                   confidence=round(confidence * 100, 2),
                                   efficiency=efficiency,
                                   filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
