# Solar Panel Fault Detection Using Deep Learning

A web-based application to detect faults in solar panels from images using a deep learning model. This system classifies various fault types and estimates the efficiency of the panel based on the detected condition.

## Features

* Upload and analyze solar panel images
* Classifies into one of six classes:

  * `Clean`
  * `Snow-Covered`
  * `Dusty`
  * `Bird-drop`
  * `Electrical-damage`
  * `Physical-Damage`
* Displays:

  * Predicted fault class
  * Confidence score
  * Estimated output efficiency
* Lightweight Flask web interface
* Uses MobileNetV2 or custom CNN models

## Model Training

Two training options are provided:

### 1. MobileNetV2 Transfer Learning (`train_model.py`)

* Uses pre-trained MobileNetV2 as feature extractor
* Custom top layers for classification
* Trains with data augmentation and validation split

### 2. Custom CNN Model (`retrain_model.py`)

* Simpler CNN for fast experiments
* Includes Dropout and Conv layers

### Dataset Preparation

```bash
python split_dataset.py
```

Splits data from `Faulty_solar_panel/` into `dataset/train/` and `dataset/test/` using an 80-20 ratio.

### Training

```bash
python train_model.py
# or
python retrain_model.py
```

Trains and saves model as `solar_fault_classifier.h5`.

## Model Evaluation

```bash
python evaluate_model.py
```

* Generates classification report
* Displays confusion matrix

## Flask Web App

### Run the App

```bash
python app.py
```

### How It Works

* Upload an image of a solar panel
* Model predicts the fault class
* Efficiency is mapped using:

```python
efficiency_map = {
  'Clean': 100,
  'Snow-Covered': 69,
  'Dusty': 60,
  'Bird-drop': 40,
  'Electrical-damage': 40,
  'Physical-Damage': 40
}
```

### Folder Structure

```
├── Faulty_solar_panel/        # Original raw image folders
├── dataset/
│   ├── train/
│   └── test/
├── static/uploads/            # Uploaded images during runtime
├── templates/
│   ├── index.html
│   └── result.html
├── app.py                     # Flask app
├── train_model.py             # MobileNetV2 training
├── retrain_model.py           # Simple CNN training
├── evaluate_model.py          # Evaluation script
├── split_dataset.py           # Dataset splitter
└── solar_fault_classifier.h5  # Trained model file
```

## Dependencies

Install via `pip`:

```bash
pip install tensorflow flask pillow scikit-learn matplotlib seaborn
```

## Example Usage

1. Start the Flask app.
2. Open `http://localhost:5000` in your browser.
3. Upload a solar panel image.
4. View:

   * Fault classification
   * Confidence %
   * Estimated efficiency

## Author

* \[S Krishna Kireeti, Sanjay Bhargav]
* &#x20;AI-Powered Image Classification

## License

This project is open-source and available under the MIT License.
