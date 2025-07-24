Fire Detection Using AI

This project uses Artificial Intelligence to detect fire from images or live video feeds using deep learning and image processing techniques. It helps in early fire detection for safety and prevention.

Features

* Fire detection in images and real-time video
* Uses Convolutional Neural Networks (CNN)
* Trained on fire and non-fire datasets
* Alerts when fire is detected
* Easy to integrate with security systems

Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Matplotlib

 Project Structure

```
fire-detection-ai/
│
├── model/               # Trained CNN model
├── dataset/             # Fire and non-fire images
├── fire_detection.py    # Main detection script
├── train_model.py       # Script to train the model
├── utils.py             # Helper functions
└── README.md
```

 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/your-username/fire-detection-ai.git
   cd fire-detection-ai
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the detection:

   ```
   python fire_detection.py
   ```

Dataset

The dataset includes labeled images of fire and non-fire scenes, collected from public datasets and manually labeled.

Model Training

To train your own mode

python train_model.py


You can adjust parameters in `train_model.py` for better accuracy.
 Requirements

* Python 3.7+
* OpenCV
* TensorFlow or Keras
* NumPy

Applications

* Smart surveillance
* Forest fire alert systems
* Industrial fire safety



