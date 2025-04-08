# 🧠 ECG Anomaly Detection using LSTM AutoEncoder

This project demonstrates how to detect anomalies in ECG time series data using an **LSTM AutoEncoder** model. By reconstructing normal ECG patterns, the model identifies abnormal sequences based on reconstruction error. It’s a robust unsupervised approach suitable for healthcare anomaly detection scenarios.

---

## 📌 Project Highlights

- 🚀 Achieved **97.93% accuracy** on ECG anomaly detection using an LSTM AutoEncoder.
- 📉 Model trained only on **normal ECG sequences**, making it robust to noise and unseen anomalies.
- 📊 Detected anomalies based on **reconstruction error thresholds**.
- 🔍 Visualized detected anomalies alongside ECG waveforms for interpretability.
- 🧪 Built using **TensorFlow**, **Keras**, **NumPy**, and **scikit-learn**.

---


### Model Performance: Training vs Validation Loss

<p align="center">
  <img src="assets/training_vs_validation_loss.jpg" width="70%">
</p>

### Reconstruction Error Distribution

<p align="center">
  <img src="assets/reconstruction_error_distribution.jpg" width="70%">
</p>



## 🧪 Methodology

1. **Data Preparation**
   - Loaded ECG time series data.
   - Normalized using `MinMaxScaler`.
   - Reshaped into sequences for LSTM input.

2. **Model Architecture**
   - LSTM-based AutoEncoder with symmetric encoder-decoder structure.
   - Trained only on normal signals to learn clean reconstruction.

3. **Anomaly Detection Logic**
   - Computed **mean squared error (MSE)** between input and reconstructed sequences.
   - Set a threshold on reconstruction error to detect anomalies.
   - Classified:
     - `0` → Normal
     - `1` → Anomaly

4. **Evaluation**
   - Compared predicted anomalies to true labels.
   - Visualized:
     - ECG signals
     - Reconstructed signals
     - Anomaly locations
     - Error histograms

---

## 📈 Results

| Metric      | Score    |
|-------------|----------|
| Accuracy    | **97.93%** |
| Detection   | Based on reconstruction MSE threshold |
| Classifier  | Unsupervised – AutoEncoder |

---
<img src="assets/training%20vs%20validation%20loss.jpg" width="70%">

---

## 📁 Folder Structure

```bash
ecg-anomaly-detection/
│
├── README.md                            # Project overview and results
├── requirements.txt                     # Dependency list
├── Anomoly_detection_using_lstm.ipynb   # LSTM model for ECG anomaly detection
├── assets/
│   └── ecg_anomaly_example.png          # Example output visualization
```


---

## ⚙️ Setup Instructions

Install the dependencies:

```bash
pip install -r requirements.txt

```
References
ECG5000 Dataset – UCI Repository

LSTM AutoEncoder for Anomaly Detection – Unsupervised Learning


