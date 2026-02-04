# 🧠 RNN vs Transformer for Time Series Forecasting

This project compares **Recurrent Neural Networks (LSTM)** and **Transformer models** for **time series prediction** using deep learning. The goal is to understand how attention-based architectures differ from recurrent models when forecasting sequential data.

---

## 📌 Problem Statement

We perform **univariate time series forecasting** to predict future temperature values using historical data. The task evaluates:

* Learning temporal dependencies
* Model convergence behavior
* Prediction accuracy
* Architectural differences between RNNs and Transformers

---

## 📂 Dataset

**Dataset:** Jena Climate Dataset
**Source:** TensorFlow Keras Datasets

| Property           | Value                      |
| ------------------ | -------------------------- |
| Total time steps   | ~420,000                   |
| Features used      | 1 (Temperature)            |
| Sequence length    | 24–48 time steps           |
| Prediction horizon | 1 step ahead               |
| Train/Test split   | 90% / 10% (temporal split) |

---

## 🏗 Models Implemented

### 🔵 1. RNN Model (LSTM)

* 2 stacked LSTM layers
* Dense output layer
* Sequential learning of temporal patterns

**Strengths**

* Handles sequential dependencies naturally
* Fewer parameters
* Stable early convergence

---

### 🟡 2. Transformer Model

* Custom **sinusoidal positional encoding**
* Multi-head self-attention (4 heads)
* Feed-forward network
* Residual connections + Layer Normalization

**Strengths**

* Learns long-term dependencies better
* Parallel processing (faster training per epoch)
* Global context modeling via attention

---

## 📊 Evaluation Metrics

| Metric       | Purpose                                 |
| ------------ | --------------------------------------- |
| **MAE**      | Average error magnitude                 |
| **RMSE**     | Penalizes large errors (primary metric) |
| **MAPE**     | Percentage error (unstable near 0°C)    |
| **R² Score** | Goodness of fit                         |

> RMSE is chosen as the primary metric because temperature values cross 0°C, making MAPE unreliable.

---

## 📉 Results Summary

| Metric      | RNN    | Transformer |
| ----------- | ------ | ----------- |
| MAE         | Low    | Lower       |
| RMSE        | Good   | Better      |
| R² Score    | High   | Higher      |
| Convergence | Strong | Stronger    |

Transformer showed improved generalization due to the attention mechanism.

---

## 🧠 Key Insights

| Aspect              | RNN                            | Transformer                  |
| ------------------- | ------------------------------ | ---------------------------- |
| Dependency modeling | Sequential memory              | Direct attention across time |
| Long-term learning  | Limited by vanishing gradients | Better global context        |
| Training speed      | Slower (sequential)            | Faster (parallel)            |
| Parameters          | Fewer                          | More                         |

---

## 📈 Visualizations Included

* Training loss curves
* Actual vs Predicted plots
* Attention weight heatmap

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

Open the notebook and run all cells:

```bash
jupyter notebook 2025AA05333_rnn_assignment.ipynb
```

---

## 🎯 Learning Outcomes

✔ Understanding of LSTM vs Transformer
✔ Implementation of positional encoding
✔ Multi-head attention usage
✔ Time-series preprocessing techniques
✔ Model evaluation and analysis

---

## 📜 License

For academic and educational use.

---
