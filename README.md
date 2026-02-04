# 🧠 RNN vs Transformer for Time Series Forecasting

![Image](https://miro.medium.com/1%2AlaH0_xXEkFE0lKJu54gkFQ.png)

![Image](https://images.openai.com/static-rsc-3/WDuHb64OVwEt0Dbfie7hNwtoCGvOKHFlgQCPeYn5XL78wA6oR2HOoPJP_2-FVdzTrlBXi7-DRySEQO2LP6p8F9BpETqN0Dl_i9A27o8jZOM?purpose=fullsize\&v=1)

![Image](https://www.business-science.io/assets/2018-12-04-time-series/time_series_deep_learning.png)

![Image](https://images.prismic.io/encord/5205c474-6bc2-446a-b145-d3582bc2254d_image7.png?auto=compress%2Cformat)

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

### 🔵 RNN Model (LSTM)

![Image](https://miro.medium.com/1%2AlaH0_xXEkFE0lKJu54gkFQ.png)

![Image](https://www.researchgate.net/publication/308837697/figure/fig1/AS%3A588802431143936%401517392847718/Stacked-Deep-LSTM-Network-Architecture.png)

![Image](https://images.openai.com/static-rsc-3/_h7dVj_KSVjB30sj52fGWP3Ir8AlN9xi854X3409-fskndsb9OZ-eqan-4foZt4pFMLvU-OmdQwBa5bhg4RMk1MpPnGe8DZXeR2u27KbFCo?purpose=fullsize\&v=1)

* 2 stacked LSTM layers
* Dense output layer
* Sequential learning of temporal patterns

**Strengths**

* Handles sequential dependencies naturally
* Fewer parameters
* Stable early convergence

---

### 🟡 Transformer Model

![Image](https://i.sstatic.net/5QQmq.gif)

![Image](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/transformer_architecture.svg)

![Image](https://i.sstatic.net/eAKQu.png)

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

## 📉 Results Visualization

![Image](https://developers.google.com/static/machine-learning/crash-course/images/metric-curve-ex01.svg)

![Image](https://www.researchgate.net/publication/335858410/figure/fig3/AS%3A804068096225281%401568716181170/Time-series-plots-of-actual-versus-prediction-values-of-the-best-model-in-both-training.png)

![Image](https://www.researchgate.net/publication/344485949/figure/fig2/AS%3A943478821371906%401601954288398/Heatmap-of-the-self-attention-weight-matrix-Each-row-shows-the-attention-distribution-a.ppm)

The notebook generates:

* Training loss curves
* Actual vs Predicted temperature plots
* Attention heatmap showing where the Transformer focuses

---

## 🧠 Key Insights

| Aspect              | RNN                            | Transformer                  |
| ------------------- | ------------------------------ | ---------------------------- |
| Dependency modeling | Sequential memory              | Direct attention across time |
| Long-term learning  | Limited by vanishing gradients | Better global context        |
| Training speed      | Slower (sequential)            | Faster (parallel)            |
| Parameters          | Fewer                          | More                         |

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
jupyter notebook 2025AA05333_rnn_assignment.ipynb
```

Run all cells to reproduce results.

---

## 🎯 Learning Outcomes

✔ Understanding of LSTM vs Transformer
✔ Implementation of positional encoding
✔ Multi-head attention usage
✔ Time-series preprocessing techniques
✔ Model evaluation and analysis

---
