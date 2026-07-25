# 🧠 Weight Initialization Techniques in Neural Networks

A hands-on deep learning demo exploring **why weight initialization matters** and how it affects a neural network's ability to learn — built as part of a deep learning starter/fundamentals series.

---

## 📌 Overview

Weight initialization is one of those small decisions that quietly makes or breaks a neural network. Initialize too small, and signals shrink to nothing as they pass through layers (vanishing gradients). Initialize too large, and they blow up instead. This project walks through that problem hands-on:

- Loading a small, **non-linearly separable** 2D dataset
- Building a 4-hidden-layer neural network in Keras
- Manually computing and injecting custom initial weights using NumPy (instead of relying purely on Keras defaults)
- Training the network and visualizing the resulting decision boundary
- Comparing that against a naive, uniformly tiny initialization scheme to see the difference initialization makes

It's meant as a learning-by-doing exercise rather than a production model — the focus is on the *mechanics* of initialization, not squeezing out maximum accuracy.

---

## 📊 Dataset

**File:** `ushape.csv`

A synthetic 2D binary classification dataset shaped like a "U" — chosen specifically because it **cannot be separated by a straight line**, which forces the network to actually learn a curved, non-linear decision boundary.

| Column | Description |
|--------|-------------|
| `X` | Feature 1 (float) |
| `Y` | Feature 2 (float) |
| `class` | Binary target label (`0` or `1`) |

- 100 samples total, perfectly balanced (50 per class)

---

## 🏗️ Model Architecture

A simple fully-connected (`Sequential`) network built in Keras:

```
Input(2 features)
   → Dense(10, activation='relu')
   → Dense(10, activation='relu')
   → Dense(10, activation='relu')
   → Dense(10, activation='relu')
   → Dense(1,  activation='sigmoid')
```

| Detail | Value |
|--------|-------|
| Total parameters | 371 (all trainable) |
| Loss function | Binary Crossentropy |
| Optimizer | Adam |
| Metric | Accuracy |
| Epochs | 100 |
| Validation split | 20% |

---

## 🧩 What This Notebook Walks Through

1. **Load & visualize the data** — scatter plot of the U-shaped classes to confirm they're not linearly separable.
2. **Build the network** — a 4-hidden-layer ReLU network with a sigmoid output for binary classification.
3. **Inspect default weights** — pull out Keras's randomly-initialized weights with `model.get_weights()`.
4. **Manually re-initialize the weights** — replace them with custom NumPy arrays, scaling each layer's random weights by `1/√(fan-in)` and zeroing the biases, following the same fan-in-aware scaling principle behind initialization schemes like He/Xavier (rather than using one fixed constant for every layer).
5. **Load the custom weights back into the model** via `model.set_weights()` and train from that starting point.
6. **Visualize the learned decision boundary** using `mlxtend.plotting.plot_decision_regions`, to see the curved boundary the network carved out.
7. **Compare against a naive baseline** — a quick check of what happens if weights are instead drawn from `np.random.randn(...) * 0.01`, a fixed tiny constant regardless of layer size, to illustrate why fan-in-aware scaling tends to work better as networks get deeper.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/avrajyoti07/22DL_Weight_Initialization.git
cd 22DL_Weight_Initialization
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib tensorflow mlxtend notebook
```

### 3. Run the notebook
```bash
jupyter notebook 15Initialization.ipynb
```

Make sure `ushape.csv` is in the same directory as the notebook before running the cells.

---

## 🗂️ Project Structure

```
22DL_Weight_Initialization/
├── 15Initialization.ipynb   # Main notebook — build, initialize, train, visualize
├── ushape.csv               # U-shaped synthetic binary classification dataset
└── README.md
```

---

## 🎯 Key Takeaways

- Weight initialization isn't just a formality — it directly affects how fast (and whether) a network learns.
- Scaling initial weights relative to a layer's **fan-in** (number of incoming connections) helps keep signal variance stable as it passes through layers.
- A fixed, uniformly tiny initialization (e.g., multiplying by `0.01` regardless of layer size) can leave deeper layers starved of useful gradient signal.
- Visualizing the decision boundary is a simple, intuitive way to confirm a small network can actually learn non-linear patterns when initialized and trained sensibly.

---

## 🔮 Possible Extensions

- Add side-by-side comparisons with other classic schemes — Zero initialization, plain random (large-scale) initialization, Xavier/Glorot, and true He initialization (`√(2/fan-in)`).
- Plot training/validation loss curves for each scheme on the same axes to compare convergence speed directly.
- Try the same experiment on other non-linear toy datasets (concentric circles, XOR, spirals) to see how well the conclusions generalize.

---

## 🙋 Author

**Avrajyoti** — [@avrajyoti07](https://github.com/avrajyoti07)

Part of a personal deep learning fundamentals learning series.

---

## 📝 License

This project is open source and available for learning purposes. Feel free to fork it, experiment with it, and adapt it for your own study.
