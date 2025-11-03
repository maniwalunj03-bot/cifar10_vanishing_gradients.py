# cifar10_vanishing_gradients.py
Vanishing Gradient Experiment in Deep CNNs (PyTorch, CIFAR-10) A comparative study of Sigmoid vs ReLU activation functions in a 4-layer convolutional neural network. Includes training curves, per-layer gradient flow analysis, and activation gradient heatmaps to visualize vanishing gradients in deep networks.
# 🔬 CIFAR-10 Vanishing Gradient Experiment  
### Sigmoid vs ReLU in a 4-Layer Deep CNN (PyTorch)

This project investigates the **vanishing gradient problem** by training two identical CNNs on CIFAR-10 — one using **Sigmoid** activations and the other using **ReLU** — and comparing:

✅ Training & test accuracy  
✅ Loss curves  
✅ Per-layer gradient magnitudes  
✅ Activation gradient heatmaps  
✅ Convergence speed & learning dynamics  

---

## 📌 Why This Experiment?

The **vanishing gradient problem** makes deep networks hard to train when activation functions like **Sigmoid / Tanh** squash gradients toward zero.

🔁 In shallow networks → **not a big issue**  
📉 In deeper networks → **training collapses / becomes very slow**

This repo provides a **clean, visual, experiment-based explanation** instead of only theory.

---

## 🧠 Model Architecture (Same for Both)

| Layer | Type | Output Shape |
|-------|------|--------------|
| Conv1 | 3 → 32 | 32×32 |
| Conv2 | 32 → 64 | 32×32 |
| Conv3 | 64 → 128 | 16×16 → 8×8 |
| Conv4 | 128 → 128 | 4×4 |
| FC    | 128×4×4 → 10 | logits |

🔁 Only **activation function changes**  
🔵 Model A → Sigmoid  
🟠 Model B → ReLU  

---

## 📊 Key Results

| Observation | Sigmoid | ReLU |
|-------------|---------|------|
| First few epochs | Slow start | Learns fast |
| Accuracy at Epoch 1 | ~45% | ~60% |
| Final Accuracy | ~72–74% | ~82–83% |
| Gradient flow | Shrinks layer-wise | Stable per layer |
| Convergence | Gradual | Rapid |

📌 Result: **ReLU trains faster and avoids vanishing gradients.**  
📌 Sigmoid eventually learns, but needs more epochs and loses accuracy.

---

## 📈 Plots & Visualizations

### ✅ Test Loss & Accuracy Curves
*(saved as `loss_acc_comparison.png`)*  
![Loss & Accuracy](save_dir/loss_acc_comparison.png)

### ✅ Per-Layer Gradient Norms
*(saved as `grad_norms_per_layer.png`)*  
![Gradient Norms](save_dir/grad_norms_per_layer.png)

### ✅ Activation Gradient Heatmap (Conv1)
*(saved as `activation_gradmaps_conv1.png`)*  
![Grad Heatmap](save_dir/activation_gradmaps_conv1.png)

---

## ▶️ How to Run

```bash
git clone https://github.com/maniwalunj03-bot/cifar10-vanishing-gradients.git
cd cifar10-vanishing-gradients
pip install -r requirements.txt
python cifar10_vanishing_gradients.py
