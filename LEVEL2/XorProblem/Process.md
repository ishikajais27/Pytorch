# 🔥 XOR PROBLEM — Multi-Layer Neural Network (MLP)

XOR is a classic problem in neural networks:

| Input | Output |
| ----- | ------ |
| 0,0   | 0      |
| 0,1   | 1      |
| 1,0   | 1      |
| 1,1   | 0      |

- XOR is **not linearly separable** → cannot be solved by a single-layer perceptron.
- Requires **multi-layer neural network** (MLP).

---

## 1️⃣ Forward Pass (Make Predictions)

- Input data `X` is fed into the network.
- Input layer → hidden layer (4 neurons) → output layer (1 neuron).
- Hidden layer uses **ReLU** activation.
- Output layer uses **Sigmoid** activation.
- Output `ŷ` (prediction) is produced.
- Internally, PyTorch builds a **computation graph**.

---

## 2️⃣ Calculate Loss (How Wrong the Model Is)

- Compare predictions `ŷ` with true labels `y`.
- Use **Binary Cross Entropy Loss** (BCELoss) because output is 0 or 1.
- Loss is a **single scalar** representing error.

---

## 3️⃣ Backward Pass (Calculate Gradients)

- Call `loss.backward()`.
- PyTorch computes **gradients using backpropagation**.
- Computes ∂Loss/∂Weight and ∂Loss/∂Bias for all parameters.
- Stored in:
  - `weight.grad`
  - `bias.grad`

Gradients indicate:

> How much each weight contributes to the error.

---

## 4️⃣ Update Weights (Optimizer Step)

- Optimizer uses gradients to update weights.
- We use **Adam optimizer**.
- Update rule conceptually:

- After update, model becomes slightly better.

---

## 5️⃣ Repeat for Many Epochs

- One full pass through the training data = **1 epoch**.
- Repeat steps: forward → loss → backward → optimizer step.
- Training continues until model learns XOR pattern.
- We use **5000 epochs** for convergence.

---

## 6️⃣ Validate on Test Data

- Use same input data (or unseen data) to check predictions.
- Turn off gradient tracking: `torch.no_grad()`.
- Only run forward pass.
- Output should be close to `[0, 1, 1, 0]`.

---

## 🔁 Complete Loop Summary

1. Reset gradients
2. Forward pass → predictions
3. Compute loss
4. Backward pass → compute gradients
5. Optimizer weight update
6. Repeat for many epochs
7. Validate / predict

This is the complete XOR MLP training process.
