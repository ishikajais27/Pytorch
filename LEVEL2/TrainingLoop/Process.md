# COMPLETE TRAINING LOOP — Full Process Explanation

## The Training Process

Training a neural network follows the same 6 fundamental steps:

---

## 1️⃣ Forward Pass (Make Predictions)

- Input data `X` is fed into the model.
- The model applies its layers (like Linear → Activation).
- Output `ŷ` (prediction) is produced.
- Internally, PyTorch builds a computation graph.

---

## 2️⃣ Calculate Loss (How Wrong the Model Is)

- Compare predictions `ŷ` with actual labels `y`.
- Use a **loss function** (e.g., MSELoss, CrossEntropyLoss).
- The loss is a **single scalar value** showing error.

Example: Loss = MSE(ŷ, y)

---

## 3️⃣ Backward Pass (Calculate Gradients)

- Call `loss.backward()`.
- PyTorch computes gradients using **backpropagation**.
- Computes ∂Loss/∂Weight and ∂Loss/∂Bias for every parameter.
- Stores them in:
  - `weight.grad`
  - `bias.grad`

These gradients tell:

> How much each weight contributed to the error.

---

## 4️⃣ Update Weights (Optimizer Step)

- The optimizer uses the gradients to update weights.
- Common optimizer: **Adam**
- Update rule (conceptual): new_weight = old_weight - learning_rate \* adjusted_gradient

- After updating, the model becomes slightly better.

---

## 5️⃣ Repeat for Several Epochs

- One full pass through training data = **1 epoch**.
- Training usually runs for tens or hundreds of epochs.
- Each epoch:
  - forward → loss → backward → update

The model improves gradually.

---

## 6️⃣ Validate on Test Data

- After training, use unseen data.
- Turn off gradient tracking (`torch.no_grad()`).
- Only run forward pass.
- Check accuracy or loss on test data.

---

## 🔁 Complete Loop Summary

1. Reset gradients
2. Forward pass
3. Compute loss
4. Backward pass
5. Optimizer weight update
6. Repeat
7. Validate

This is the core of training any neural network.
