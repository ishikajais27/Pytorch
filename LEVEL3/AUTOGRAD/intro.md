# 🔹 AUTOGRAD — Automatic Differentiation in PyTorch

PyTorch Autograd automatically calculates **gradients** for tensors.  
Gradients are essential for training neural networks because they tell the network **how to adjust its parameters** to reduce error.

---

## 1️⃣ What is a Gradient?

- A **gradient** is the **rate of change** or slope of a function.
- For a function `y = f(x)`, the gradient tells us:
  > "If I change `x` a little, how much does `y` change?"

**Purpose in neural networks:**

- To know **how each weight affects the loss** and how to update it.

---

## 2️⃣ Manual vs Automatic Gradients

- **Manual:** You compute derivatives by hand.
- **Autograd:** PyTorch automatically computes derivatives for all tensors with `requires_grad=True`.
- Autograd uses a **computation graph** and the **chain rule** internally.

---

## 3️⃣ Computation Graph

- Every operation on a tensor with `requires_grad=True` builds a **directed graph**.
- **Nodes:** tensors
- **Edges:** operations
- Backward pass traverses the graph to compute gradients.

**Key point:** Only tensors with `requires_grad=True` are tracked.

---

## 4️⃣ Forward Pass

- Perform operations on input tensors → compute output
- PyTorch **records all operations** in the computation graph.

---

## 5️⃣ Backward Pass

- Call `.backward()` on a scalar output
- PyTorch computes **gradients for all input tensors** involved in the forward pass
- Gradients are stored in `.grad` attributes of tensors

**Purpose:** Provides directions to **update weights** to reduce error.

---

## 6️⃣ Why We Derive / Calculate Gradients

- Neural networks learn by adjusting weights.
- Gradient tells us **how to change each weight to reduce error**.

**Example:**

- Output = weight × input, True label = 10, Prediction = 6
- Error = 4, Gradient = 2 → Update weight by 2 × learning rate

**Analogy:**

- Hiking down a hill: slope = gradient, lowest point = minimum loss

---

## 7️⃣ Chain Rule

- Autograd automatically applies the **chain rule** for nested or complex functions.
- Example: z = (x + y)²

  - ∂z/∂x = 2\*(x + y)
  - ∂z/∂y = 2\*(x + y)

- Works for **multi-layer neural networks** as well.

---

## 8️⃣ Neural Network Parameters

- Weights and biases are **tensors with requires_grad=True**
- Forward pass → compute output
- Compute loss
- Backward pass → compute gradients automatically
- Optimizer updates weights using gradients

---

## 9️⃣ Gradient Accumulation

- Gradients **accumulate by default** in PyTorch
- Always **zero gradients** before the next backward pass to avoid accumulation errors:
  ```text
  optimizer.zero_grad()
  ```
