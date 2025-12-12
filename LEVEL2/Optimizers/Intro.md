# What is SGD? - optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

Instead of using ALL data to compute gradient →
we use one sample or a small batch.
So gradients become noisy → faster but unstable.
Mathematical Update Rule

For weights W:
W = W − η \* dL/dW

For bias b:
b = b − η \* dL/db

Where: η (eta) = learning rate
dL/dW = gradient of loss w.r.t weight

# Adam - optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Adam is an optimizer used in deep learning to update weights smarter and faster than normal Gradient Descent.
It combines the benefits of:
Momentum
RMSProp

It keeps track of - Mean of gradients (momentum) and Mean of squared gradients (adaptive learning rate).
