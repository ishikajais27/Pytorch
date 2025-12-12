Input X (scalar/vector)
|
v
[Neuron] w, b
| Forward: z = X·w + b
v
Activation: a = f(z)
v
Prediction ŷ
|
v
Loss L = loss(ŷ, y)
|
v Backward:
dL/dŷ → dz = dL/da * f'(z) → dw = X*dz, db = dz
|
v
Update w,b
