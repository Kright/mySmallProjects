There are my experiments for with neural networks with complex numbers and geometric multivectors.

Used dataset is mnist, because models trained on it really fast and I can do a lot of experiments on my machine

I use pytorch.
Since it is just a fast experiment, code is AI-generated.

I tried:
* real numbers
* complex
* CL(2, 0)
* CL(3, 0)

## Activation functions:
* PReLU
* nn.ReLU
* nn.LeakyReLU(0.1)
* Cardioid
* Ga3ModGLU
* SplitELU
* SplitTanh
* MagnitudeBasedGLU
* Ga3SoftRooting
* Ga3Sigmoid
* ModReLU
* LogModReLU
* SoftModReLU
* zReLU

## Top activations:

ReLU-like are the best (ReLU, LeakyReLU, PReLU). 
These activations are element-wise and know nothing about multivectors structure.

Cardioid: 
```
class Cardioid(nn.Module):
    def forward(self, x):
        a0, a1, a2, a12 = get_all(x)
        mag = torch.sqrt(a0**2 + a1**2 + a2**2 + a12**2 + 1e-8)
        scale = 0.5 * (1.0 + a0 / mag)
        return make_ga2(a0 * scale, a1 * scale, a2 * scale, a12 * scale)
```

ModGLU
```
class Ga2ModGLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.make_gates = Ga2Conv2d(channels, channels, 1)

    def forward(self, x):
        if x.ndim == 2:
            gates = self.make_gates(x.view(*x.shape, 1, 1)).view(x.shape[0], -1)
        else:
            gates = self.make_gates(x)

        g0, g1, g2, g12 = get_all(gates)
        # Analogy to ComplexModGLU: first component sigmoid, others tanh
        mult = torch.sigmoid(g0) * torch.tanh(g1) * torch.tanh(g2) * torch.tanh(g12)

        a0, a1, a2, a12 = get_all(x)
        return make_ga2(a0 * mult, a1 * mult, a2 * mult, a12 * mult)
```
