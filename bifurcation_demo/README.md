# 1D simulated-bifurcation visual simulator

This Flask app evolves many independent particles in

`V(x,t) = K x^4 / 4 + a(t) x^2 / 2`, with `a(t)` linearly ramped from positive
to negative. The backend uses symplectic-Euler Hamiltonian updates:

`dx/dt = p/m`, `dp/dt = -(Kx^3 + ax) - gamma p`.

## Run

From this directory, create an environment and install Flask:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

The default damping is zero. The useful readout is `sign(x)`: particles can
keep oscillating after the bifurcation, so their momenta need not approach zero.
