"""Small Flask backend for a one-dimensional simulated-bifurcation demo."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@dataclass
class Settings:
    count: int = 180
    x_spread: float = 0.16
    p_spread: float = 0.32
    ramp_rate: float = 0.18
    k: float = 1.0
    gamma: float = 0.0
    dt: float = 0.012
    interaction_strength: float = 0.12
    mass: float = 1.0
    a_start: float = 1.2
    a_end: float = -1.2


@dataclass
class Simulation:
    settings: Settings = field(default_factory=Settings)
    x: list[float] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    time: float = 0.0

    def reset(self, incoming: dict | None = None) -> None:
        if incoming:
            for key in asdict(self.settings):
                if key in incoming:
                    setattr(self.settings, key, type(getattr(self.settings, key))(incoming[key]))
        s = self.settings
        self.x = [random.gauss(0, s.x_spread) for _ in range(s.count)]
        self.p = [random.gauss(0, s.p_spread) for _ in range(s.count)]
        edge_count = min(s.count * 2, s.count * (s.count - 1) // 2)
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < edge_count:
            i, j = random.sample(range(s.count), 2)
            pairs.add((min(i, j), max(i, j)))
        self.edges = [(i, j, random.choice((-1.0, 1.0)) * s.interaction_strength * random.uniform(0.35, 1.0)) for i, j in pairs]
        self.time = 0.0

    def a(self) -> float:
        # Linear ramp; it stops at the requested final double-well shape.
        return max(self.settings.a_end, self.settings.a_start - self.settings.ramp_rate * self.time)

    def step(self, steps: int = 4) -> None:
        s = self.settings
        for _ in range(max(1, min(steps, 40))):
            a = self.a()
            coupling_force = [0.0] * s.count
            for i, j, strength in self.edges:
                coupling_force[i] += strength * self.x[j]
                coupling_force[j] += strength * self.x[i]
            # Symplectic Euler with a simple viscous damping force -gamma p.
            for i, xi in enumerate(self.x):
                force = -(s.k * xi**3 + a * xi) - s.gamma * self.p[i] + coupling_force[i]
                self.p[i] += s.dt * force
                self.x[i] += s.dt * self.p[i] / s.mass
            self.time += s.dt

    def payload(self) -> dict:
        return {"x": self.x, "p": self.p, "edges": self.edges, "time": self.time, "a": self.a(), "settings": asdict(self.settings)}


sim = Simulation()
sim.reset()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/reset")
def reset():
    sim.reset(request.get_json(silent=True) or {})
    return jsonify(sim.payload())


@app.post("/api/step")
def step():
    body = request.get_json(silent=True) or {}
    sim.step(int(body.get("steps", 4)))
    return jsonify(sim.payload())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
