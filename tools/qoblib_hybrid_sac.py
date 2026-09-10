"""Conditional mixed-action SAC for experiment-side solver control.

Discrete actions are enumerated exactly; continuous actions use a conditional
tanh Gaussian and pathwise gradients. No solver or landscape model is learned.
Reference: Delalleau et al., https://arxiv.org/abs/1912.11077.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_biqmac_sac import DiscreteSAC


RANGES = {
    "SBM": [("dt", .05, 1.2, True), ("a0", .3, 3., True),
            ("c0", 1e-8, .1, True), ("gamma", 1e-5, .03, True),
            ("initial_scale", .005, .3, True)],
    "SVL": [("dt", .001, .05, True), ("mass", .2, 5., True),
            ("damping", .02, 3., True), ("temperature", 1e-5, .3, True),
            ("transverse_field_initial", .2, 4., True),
            ("transverse_field_final", .001, .2, True),
            ("problem_scale_initial", 1e-9, .001, True),
            ("problem_scale_final", 1e-7, 2., True)],
    "TRF": [("time_step", .005, .2, True), ("mobility", .3, 3., True),
            ("route_strength", .05, 5., True), ("gamma", .001, 1., True),
            ("kappa_initial", -3., -.1, False), ("kappa_final", .3, 5., True),
            ("schedule_exponent", .3, 3., True)],
}


def branches(solver):
    if solver == "SBM":
        return [dict(c0_zero=a, gamma_zero=b) for a, b in itertools.product((False, True), repeat=2)]
    if solver == "SVL":
        return [dict(integrator=i, temperature_zero=t,
                     transverse_field_final_zero=f, problem_scale_initial_zero=p)
                for i, t, f, p in itertools.product(
                    ("euler_maruyama", "weak_order_2"), (False, True), (False, True), (False, True))]
    return [dict(integrator=i, route_strength_zero=r, gamma_zero=g, candidate_interval=c)
            for i, r, g, c in itertools.product(("euler", "heun"), (False, True),
                                               (False, True), (10, 100, 250))]


def decode(solver, action):
    discrete, continuous = action
    result = {}
    for coordinate, (name, low, high, logarithmic) in zip(continuous, RANGES[solver]):
        fraction = (float(np.clip(coordinate, -1., 1.)) + 1.) / 2.
        result[name] = (math.exp(math.log(low) + fraction * math.log(high / low))
                        if logarithmic else low + fraction * (high - low))
    for name, value in branches(solver)[discrete].items():
        if name.endswith("_zero"):
            if value:
                result[name[:-5]] = 0.
        else:
            result[name] = value
    return result


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, outputs))


class ConditionalActor(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.count, self.dim = len(branches(solver)), len(RANGES[solver])
        self.logits = mlp(8, self.count)
        self.gaussian = mlp(8, self.count * self.dim * 2)
        mask = torch.ones(self.count, self.dim)
        for i, branch in enumerate(branches(solver)):
            for j, (name, *_rest) in enumerate(RANGES[solver]):
                if branch.get(name + "_zero", False):
                    mask[i, j] = 0.
        self.register_buffer("mask", mask)

    def forward(self, states, deterministic=False):
        log_discrete = self.logits(states).log_softmax(-1)
        mean, log_std = self.gaussian(states).reshape(-1, self.count, 2, self.dim).unbind(2)
        log_std = log_std.clamp(-5., 1.)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        actions = raw.tanh()
        correction = 2. * (math.log(2.) - raw - F.softplus(-2. * raw))
        log_continuous = ((normal.log_prob(raw) - correction) * self.mask).sum(-1)
        return log_discrete, actions * self.mask, log_continuous


class HybridSAC:
    def __init__(self, solver, seed):
        torch.manual_seed(seed)
        self.solver = solver
        self.actor = ConditionalActor(solver)
        self.count, self.dim = self.actor.count, self.actor.dim
        self.q1, self.q2 = mlp(8 + self.dim + self.count, 1), mlp(8 + self.dim + self.count, 1)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=.0003)
        self.qParameters = [*self.q1.parameters(), *self.q2.parameters()]
        self.criticOptimizer = torch.optim.Adam(self.qParameters, lr=.0003)
        self.replay, self.cursor, self.updates = [], 0, 0
        self.rng = np.random.default_rng(seed)
        self.lastMetrics = {}

    def choose(self, state, deterministic=False):
        if len(self.replay) < 64 and not deterministic:
            branch = int(self.rng.integers(self.count))
            values = self.rng.uniform(-1., 1., self.dim) * self.actor.mask[branch].numpy()
            return branch, values.astype(np.float32)
        with torch.no_grad():
            logp, values, _ = self.actor(torch.as_tensor(state).unsqueeze(0), deterministic)
            branch = int(logp[0].argmax()) if deterministic else int(torch.distributions.Categorical(logits=logp[0]).sample())
            return branch, values[0, branch].numpy().copy()

    def inputs(self, states, discrete, continuous):
        return torch.cat((states, F.one_hot(discrete, self.count).float(), continuous), -1)

    def values(self, states, q1, q2):
        logp, continuous, logc = self.actor(states)
        size = len(states)
        inputs = self.inputs(states[:, None, :].expand(-1, self.count, -1),
                             torch.arange(self.count).expand(size, -1), continuous)
        q = torch.minimum(q1(inputs).squeeze(-1), q2(inputs).squeeze(-1))
        # Separate entropy terms; masked-off continuous coordinates earn none.
        return (logp.exp() * (q - .002 * logp - .002 * logc)).sum(-1)

    def remember(self, state, action, reward, next_state, terminal):
        item = (state.copy(), (int(action[0]), np.array(action[1], dtype=np.float32)),
                float(reward), next_state.copy(), float(terminal))
        if len(self.replay) < 100000:
            self.replay.append(item)
        else:
            self.replay[self.cursor] = item
        self.cursor = (self.cursor + 1) % 100000
        if len(self.replay) >= 64:
            for _ in range(2):
                self.update()

    def update(self):
        batch = [self.replay[int(i)] for i in self.rng.choice(len(self.replay), 64, replace=False)]
        states = torch.as_tensor(np.stack([b[0] for b in batch]))
        discrete = torch.tensor([b[1][0] for b in batch], dtype=torch.long)
        continuous = torch.as_tensor(np.stack([b[1][1] for b in batch]))
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        following = torch.as_tensor(np.stack([b[3] for b in batch]))
        terminal = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        with torch.no_grad():
            target = rewards + (1. - terminal) * self.values(following, self.target1, self.target2)
        inputs = self.inputs(states, discrete, continuous)
        loss = F.smooth_l1_loss(self.q1(inputs).squeeze(-1), target)
        loss = loss + F.smooth_l1_loss(self.q2(inputs).squeeze(-1), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Hybrid SAC critic loss")
        self.criticOptimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.qParameters, 10.)
        self.criticOptimizer.step()
        for parameter in self.qParameters:
            parameter.requires_grad_(False)
        actor_loss = -self.values(states, self.q1, self.q2).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Nonfinite Hybrid SAC actor loss")
        self.actorOptimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actorOptimizer.step()
        for parameter in self.qParameters:
            parameter.requires_grad_(True)
        with torch.no_grad():
            for target_model, model in ((self.target1, self.q1), (self.target2, self.q2)):
                for target_parameter, parameter in zip(target_model.parameters(), model.parameters()):
                    target_parameter.lerp_(parameter, .005)
        self.updates += 1
        self.lastMetrics = dict(critic_loss=float(loss.detach()), actor_loss=float(actor_loss.detach()),
                                updates=self.updates, replay_size=len(self.replay))

    def checkpoint(self):
        return {"solver": self.solver, "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "target1": self.target1.state_dict(), "target2": self.target2.state_dict(),
                "actor_optimizer": self.actorOptimizer.state_dict(),
                "critic_optimizer": self.criticOptimizer.state_dict(), "replay": self.replay,
                "cursor": self.cursor, "updates": self.updates,
                "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}


def frozen_actor(model):
    return copy.deepcopy(model.actor).eval()


def frozen_choice(actor, state, hybrid):
    with torch.no_grad():
        states = torch.as_tensor(state).unsqueeze(0)
        if not hybrid:
            return int(actor(states)[0].argmax())
        logp, actions, _ = actor(states, deterministic=True)
        branch = int(logp[0].argmax())
        return branch, actions[0, branch].numpy().copy()
