"""Exact block improvement and budgeted proof search for geometric/SUSY solvers.

The master model is the exact dyadic value of every input IEEE coefficient,
including duplicate edges, diagonals and the offset. Python integers prevent
overflow. This CPU phase is deliberately separate from floating GPU dynamics.
"""

from fractions import Fraction
from itertools import product


class _ExactIsing:
    def __init__(self, problem):
        ratios = [float(v).as_integer_ratio() for v in
                  [problem.offset, *problem.linear, *problem.quadraticBiases]]
        denominator = max(d for _, d in ratios)
        coefficients = [a*(denominator//b) for a, b in ratios]
        n = problem.variableCount
        offset, linear = coefficients[0], coefficients[1:n+1]
        edges = {}
        for head, tail, value in zip(problem.quadraticHeads, problem.quadraticTails, coefficients[n+1:]):
            i, j = sorted((int(head), int(tail)))
            if i == j:
                linear[i] += value
            else:
                edges[i, j] = edges.get((i, j), 0) + value
        self.edges = {key: value for key, value in edges.items() if value}
        self.h = [2*v for v in linear]
        self.c = 4*offset + 2*sum(linear) + sum(self.edges.values())
        self.adj = [{} for _ in range(n)]
        for (i, j), value in self.edges.items():
            self.adj[i][j] = self.adj[j][i] = value
            self.h[i] += value
            self.h[j] += value
        self.denominator = 4*denominator
        self.groups = tuple(tuple(group) for group in problem.iterOneHotGroups())

    def energy(self, spins):
        return (self.c + sum(h*s for h, s in zip(self.h, spins))
                + sum(v*spins[i]*spins[j] for (i, j), v in self.edges.items()))

    def feasible(self, spins):
        return all(sum(spins[i] == 1 for i in group) == 1 for group in self.groups)

    def bound(self, constant, fields, iterations):
        edges = [(i, j, v) for i in fields for j, v in self.adj[i].items() if j in fields and i < j]
        degree = {i: 0 for i in fields}
        bound = constant-sum(abs(v) for v in fields.values())-sum(abs(v) for _, _, v in edges)
        for i, j, v in edges:
            degree[i] += abs(v)
            degree[j] += abs(v)
        # Disjoint-pair strengthening does not reuse any local field.
        used = set()
        for i, j, v in sorted(edges, key=lambda e: (-abs(e[2]), e[0], e[1])):
            if i not in used and j not in used:
                pair = min(v-abs(fields[i]+fields[j]), -v-abs(fields[i]-fields[j]))
                bound += pair+abs(fields[i])+abs(fields[j])+abs(v)
                used.update((i, j))
        maximum = max(degree.values(), default=0)
        if not iterations or not maximum:
            return bound
        # Quantized projected relaxation. Every iterate is independently verified
        # by the integer tangent formula, so truncation cannot invalidate pruning.
        denominator = 1 << 16
        x = {i: 0 for i in fields}
        for _ in range(iterations):
            ax = {i: degree[i]*x[i] for i in fields}
            for i, j, v in edges:
                ax[i] += v*x[j]
                ax[j] += v*x[i]
            q = {i: ax[i]+denominator*fields[i] for i in fields}
            numerator = (2*denominator**2*constant-denominator**2*sum(degree.values())
                         -sum(x[i]*ax[i] for i in fields)
                         -2*denominator*sum(abs(v) for v in q.values()))
            divisor = 2*denominator**2
            bound = max(bound, -((-numerator)//divisor))
            x = {i: max(-denominator, min(denominator, x[i]-q[i]//(2*maximum))) for i in fields}
        return bound


def refineAndCertify(problem, sample, p):
    """Return a feasible incumbent and, when requested, exact certified bounds.

Proof bounds also apply with declared one-hot groups: infeasible branches are
discarded, while objective relaxations remain valid lower bounds. Equality of
the exact bounds is the only condition producing OPTIMAL.
"""
    model = _ExactIsing(problem)
    spins = [2*int(v)-1 for v in sample]
    n = len(spins)
    upper = model.energy(spins)
    fields = [model.h[i]+sum(v*spins[j] for j, v in model.adj[i].items()) for i in range(n)]
    moves = 0
    for sweep in range(p['block_sweeps']):
        order = list(range(n))
        shift = sweep % n
        order = order[shift:]+order[:shift]
        for start in range(0, n, p['block_size']):
            block = order[start:start+p['block_size']]
            blockSet = set(block)
            local = {i: fields[i]-sum(v*spins[j] for j, v in model.adj[i].items() if j in blockSet)
                     for i in block}
            edges = [(i, j, v) for i in block for j, v in model.adj[i].items() if j in blockSet and i < j]
            groups = [group for group in model.groups if blockSet.intersection(group)]
            old = sum(local[i]*spins[i] for i in block)+sum(v*spins[i]*spins[j] for i, j, v in edges)
            best, chosen = old, None
            for values in product((-1, 1), repeat=len(block)):
                trial = dict(zip(block, values))
                if any(sum(trial.get(i, spins[i]) == 1 for i in group) != 1 for group in groups):
                    continue
                energy = sum(local[i]*trial[i] for i in block)+sum(v*trial[i]*trial[j] for i, j, v in edges)
                if energy < best:
                    best, chosen = energy, trial
            if chosen is not None:
                for i, value in chosen.items():
                    delta = value-spins[i]
                    for j, coupling in model.adj[i].items():
                        fields[j] += coupling*delta
                    spins[i] = value
                upper += best-old
                moves += 1
    diagnostics = dict(status='HEURISTIC', blockMoves=moves,
                       exactCoefficientModel='IEEE dyadic', exactUpperBound=str(Fraction(upper, model.denominator)))
    if not p['proof_nodes']:
        return tuple((s+1)//2 for s in spins), diagnostics

    rootFields = dict(enumerate(model.h))
    lower = model.bound(model.c, rootFields, p['bound_iterations'])
    frontier = [(lower, model.c, rootFields, {})]
    nodes = 0
    while frontier and nodes < p['proof_nodes']:
        bound, constant, fields, fixed = frontier.pop()
        nodes += 1
        if bound >= upper:
            continue
        if not fields:
            candidate = [fixed[i] for i in range(n)]
            if model.feasible(candidate) and constant < upper:
                upper, spins = constant, candidate
            continue
        key = max(fields, key=lambda i: (abs(fields[i])+sum(abs(v) for j, v in model.adj[i].items()
                                                                           if j in fields), -i))
        # LIFO: visit incumbent sign first. Both children are always accounted for.
        for sign in (-spins[key], spins[key]):
            childFixed = dict(fixed)
            childFixed[key] = sign
            if any(sum(childFixed.get(i) == 1 for i in group) > 1 or
                   (all(i in childFixed for i in group) and not any(childFixed[i] == 1 for i in group))
                   for group in model.groups):
                continue
            childConstant = constant+sign*fields[key]
            childFields = {i: value+sign*model.adj[key].get(i, 0) for i, value in fields.items() if i != key}
            childBound = max(bound, model.bound(childConstant, childFields, p['bound_iterations']))
            if childBound < upper:
                frontier.append((childBound, childConstant, childFields, childFixed))
    lower = min([upper]+[item[0] for item in frontier])
    diagnostics.update(status='OPTIMAL' if lower == upper else 'FEASIBLE_GAP',
        proofNodes=nodes, unresolvedNodes=sum(item[0] < upper for item in frontier),
        exactLowerBound=str(Fraction(lower, model.denominator)),
        exactUpperBound=str(Fraction(upper, model.denominator)),
        exactGap=str(Fraction(upper-lower, model.denominator)))
    return tuple((s+1)//2 for s in spins), diagnostics
