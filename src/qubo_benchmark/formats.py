"""Strict source parsers; source scorers do not call matrix normalization."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .model import Problem, binaryVector
from .storage import rejectHtml

PARSER_VERSION = '1.0.0'


def lines(payload):
    rejectHtml(payload)
    for line in payload.decode('utf-8-sig').splitlines():
        words = line.strip().split()
        if words and words[0] not in ('c','#','%') and not words[0].startswith(('#','%')):
            yield words


@dataclass
class MatrixSource:
    n: int
    entries: dict
    sign: int

    def canonical(self):
        pairs = {}
        for (i,j), value in self.entries.items():
            pairs[min(i,j),max(i,j)] = self.sign*value
        keys = sorted(k for k,v in pairs.items() if v != 0)
        return Problem(self.n,np.array([k[0] for k in keys],dtype=np.int64),
                       np.array([k[1] for k in keys],dtype=np.int64),
                       np.array([pairs[k] for k in keys],dtype=np.int64))

    def score(self, sample):
        x = binaryVector(sample,self.n)
        # Sum supplied directed entries, then account only for absent reverses.
        total = 0
        for (i,j),v in self.entries.items():
            total += int(v)*int(x[i])*int(x[j])
            if i != j and (j,i) not in self.entries:
                total += int(v)*int(x[i])*int(x[j])
        return self.sign*total


def parseMatrix(payload, bundlePosition=None, sign=1):
    if sign not in (-1,1): raise ValueError('Invalid objective sign')
    records = iter(lines(payload))
    try:
        count = 1
        if bundlePosition is not None:
            first = next(records)
            if len(first) != 1: raise ValueError('Expected bundle problem count')
            count = int(first[0])
            if not 1 <= bundlePosition <= count: raise ValueError('Bundle position out of range')
        selected = None
        for pos in range(1,count+1):
            header = next(records)
            if len(header) != 2: raise ValueError('Expected matrix dimension and entry count')
            n,m = map(int,header)
            if n <= 0 or m < 0 or m > n*n: raise ValueError('Invalid matrix header')
            entries = {}
            for _ in range(m):
                row = next(records)
                if len(row) != 3: raise ValueError('Expected i j coefficient')
                i,j,v = map(int,row); i -= 1; j -= 1
                if not (0 <= i < n and 0 <= j < n): raise ValueError('Matrix label out of range')
                if (i,j) in entries: raise ValueError('Duplicate directed matrix entry')
                if (j,i) in entries and entries[j,i] != v: raise ValueError('Conflicting symmetric entries')
                if not -2**63 <= v < 2**63: raise ValueError('Coefficient exceeds int64 storage')
                entries[i,j] = v
            if bundlePosition is None or pos == bundlePosition: selected = MatrixSource(n,entries,sign)
        if next(records,None) is not None: raise ValueError('Trailing matrix records')
        return selected
    except StopIteration as exc:
        raise ValueError('Truncated matrix data') from exc


@dataclass
class GraphSource:
    n: int
    edges: set

    def canonical(self):
        pairs = [(i,i,-1) for i in range(self.n)]
        pairs += [(i,j,1) for i,j in combinations(range(self.n),2) if (i,j) not in self.edges]
        r,c,v = zip(*pairs)
        return Problem(self.n,np.array(r),np.array(c),np.array(v))

    def details(self, sample):
        selected = np.flatnonzero(binaryVector(sample,self.n)).tolist()
        violations = sum((i,j) not in self.edges for i,j in combinations(selected,2))
        return dict(objective=-len(selected)+2*violations,selected_count=len(selected),
                    nonedge_violations=violations,is_clique=violations == 0,
                    valid_clique_size=len(selected) if violations == 0 else None)

    def score(self,sample): return self.details(sample)['objective']


def parseGraph(payload):
    n = m = None; edges = set()
    for row in lines(payload):
        if row[0] == 'p':
            if n is not None or len(row) != 4 or row[1] not in ('edge','edges'):
                raise ValueError('Invalid/duplicate DIMACS header')
            n,m = map(int,row[2:])
            if n <= 0 or not 0 <= m <= n*(n-1)//2: raise ValueError('Invalid graph counts')
        elif row[0] == 'e':
            if n is None or len(row) != 3: raise ValueError('Edge before header or malformed edge')
            i,j = (int(v)-1 for v in row[1:])
            if not 0 <= i < n or not 0 <= j < n or i == j: raise ValueError('Invalid vertex/self-loop')
            edge = (min(i,j),max(i,j))
            if edge in edges: raise ValueError('Duplicate undirected edge')
            edges.add(edge)
        else: raise ValueError('Unknown DIMACS record')
    if n is None or len(edges) != m: raise ValueError(f'DIMACS count mismatch: declared {m}, found {len(edges)}')
    return GraphSource(n,edges)


def parseWitness(payload, graph, target):
    records = list(lines(payload))
    if len(records) != 2 or len(records[0]) != 3: raise ValueError('Invalid result header/labels')
    cardinality,nodes,milliseconds = map(int,records[0])
    labels = list(map(int,records[1]))
    # The publisher MaxClique.java reader subtracts 1 and printer emits i+1;
    # BBMC.saveSolution reverses its internal vertex permutation before printing.
    if len(labels) != cardinality or len(set(labels)) != cardinality or any(v < 1 or v > graph.n for v in labels):
        raise ValueError('Invalid witness labels/cardinality')
    x = np.zeros(graph.n,dtype=np.int64); x[np.array(labels)-1] = 1
    details = graph.details(x)
    if not details['is_clique'] or details['objective'] != target or graph.canonical().score(x) != target:
        raise ValueError('Reference witness is not a matching clique')
    return x,dict(details,source_labels=labels,index_mapping='source label k -> index k-1',
                  search_nodes=nodes,published_milliseconds=milliseconds)
