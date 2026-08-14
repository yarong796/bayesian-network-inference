#!/usr/bin/env python3
# Vandita Soni (vsoni3@u.rochester.edu)
# Ji Woong Hong (jhong36@u.rochester.edu)
# Yarong Xiao (yxiao37@u.rochester.edu)

import sys
import math
import random
from collections import defaultdict

class BayesianNetwork:
    def __init__(self):
        self.variables = []
        self.domains = {}
        self.parents = {}
        self.cpt = {}
        self.var_index = {}

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        cleaned_lines = []
        for ln in lines:
            ln = ln.split('#')[0].strip()
            if ln:
                cleaned_lines.append(ln)

        idx = 0

        num_vars = int(cleaned_lines[idx])
        idx += 1

        self.variables = []
        for _ in range(num_vars):
            parts = cleaned_lines[idx].split()
            idx += 1
            varname = parts[0]
            domvals = parts[1:]
            self.variables.append(varname)
            self.domains[varname] = domvals

        self.var_index = {v: i for i, v in enumerate(self.variables)}

        num_cpts = int(cleaned_lines[idx])
        idx += 1

        for _ in range(num_cpts):
            header = cleaned_lines[idx].split()
            idx += 1

            child_var = header[0]

            parent_vars = header[1:]
            self.parents[child_var] = parent_vars

            self.cpt[child_var] = {}

            num_parent_combos = 1
            for p in parent_vars:
                num_parent_combos *= len(self.domains[p])

            for i in range(num_parent_combos):
                probs_line = cleaned_lines[idx].split()
                idx += 1

                probs = [float(x) for x in probs_line]

                parent_assign = self.index_to_assignment(i, parent_vars)

                for val_i, child_val in enumerate(self.domains[child_var]):
                    key = tuple([child_val] + list(parent_assign))
                    self.cpt[child_var][key] = probs[val_i]

    def index_to_assignment(self, index, vars_list):

        if not vars_list:
            return ()

        domain_sizes = [len(self.domains[v]) for v in vars_list]
        assignments = []
        temp = index
        for size in reversed(domain_sizes):
            assignments.append(temp % size)
            temp //= size
        assignments.reverse()

        result = []
        for var, dom_i in zip(vars_list, assignments):
            dom_val = self.domains[var][dom_i]
            result.append(dom_val)
        return tuple(result)

    def get_probability(self, var, val, assignment):
        pvals = [assignment[p] for p in self.parents[var]]
        key = tuple([val] + pvals)
        return self.cpt[var][key]

    def sample_prior(self):
        assignment = {}
        for var in self.variables:
            r = random.random()
            cdf = 0.0
            for val in self.domains[var]:
                p = self.get_probability(var, val, assignment)
                cdf += p
                if r <= cdf:
                    assignment[var] = val
                    break
        return assignment

    def full_joint_prob(self, assignment):
        p = 1.0
        for var in self.variables:
            p *= self.get_probability(var, assignment[var], assignment)
        return p


def normalize(prob_list):
    s = sum(prob_list)
    if s == 0:
        return [1.0/len(prob_list)] * len(prob_list)
    return [p/s for p in prob_list]


def enumerate_all(bn, vars_list, evidence):
    if not vars_list:
        return 1.0
    Y = vars_list[0]
    rest = vars_list[1:]
    if Y in evidence:
        val = evidence[Y]
        probY = bn.get_probability(Y, val, evidence)
        return probY * enumerate_all(bn, rest, evidence)
    else:
        # sum over all possible values of Y
        total = 0.0
        for y_val in bn.domains[Y]:
            evidence[Y] = y_val
            probY = bn.get_probability(Y, y_val, evidence)
            total += probY * enumerate_all(bn, rest, evidence)
        del evidence[Y]
        return total


def exact_inference(bn, query_var, evidence):
    probs = []
    for qVal in bn.domains[query_var]:
        evidence[query_var] = qVal
        prob = enumerate_all(bn, bn.variables, evidence)
        probs.append(prob)
    del evidence[query_var]
    return normalize(probs)


def rejection_sampling(bn, query_var, evidence, num_samples=10000):
    counts = [0]*len(bn.domains[query_var])
    for _ in range(num_samples):
        sample = bn.sample_prior()
        consistent = True
        for v, val in evidence.items():
            if sample[v] != val:
                consistent = False
                break
        if not consistent:
            continue
        q_val = sample[query_var]
        q_index = bn.domains[query_var].index(q_val)
        counts[q_index] += 1

    return normalize(counts)


def gibbs_sampling(bn, query_var, evidence, num_samples=10000, burn_in=1000):
    hidden_vars = [v for v in bn.variables if v not in evidence]
    state = dict(evidence)  # copy
    for hv in hidden_vars:
        domvals = bn.domains[hv]
        state[hv] = random.choice(domvals)

    counts = [0]*len(bn.domains[query_var])

    for i in range(num_samples + burn_in):
        for hv in hidden_vars:
            probs = []
            for val in bn.domains[hv]:
                state[hv] = val
                p = bn.get_probability(hv, val, state)
                for child in bn.variables:
                    if hv in bn.parents[child]:
                        p *= bn.get_probability(child, state[child], state)
                probs.append(p)
            probs = normalize(probs)
            r = random.random()
            cum = 0.0
            for idx_val, pval in enumerate(probs):
                cum += pval
                if r <= cum:
                    state[hv] = bn.domains[hv][idx_val]
                    break
        if i >= burn_in:
            q_val = state[query_var]
            q_index = bn.domains[query_var].index(q_val)
            counts[q_index] += 1

    return normalize(counts)


def print_distribution(distribution):
    out_str = " ".join(f"{p:.4f}" for p in distribution)
    print(out_str)


def main_repl():
    bn = BayesianNetwork()

    default_num_samples = 10000
    default_burn_in = 1000

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == 'quit':
            break

        elif cmd == 'load':
            if len(parts) < 2:
                sys.stderr.write("Usage: load <filename>\n")
                continue
            filename = " ".join(parts[1:])
            bn.load_from_file(filename)

        elif cmd == 'xquery':
            query_var = parts[1]
            evidence = {}
            for ev in parts[2:]:
                if '=' in ev:
                    var, val = ev.split('=')
                    evidence[var] = val

            dist = exact_inference(bn, query_var, evidence)
            print_distribution(dist)

        elif cmd == 'rquery':

            query_var = parts[1]
            evidence = {}
            for ev in parts[2:]:
                if '=' in ev:
                    var, val = ev.split('=')
                    evidence[var] = val

            dist = rejection_sampling(bn, query_var, evidence,
                                     num_samples=default_num_samples)
            print_distribution(dist)

        elif cmd == 'gquery':

            query_var = parts[1]
            evidence = {}
            for ev in parts[2:]:
                if '=' in ev:
                    var, val = ev.split('=')
                    evidence[var] = val

            dist = gibbs_sampling(bn, query_var, evidence,
                                  num_samples=default_num_samples,
                                  burn_in=default_burn_in)
            print_distribution(dist)

        else:
            sys.stderr.write(f"Unknown command: {cmd}\n")


if __name__ == "__main__":
    main_repl()
