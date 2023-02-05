import itertools


def all_combinations(l: int) -> itertools.product:
    return itertools.product([-1, 1], repeat=l)


def all_n_combinations(atoms: [str]) -> [list]:
    combs = list(all_combinations(len(atoms)))
    return [[atom for atom, v in zip(atoms, val) if v == 1] for val in combs]