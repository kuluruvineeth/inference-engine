from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class CandidateTree:
    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def root(self) -> int:
        return 0

    def add(self, token: int, parent: int) -> int:
        if not self.tokens and parent != -1:
            raise ValueError("the first node must be the root, with parent -1")
        if self.tokens and not 0 <= parent < len(self.tokens):
            raise ValueError(f"parent {parent} is not an existing node")
        self.tokens.append(token)
        self.parents.append(parent)
        return len(self.tokens) - 1

    def depth_of(self, node: int) -> int:
        depth = 0
        while self.parents[node] != -1:
            node = self.parents[node]
            depth += 1
        return depth

    def ancestors(self, node: int) -> list[int]:
        chain: list[int] = []
        cursor = self.parents[node]
        while cursor != -1:
            chain.append(cursor)
            cursor = self.parents[cursor]
        return list(reversed(chain))

    def children_of(self, node: int) -> list[int]:
        return [i for i, parent in enumerate(self.parents) if parent == node]

    def leaves(self) -> list[int]:
        has_child = set(self.parents)
        return [i for i in range(self.size) if i not in has_child]

    def path_to(self, node: int) -> list[int]:
        return self.ancestors(node) + [node]

    def paths(self) -> list[list[int]]:
        return [self.path_to(leaf) for leaf in self.leaves()]

    def visibility_mask(self) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=bool)
        for node in range(self.size):
            mask[node, node] = True
            for ancestor in self.ancestors(node):
                mask[node, ancestor] = True
        return mask

    def positions_from(self, base_position: int) -> list[int]:
        return [base_position + self.depth_of(node) for node in range(self.size)]


def build_dense_tree(root_token: int, level_candidates: list[list[int]]) -> CandidateTree:
    tree = CandidateTree()
    tree.add(root_token, -1)
    frontier = [tree.root]

    for candidates in level_candidates:
        next_frontier: list[int] = []
        for parent in frontier:
            for token in candidates:
                next_frontier.append(tree.add(int(token), parent))
        frontier = next_frontier

    return tree


def build_chain(root_token: int, tokens: list[int]) -> CandidateTree:
    tree = CandidateTree()
    tree.add(root_token, -1)
    parent = tree.root
    for token in tokens:
        parent = tree.add(int(token), parent)
    return tree
