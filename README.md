$ cat /Users/ellamorgenstern/README.md

# Ball Tree

A Python implementation of a Ball Tree — a space-partitioning data structure for efficient nearest-neighbour search in high-dimensional spaces.

## How it works

A Ball Tree recursively partitions points into nested hyperspheres ("balls"). At each node, the dataset is split along the dimension of greatest spread using an approximate median pivot. This allows nearest-neighbour queries to prune entire subtrees by checking whether the closest possible point in a ball is farther than the current best candidate.

         [pivot, r=5.1]
        /               \
  [pivot, r=2.3]    [pivot, r=3.0]
   /        \          /       \
 leaf      leaf      leaf     leaf

## Usage

Each entry in the dataset is a `(point, data)` pair, where `point` is a coordinate tuple and `data` is any payload.

```python
from ball_tree import BallTree

dataset = [
    ((1, 2), "a"),
    ((3, 4), "b"),
    ((5, 6), "c"),
    ((7, 8), "d"),
]

bt = BallTree()
bt.build(dataset)

# Exact lookup
bt.find_exact((3, 4))      # → "b"
bt.find_exact((0, 0))      # → None

# 2 nearest neighbours to (4, 4)
bt.knn_search((4, 4), k=2) # → [(3, 4), (5, 6)]
```

Works in any number of dimensions:

```python
dataset = [((x, y, z), label) for x, y, z, label in records]
bt.build(dataset)
bt.knn_search((0.5, 0.5, 0.5), k=3)
```

## API

| Method | Description |
|---|---|
| `build(D)` | Build the tree from a list of `(point, data)` pairs |
| `find_exact(goal)` | Return the data for an exact point match, or `None` |
| `knn_search(goal, k)` | Return the `k` nearest pivot points to `goal` |

## Complexity

| Operation | Time | Space |
|---|---|---|
| Build | O(n log² n) | O(n) |
| Exact search | O(log n) avg | O(log n) |
| k-NN search | O(k log n) avg | O(log n) |

Performance degrades in very high dimensions (the "curse of dimensionality") — a known limitation of all tree-based spatial indexes.
