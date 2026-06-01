import math
import random
from heapq import heappush, heappop


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def euclidean_distance(point, center):
    """Return the Euclidean distance between two equal-length coordinate tuples."""
    return math.sqrt(sum((p - c) ** 2 for p, c in zip(point, center)))


def most_spread_dimension(D):
    """Return the index of the dimension with the greatest value range in D."""
    best_dim, best_spread = 0, float("-inf")
    for dim in range(len(D[0][0])):
        values = [row[0][dim] for row in D]
        spread = max(values) - min(values)
        if spread > best_spread:
            best_spread, best_dim = spread, dim
    return best_dim


def approximate_median(D, dim):
    """
    Return (median_value, median_row) along *dim* using a small random sample.

    Samples up to 11 points, insertion-sorts them along *dim*, then picks the
    middle element — a standard median-of-sample heuristic.
    """
    sample = random.sample(D, min(11, len(D)))

    for i in range(len(sample)):
        min_idx = i
        for j in range(i + 1, len(sample)):
            if sample[j][0][dim] < sample[min_idx][0][dim]:
                min_idx = j
        sample[i], sample[min_idx] = sample[min_idx], sample[i]

    median_row = sample[len(sample) // 2]
    return median_row[0][dim], median_row


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class Node:
    """A single node in a Ball Tree.

    Each node represents a D-dimensional ball enclosing a subset of the
    dataset. Leaf nodes hold exactly one point; internal nodes split their
    points between left and right child balls.

    Attributes:
        pivot  -- coordinates of the ball's centre point (tuple)
        data   -- arbitrary payload attached to the pivot
        radius -- max distance from pivot to any enclosed point
        left   -- left child Node, or None
        right  -- right child Node, or None
    """

    def __init__(self, pivot, data, radius=0, left=None, right=None):
        self.pivot  = pivot
        self.data   = data
        self.radius = radius
        self.left   = left
        self.right  = right


# ---------------------------------------------------------------------------
# Ball Tree
# ---------------------------------------------------------------------------

class BallTree:
    """Space-partitioning tree for efficient nearest-neighbour search.

    Builds an offline Ball Tree from a list of (point, data) pairs, then
    supports exact lookup and k-nearest-neighbour queries.

    Example
    -------
    >>> points = [((1, 2), "a"), ((3, 4), "b"), ((5, 6), "c")]
    >>> bt = BallTree()
    >>> bt.build(points)
    >>> bt.find_exact((3, 4))
    'b'
    >>> bt.knn_search((3, 4), k=2)
    [(3, 4), (1, 2)]
    """

    def __init__(self):
        self._root = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, D):
        """Build the tree from *D*, a list of (point_tuple, data) pairs."""
        self._root = self._build(D)
        return self._root

    def _build(self, D):
        if not D:
            return None

        if len(D) == 1:
            return Node(D[0][0], D[0][1])

        dim = most_spread_dimension(D)
        median_val, median_row = approximate_median(D, dim)
        pivot, pivot_data = median_row[0], median_row[1]

        left, right = [], []
        for row in D:
            pt = row[0]
            if pt[dim] < median_val:
                left.append(row)
            elif pt[dim] > median_val or pt != pivot:
                right.append(row)

        radius = max(euclidean_distance(row[0], pivot) for row in D)

        return Node(
            pivot, pivot_data, radius,
            left=self._build(left),
            right=self._build(right),
        )

    # ------------------------------------------------------------------
    # Exact search
    # ------------------------------------------------------------------

    def find_exact(self, goal):
        """Return the data associated with *goal*, or None if not found."""
        return self._find_exact(self._root, goal)

    def _find_exact(self, node, goal):
        if node is None:
            return None
        if node.pivot == goal:
            return node.data
        if euclidean_distance(goal, node.pivot) > node.radius:
            return None
        return self._find_exact(node.left, goal) or self._find_exact(node.right, goal)

    # ------------------------------------------------------------------
    # k-nearest-neighbour search
    # ------------------------------------------------------------------

    def knn_search(self, goal, k):
        """Return the *k* nearest pivot points to *goal* (unsorted)."""
        heap = []
        self._knn_search(self._root, goal, k, heap)
        return [pivot for (_, pivot) in heap]

    def _knn_search(self, node, goal, k, heap):
        if node is None:
            return

        dist = euclidean_distance(goal, node.pivot)

        # Prune: closest possible point in this ball is already farther than
        # the worst candidate collected so far
        if len(heap) == k and (dist - node.radius) >= -heap[0][0]:
            return

        # heapq is a min-heap, but we need a max-heap to efficiently track the
        # k closest points and evict the farthest. Negating dist lets the
        # largest distance sort to the top so heappop removes it when k is exceeded.
        heappush(heap, (-dist, node.pivot))
        if len(heap) > k:
            heappop(heap)

        if node.left is None and node.right is None:
            return

        if node.left is None:
            self._knn_search(node.right, goal, k, heap)
        elif node.right is None:
            self._knn_search(node.left, goal, k, heap)
        else:
            left_dist  = euclidean_distance(goal, node.left.pivot)
            right_dist = euclidean_distance(goal, node.right.pivot)
            near, far  = (node.left, node.right) if left_dist <= right_dist \
                         else (node.right, node.left)
            self._knn_search(near, goal, k, heap)
            self._knn_search(far,  goal, k, heap)