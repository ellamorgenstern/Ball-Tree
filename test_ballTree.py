import math
import random
import pytest
from ballTree import *

# NOTE ON RANDOMNESS:
# BallTree construction uses randomness (approximate median selection),
# so tree structure may vary between runs.
# Tests focus on correctness of results rather than exact structure.

####################
##   Oracle Class  ##
####################

class BallTreeOracle:
    """Brute-force reference implementation for verifying BallTree correctness."""

    def __init__(self, D):
        self.data = D  # list of (point, data) tuples

    # linear scan — return data for the first matching point, or None
    def findExact(self, goal):
        for point, data in self.data:
            if point == goal:
                return data
        return None

    # sort all points by distance to goal, return the k closest pivots
    def knnSearch(self, goal, k):
        distances = []
        for point, data in self.data:
            dist = sphereDistance(point, goal)
            distances.append((dist, point))
        distances.sort()
        return [point for dist, point in distances[:k]]


####################
##    Helpers      ##
####################

# collect every node in the tree via DFS
def collect_nodes(node):
    if node is None:
        return []
    return [node] + collect_nodes(node.leftBall) + collect_nodes(node.rightBall)

# collect every pivot in a subtree
def all_descendant_pivots(node):
    if node is None:
        return []
    return [node.pivot] + all_descendant_pivots(node.leftBall) + all_descendant_pivots(node.rightBall)

# walk the tree and assert structural correctness at every node
def assert_valid_balltree(node):
    if node is None:
        return

    # no node should point to itself
    assert node.leftBall is not node
    assert node.rightBall is not node

    # leaf nodes must have radius 0 since they contain only the pivot
    if node.leftBall is None and node.rightBall is None:
        assert node.radius == 0, f"Leaf {node.pivot} should have radius 0"

    # every descendant must lie within this node's ball
    all_children = all_descendant_pivots(node.leftBall) + all_descendant_pivots(node.rightBall)
    for pivot in all_children:
        dist = sphereDistance(pivot, node.pivot)
        assert dist <= node.radius + 1e-9, (
            f"Descendant {pivot} is distance {dist} from pivot {node.pivot} "
            f"but radius is only {node.radius}"
        )

    assert_valid_balltree(node.leftBall)
    assert_valid_balltree(node.rightBall)

# convert knnSearch result to a set so order does not matter
def knn_pivots(result):
    return set(result)


SIMPLE_DATA = [
    ((1, 2), "a"),
    ((3, 4), "b"),
    ((5, 1), "c"),
    ((2, 8), "d"),
    ((7, 3), "e"),
]


#########################
## sphereDistance Tests ##
#########################

# a point should have zero distance to itself
def test_sphere_distance_self():
    assert sphereDistance((3, 4), (3, 4)) == 0

# distance from a to b must equal distance from b to a
def test_sphere_distance_symmetry():
    a, b = (1, 2, 3), (4, 5, 6)
    assert sphereDistance(a, b) == pytest.approx(sphereDistance(b, a))

# formula must work beyond 2D
def test_sphere_distance_high_dimension():
    a = tuple(range(10))
    b = tuple(0 for _ in range(10))
    expected = math.sqrt(sum(i**2 for i in range(10)))
    assert sphereDistance(a, b) == pytest.approx(expected)


###########################
## buildBallTree Tests    ##
###########################

# building from nothing should return no root
def test_build_empty():
    bt = BallTree()
    root = bt.buildBallTree([])
    assert root is None

# one point means one leaf node: correct pivot, zero radius, no children
def test_build_single_point():
    bt = BallTree()
    root = bt.buildBallTree([((0, 0), "only")])
    assert root is not None
    assert root.pivot == (0, 0)
    assert root.radius == 0
    assert root.leftBall is None
    assert root.rightBall is None

# tree must contain exactly one node per input point
def test_build_node_count():
    bt = BallTree()
    root = bt.buildBallTree(SIMPLE_DATA)
    assert len(collect_nodes(root)) == len(SIMPLE_DATA)

# every input point must appear somewhere in the tree
def test_build_all_points_present():
    bt = BallTree()
    root = bt.buildBallTree(SIMPLE_DATA)
    input_points = {row[0] for row in SIMPLE_DATA}
    tree_points = {n.pivot for n in collect_nodes(root)}
    assert input_points == tree_points

# structural invariants must hold on a typical dataset
def test_build_structural_invariants():
    bt = BallTree()
    root = bt.buildBallTree(SIMPLE_DATA)
    assert_valid_balltree(root)

# structural invariants must hold on a larger random dataset
def test_build_structural_invariants_large():
    random.seed(42)
    data = [(tuple(random.uniform(0, 100) for _ in range(3)), i) for i in range(100)]
    bt = BallTree()
    root = bt.buildBallTree(data)
    assert_valid_balltree(root)

# collinear points stress-test the split logic (one dimension has all the spread)
def test_build_collinear_points():
    data = [((i, 0), str(i)) for i in range(10)]
    bt = BallTree()
    root = bt.buildBallTree(data)
    assert_valid_balltree(root)
    assert len(collect_nodes(root)) == len(data)


###########################
## findExact Tests        ##
###########################

# every point in the dataset must be findable, matching the oracle's return value
def test_find_exact_all_points():
    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)
    oracle = BallTreeOracle(SIMPLE_DATA)
    for point, _ in SIMPLE_DATA:
        assert bt.findExact(point) == oracle.findExact(point)

# searching for a point not in the tree must return None
def test_find_exact_missing_point():
    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)
    oracle = BallTreeOracle(SIMPLE_DATA)
    goal = (99, 99)
    assert bt.findExact(goal) == oracle.findExact(goal)

# the root pivot is reachable without any recursion — sanity check
def test_find_exact_root_pivot():
    bt = BallTree()
    root = bt.buildBallTree(SIMPLE_DATA)
    oracle = BallTreeOracle(SIMPLE_DATA)
    assert bt.findExact(root.pivot) == oracle.findExact(root.pivot)

# single-point tree: hit and miss both work
def test_find_exact_single_point():
    data = [((1, 1), "solo")]
    bt = BallTree()
    bt.buildBallTree(data)
    oracle = BallTreeOracle(data)
    assert bt.findExact((1, 1)) == oracle.findExact((1, 1))
    assert bt.findExact((2, 2)) == oracle.findExact((2, 2))


###########################
## knnSearch Tests        ##
###########################

# k=1 must return a closest point to the goal
# If two points are equally close, either one is valid
def test_knn_k1():
    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)

    goal = (2, 3)
    result = knn_pivots(bt.knnSearch(goal, 1))

    assert result == {(1, 2)} or result == {(3, 4)}

# k == n must return every point in the dataset
def test_knn_k_equals_n():
    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)
    oracle = BallTreeOracle(SIMPLE_DATA)
    goal = (0, 0)
    k = len(SIMPLE_DATA)
    assert knn_pivots(bt.knnSearch(goal, k)) == set(oracle.knnSearch(goal, k))

# if the goal is already in the tree it should be its own nearest neighbor
def test_knn_goal_in_tree():
    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)
    oracle = BallTreeOracle(SIMPLE_DATA)
    goal = SIMPLE_DATA[0][0]
    assert knn_pivots(bt.knnSearch(goal, 1)) == set(oracle.knnSearch(goal, 1))

# results must match the oracle for every value of k from 1 to n
def test_knn_various_k():
    random.seed(42)

    bt = BallTree()
    bt.buildBallTree(SIMPLE_DATA)

    oracle = BallTreeOracle(SIMPLE_DATA)
    goal = (4, 3.5)

    for k in range(1, len(SIMPLE_DATA) + 1):
        assert knn_pivots(bt.knnSearch(goal, k)) == set(oracle.knnSearch(goal, k)), f"Failed at k={k}"

# stress test: 20 random queries on 100 points must all match the oracle
def test_knn_large_random():
    random.seed(42)
    data = [(tuple(random.uniform(0, 100) for _ in range(3)), i) for i in range(100)]
    bt = BallTree()
    bt.buildBallTree(data)
    oracle = BallTreeOracle(data)

    random.seed(7)
    for _ in range(20):
        goal = tuple(random.uniform(0, 100) for _ in range(3))
        k = random.randint(1, 10)
        assert knn_pivots(bt.knnSearch(goal, k)) == set(oracle.knnSearch(goal, k))

# knn must work correctly in higher dimensions, not just 2D
def test_knn_high_dimension():
    random.seed(0)
    data = [(tuple(random.uniform(0, 10) for _ in range(8)), i) for i in range(50)]
    bt = BallTree()
    bt.buildBallTree(data)
    oracle = BallTreeOracle(data)
    goal = tuple(random.uniform(0, 10) for _ in range(8))
    assert knn_pivots(bt.knnSearch(goal, 5)) == set(oracle.knnSearch(goal, 5))

pytest.main(["-v", "-s", "test_ballTree.py"])
