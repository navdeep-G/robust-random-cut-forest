""" Tests for the random_cut_forest module """
import numpy as np
from random_cut_forest import random_cut_forest
from utils.assertion import assert_equals_list
from utils.logging import dump_node_data


class TestRandomCutForest(object):

    """Test the RandomCutForestClass"""

    @classmethod
    def setup_class(cls):
        """ Called before each test. """
        cls._X = np.array([[0, 0, 0], [5, 5, 5], [20, 20, 20]])
        cls._forest = random_cut_forest.RandomCutForest(
            max_samples=cls._X.shape[0])
        cls._forest = cls._forest.fit(cls._X)

    def test_decision_function(self):
        """ Test the decision_function method. """

        # Note: The scores for these points will be very close to the scores of
        # the original X due to the fact that the forest has not been fitted on
        # much data. However, the points will still have lower path lengths
        # than the original vector
        points = np.array([
            [100, 100, 100],
            [1000, 1000, 1000],
            [-1000, -1000, -1000],
            [-10000, -10000, -10000],
            [-100000, -100000, -100000]
        ])

        for point in points:
            self._forest.add_point(point)
            is_more_anomalous = (self._forest.decision_function(
                [point]) < self._forest.decision_function(self._X)).all()
            assert is_more_anomalous == True

    def test_max_node_depth(self):
        """ Ensure we won't get depths above the average node depth """
        max_samples = 15
        max_node_depth = 5.0  # Don't bother traversing the tree beyond this
        num_points = 100
        features = 3
        max_expected_depth = random_cut_forest.average_path_length(max_samples)
        forest = random_cut_forest.RandomCutForest(
            max_node_depth=max_node_depth, max_samples=max_samples)
        X = np.random.randn(num_points * features).reshape(num_points, features)
        forest = forest.fit(X)
        depths = [tree.decision_function(X) for tree in forest.trees]

        # None of the depths should come out greater than the expected average
        assert np.all(depths <= max_expected_depth)

    def test_threshold_calculation(self):
        """ Test the calculation of a threshold within a forest """
        n_normal = 100
        p = 2
        X = np.random.randn(n_normal * p).reshape(n_normal, p)

        forest = random_cut_forest.RandomCutForest(
            max_samples=10).fit(X, calculate_threshold=True)

        # Never need a threshold less than zero
        threshold = max(0.0, np.round(forest.threshold, decimals=1))
        print(threshold)

        # No points in the data should be anomalous!
        assert threshold == 0

        # Should have a different threshold
        assert forest.threshold != self._forest.threshold


class TestRandomCutTree(object):

    """Test the RandomCutTree class"""

    @classmethod
    def setup_class(cls):
        """ Called before each test. """
        cls._X = np.array([[0, 0, 0], [10, 10, 10]])
        cls._tree = random_cut_forest.RandomCutTree()
        cls._tree.fit(cls._X)  # Create the base node

    def test_add_point(self):
        """Test the add_point method.

        Note: From the view of a tree we can more easily test the add_point
        method. There is some coverage from the TestTreeNode class but it is
        insufficient.
        """

        # Set the threshold so we know where new inserts will fall
        self._tree._root.split_threshold = 5.0

        assert_equals_list(self._tree._root.feature_mins, self._X[0])
        assert_equals_list(self._tree._root.feature_maxs, self._X[1])

        point = np.array([1, 1, 1])
        self._tree.add_point(point)
        assert_equals_list(
            self._tree._root.child_left.feature_maxs, point)

        # Add a point above the threshold
        greater_point = np.array([7, 7, 7])
        self._tree.add_point(greater_point)

        assert_equals_list(
            self._tree._root.child_right.feature_mins, greater_point)

        # Add a point outside of the feature range
        odd_point = np.array([-10, -10, -10])
        self._tree.add_point(odd_point)

        is_left = self._tree._root.is_point_left(odd_point)

        bottom = top = None
        if is_left:
            bottom = self._tree._root.child_left.feature_mins
            top = self._tree._root.child_left.feature_maxs
        else:
            bottom = self._tree._root.child_right.feature_mins
            top = self._tree._root.child_right.feature_maxs

        assert np.all(bottom <= odd_point) and np.all(odd_point <= top)
        assert_equals_list(odd_point, self._tree._root.feature_mins)

        odd_point = np.array([20, 20, 20])
        self._tree.add_point(odd_point)

        # is_right = odd_point in self._tree._root.child_right.X
        bottom = self._tree._root.child_right.feature_mins
        top = self._tree._root.child_right.feature_maxs

        assert np.all(bottom <= odd_point) and np.all(odd_point <= top)

        assert_equals_list(odd_point, self._tree._root.feature_maxs)

    def test_inserts_nodes_correctly(self):
        """ Make sure we can insert a node correctly into the tree """
        tree = random_cut_forest.RandomCutTree().fit(self._X)
        point = np.array([7, 7, 7])
        tree.add_point(point)

        is_left = tree._root.is_point_left(point)
        if is_left:
            assert_equals_list(point, tree._root.child_left.feature_maxs)
        else:
            assert_equals_list(point, tree._root.child_right.feature_mins)

    def test_insert_same_node(self):
        """ Ensure we add the same point to the tree correctly """
        tree = random_cut_forest.RandomCutTree()
        tree = tree.fit(self._X)
        point = np.array([2, 2, 2])
        tree.add_point(point)

        assert tree._root.num_points() == 3
        if tree._root.is_point_left(point):
            assert tree._root.child_left.num_points() == 2
        else:
            assert tree._root.child_right.num_points() == 2

        tree.add_point(point)
        assert tree._root.num_points() == 4
        if tree._root.is_point_left(point):
            assert tree._root.child_left.num_points() == 3
        else:
            assert tree._root.child_right.num_points() == 3

    def test_ensure_root_node_is_kept(self):
        """ Make sure the root node always has a `None` parent """
        num_points = np.random.randint(10, 101)
        features = np.random.randint(2, 10)
        points = np.random.randn(
            num_points * features).reshape(num_points, features)
        X = points[0:4, :]
        # points = points[4:, :]
        tree = random_cut_forest.RandomCutTree().fit(X)
        for point in points:
            tree.add_point(point)
            assert tree._root.parent is None

    def test_num_points_in_tree(self):
        """ Add a lot of points to the tree then make sure they are recorded """
        num_points = np.random.randint(10, 101)
        features = np.random.randint(2, 10)
        points = np.ones(shape=(num_points, features))
        X = np.ones(shape=(5, features))

        pts = 5
        tree = random_cut_forest.RandomCutTree().fit(X)
        assert tree._root.num_points() == pts
        for point in points:
            tree.add_point(point)
            pts += 1
            assert tree._root.num_points() == pts

    def test_custom_float_min(self):
        float_min = 4
        X = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
            [5, 5, 5],
            [7, 7, 7],
            [9, 9, 9],
            [10, 10, 10]
        ])
        tree = random_cut_forest.RandomCutTree(float_min=float_min)
        tree = tree.fit(X)
        points = np.array([
            [-1, -1, -1],
            [-2, -2, -2],
            [-3, -3, -3]
        ])
        for point in points:
            tree._root.split_threshold = 3.0
            tree._root = tree.add_new_node(node=tree._root, point=point)

        assert tree._root.num_points() == X.shape[0] + points.shape[0]
        assert_equals_list(points[-1], tree._root.child_left.feature_mins)

    def test_max_tree_depth(self):
        """ Ensure we can define a maximum depth and that all nodes beyond it
        are returned as non-anomalies - their score will be one """

        # Anything longer than this will be ignored automatically
        num_points = 100
        features = 3
        X = np.random.randn(num_points * features).reshape(num_points, features)
        max_depth = 5  # nodes
        tree = random_cut_forest.RandomCutTree(max_node_depth=max_depth)
        tree = tree.fit(X)
        depths = tree.decision_function(X)
        assert np.all(depths <= max_depth)

    def test_tree_depth_with_stump(self):
        """ Ensure that depth is preserved on fit and adding points """
        num_points = 100
        features = 3
        max_depth = 0
        X = np.random.randn(num_points * features).reshape(num_points, features)
        points = np.array([
            [-10, -10, -10],
            [-20, -20, -20],
            [-30, -30, -30],
            [40,  40,  40]
        ])
        tree = random_cut_forest.RandomCutTree(max_node_depth=max_depth)
        tree = tree.fit(X)

        min_max = (np.amin(X, axis=0), np.amax(X, axis=0))

        dump_node_data(tree._root)
        print(min_max)
        print(tree._root.feature_data)

        assert tree._root.child_left is None
        assert tree._root.child_right is None
        assert_equals_list(min_max[0], tree._root.feature_mins)
        assert_equals_list(min_max[1], tree._root.feature_maxs)

        for point in points:
            tree.add_point(point)

        min_max = (np.amin(points, axis=0), np.amax(points, axis=0))
        assert tree._root.child_left is None
        assert tree._root.child_right is None
        assert_equals_list(min_max[0], tree._root.feature_mins)
        assert_equals_list(min_max[1], tree._root.feature_maxs)


class TestTreeNode(object):

    """Tests the TreeNode class"""

    @classmethod
    def setup_class(cls):
        """ Called before each test. """
        cls._X = np.array([[1, 2, 3], [4, 5, 6]])
        feature_mins = np.min(cls._X, axis=0)
        feature_maxs = np.max(cls._X, axis=0)
        feature_data = (feature_mins, feature_maxs)

        cls._node = random_cut_forest.TreeNode(
            n=cls._X.shape[0],
            parent=None,
            feature_data=feature_data
        )
        cls._node.is_root = True
        cls._node._split(X=cls._X)

    def test_split(self):
        """ Test the _split method. """
        # Split the node first
        self._node._split(X=self._X)
        left_node = self._node.child_left
        right_node = self._node.child_right

        # Both should be leaves
        assert left_node.is_leaf and right_node.is_leaf

        # Ensure their feature mins/maxs are as expected
        assert_equals_list(left_node.feature_mins, self._X[0])
        assert_equals_list(left_node.feature_maxs, self._X[0])
        assert_equals_list(right_node.feature_mins, self._X[1])
        assert_equals_list(right_node.feature_maxs, self._X[1])

        new_X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        feature_mins = np.min(new_X, axis=0)
        feature_maxs = np.max(new_X, axis=0)
        feature_data = (feature_mins, feature_maxs)

        self._node = random_cut_forest.TreeNode(
            n=new_X.shape[0],
            parent=None,
            feature_data=feature_data
        )
        self._node.is_root = True
        self._node.split_threshold = 6.5  # Ensure we know how this will split
        self._node._split(X=new_X)

        assert_equals_list(self._node.child_left.feature_mins, new_X[0])
        assert_equals_list(self._node.child_left.feature_maxs, new_X[1])
        assert_equals_list(self._node.child_right.feature_mins, new_X[2])
        assert_equals_list(self._node.child_right.feature_maxs, new_X[3])

    def test_initialize_node(self):
        """ Test the initialization of a new TreeNode. """
        feature_ranges = [3, 3, 3]
        feature_mins = [1, 2, 3]
        feature_maxs = [4, 5, 6]

        assert_equals_list(self._node.feature_mins, feature_mins)
        assert_equals_list(self._node.feature_maxs, feature_maxs)
        assert_equals_list(self._node.feature_ranges, feature_ranges)

    def test_get_depth(self):
        """ Test the get_depth method. """
        point = [0, 0, 0]  # Should always fall to the left
        assert self._node.get_depth(point) == 1
        assert self._node.child_left.get_depth(point) == 0


def test_harmonic_approx():
    """Tests the harmonic_approx function."""
    # Simple test
    assert random_cut_forest.harmonic_approx(1) == 1.0022156649015328


def test_average_path_length():
    """Tests the average_path_length function. """
    assert random_cut_forest.average_path_length(2) == 1.0044313298030656
