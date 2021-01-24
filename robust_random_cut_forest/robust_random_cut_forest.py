from __future__ import division
import numpy as np
from scipy import stats
from warnings import warn


class RobustRandomCutForest(object):
    """Robust Random Cut Forest

    Return the anomaly score of each sample using the Robust  Random Cut Forest algorithm
    The Robust Random Cut Forest 'isolates' observations by randomly selecting a feature
    with probability proportional to its range and then uniformly selecting a split
    at random between the maximum and minimum values of the selected feature.
    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a
    measure of abnormality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.


    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    threshold : float in (0., 0.5), optional (default=0.25)
        The threshold of the score used to determine outliers.  Lower means
        fewer points are considered outliers.  This can be calculated to
        allow for an expected proportion of points to be considered outliers
        through the contamination_pct parameter and setting calculate_threshold
        to be True during fitting.

    contamination_pct : float in (0., 0.5), optional (default=0.01)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    random_features : boolean, optional (default=False)
        If True, the feature a tree splits on is chosen uniformaly at random
        (among all features for which the range is nonzero) rather than
        proportional to its range.

    float_min : float in (0, infinity), optional (default=np.finfo(np.float64).eps*10000)
        The minimum range for a feature to be considered worth splitting. If
        all features have ranges smaller than this, no more splits will be made.
        This is put in place to deal with floating point errors where all points
        can land on the same side of a split.
    """

    def __init__(self, n_estimators=100, max_samples=256, max_node_depth=None, threshold=0.7, contamination_pct=.01, bootstrap=False, random_features=False, float_min=np.finfo(np.float64).eps * 10000):
        self._n_estimators = n_estimators
        self.max_samples = max_samples
        self.threshold = threshold
        self.contamination_pct = contamination_pct
        self.bootstrap = bootstrap
        self.random_features = random_features
        self.float_min = float_min
        self.max_node_depth = max_node_depth

    def fit(self, X, calculate_threshold=False):
        '''Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        calculate_threshold : boolean, optional (default=False)
            If True, the treshold for an outlier used in the predict
            function is calculated by computing the scores of the training
            data and choosing the threshold to be the number at which
            the specified contamination percent proportion of samples are
            considered outliers.  Otherwise, the threshold specified when
            initializing the forest is used.

        Returns
        -------
        self : object
            Returns self.
        '''
        self._n = X.shape[0]
        self.max_samples_ = _get_max_samples(self.max_samples, self._n)
        avg_depth = average_path_length(self.max_samples_)
        # child_rightfit
        self.trees = [
            RandomCutTree(
                random_features=self.random_features,
                float_min=self.float_min,
                max_node_depth=self.max_node_depth,
                avg_depth=avg_depth
            ).fit(_sample(X, self.max_samples_, self.bootstrap))
            for i in range(self._n_estimators)
        ]

        # calculate the threshold for outliers by evaluating the scores of the
        # training set
        if calculate_threshold:
            self.threshold = \
                -stats.scoreatpercentile(-self.decision_function(X),
                                         100. * (1. - self.contamination_pct))

        return self

    def predict(self, X):
        '''Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        '''
        scores = self.decision_function(X)
        is_inlier = np.zeros(X.shape[0], dtype=int)
        is_inlier[scores <= self.threshold] = 1
        return is_inlier

    def decision_function(self, X, transformed=True):
        '''Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        a function of the mean of its depth across all trees in the forest.
        The depths of a leaf is equivalent to the number of splittings required
        to isolate this point.  In case of several identical observations in the
        leaf, the average path length required to separate that many points is
        added to the length.  These default is to apply a transformation to the
        depths of the leaves to return a more easily interpretable score in the
        range (0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        transformed : boolean, optional (default=False)
            If True, the score is transformed to lie in the range (0, 1) where
            high scores represent normal points and low scores represent anomalies.
            If False, the average tree depth is returned.

        Returns
        -------
        scores : array of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        '''
        depths = np.column_stack([tree.decision_function(X)
                                  for tree in self.trees])
        mean_depths = np.mean(depths, axis=1)

        if transformed:
            scores = np.power(2., -mean_depths /
                              average_path_length(self.max_samples_))
        else:
            scores = mean_depths

        return 0.5 - scores

    def add_point(self, x):
        '''add a point to the forest

        This method streams a point into the forest.  The point is added and
        and existing point is removed randomly based on reservoir sampling.

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape (n_features,)
            An individual training sample
        '''
        self._n += 1
        add_point_probability = np.float(self.max_samples_) / self._n
        should_add_points = np.random.binomial(
            1, add_point_probability, size=self._n_estimators)
        for tree, should_add_point in zip(self.trees, should_add_points):
            if should_add_point == 1:
                if tree._X.shape[0] > 1:
                    tree.remove_point()
                tree.add_point(x)


class RandomCutTree(object):
    '''Random Cut Tree

    An individual tree in a random cut forest.

    Parameters
    ----------
    random_features : boolean, optional (default=False)
        If True, the feature a tree splits on is chosen uniformaly at random
        (among all features for which the range is nonzero) rather than
        proportional to its range.

    float_min : float in (0, infinity), optional (default=np.finfo(np.float64).eps*10000)
        The minimum range for a feature to be considered worth splitting. If
        all features have ranges smaller than this, no more splits will be made.
        This is put in place to deal with floating point errors where all points
        can land on the same side of a split.
    '''

    def __init__(self, avg_depth=None, max_node_depth=None, random_features=False, float_min=np.finfo(np.float64).eps * 10000):
        self.random_features = random_features
        self.float_min = float_min
        self.max_node_depth = max_node_depth
        self._avg_depth = avg_depth

    def _update_vector(self, X):
        self._X = X
        self._X.setflags(write=False)  # Ensure the base array doesn't change

    def fit(self, X):
        '''fit a random cut tree'''
        self._update_vector(X=X)

        feature_mins = np.min(self._X, axis=0)
        feature_maxs = np.max(self._X, axis=0)
        feature_data = (feature_mins, feature_maxs)

        self._root = TreeNode(
            n=self._X.shape[0],
            parent=None,
            feature_data=feature_data,
            random_features=self.random_features,
            float_min=self.float_min
        )
        self.split_node(node=self._root, X=X)
        return self

    def split_node(self, node, X, current_depth=0):
        if self.max_node_depth is not None and current_depth >= self.max_node_depth:
            node._X = X
        elif not node.is_leaf:
            (child_left, child_right, X_left, X_right) = node._split(X)
            self.split_node(child_left, X_left, current_depth + 1)
            self.split_node(child_right, X_right, current_depth + 1)
        else:
            node._X = X

    def decision_function(self, X):
        '''return the decision function (the depth) for each point in the tree'''
        return np.array([self._root.get_depth(x, self.max_node_depth, self._avg_depth) for x in X])

    def add_point(self, x):
        '''insert a new point into the tree'''
        self._update_vector(np.row_stack([self._X, x]))
        self._root = self.add_new_node(
            node=self._root, point=x, current_depth=0)

    def remove_point(self):
        '''forget a point at random from the tree'''

        choice = np.random.choice(self._root.num_points())
        point_to_forget = self._X[choice]

        current_point_index = np.argmax(
            np.all(self._X == point_to_forget, axis=1))
        self._X = np.delete(self._X, current_point_index, 0)
        self._root = self.remove_node(node=self._root, point=point_to_forget)

    def add_new_node(self, node, point, current_depth=0):
        '''insert a new point into the tree'''
        if node is None:
            return TreeNode(n=1, parent=None, feature_data=(point, point))

        # pick a split for if the new point is included
        (feature_mins, feature_maxs) = node.get_feature_ranges(point)
        feature_ranges = feature_maxs - feature_mins

        (alone_left, alone_right, split_data) = \
            node.get_new_node_position(feature_ranges, feature_mins)

        if self.max_node_depth is not None and current_depth >= self.max_node_depth:
            node.feature_data = (feature_mins, feature_maxs)
            node.point_added()

        # Check if the new split separates the new point from the node
        # If it does we will use this new split
        elif alone_left or alone_right:
            child = node
            node = TreeNode(
                n=child.num_points() + 1,
                parent=node.parent,
                feature_data=(feature_mins, feature_maxs),
                split_data=split_data,
                random_features=child.random_features,
                float_min=child.float_min
            )

            if alone_left:
                node.child_right = child
                node.child_left = TreeNode(
                    n=1,
                    parent=node,
                    feature_data=(point, point),
                    is_left_of_parent=True,
                    float_min=node.float_min
                )

            else:
                node.child_left = child
                node.child_right = TreeNode(
                    n=1,
                    parent=node,
                    feature_data=(point, point),
                    is_left_of_parent=False,
                    float_min=node.float_min
                )

        else:
            # update node statistics to include new point
            node.feature_data = (feature_mins, feature_maxs)
            node.point_added()
            # find the new point's leaf starting at the node below
            # note that we use the old splitting criterion

            if node.is_point_left(point):
                node.child_left = self.add_new_node(
                    node.child_left, point, current_depth + 1)
                node.child_left.parent = node
            else:
                node.child_right = self.add_new_node(
                    node.child_right, point, current_depth + 1)
                node.child_right.parent = node

        return node

    def remove_node(self, node, point):
        '''forget a point from the tree'''
        if node.is_leaf:
            return self._remove_from_leaf(node, point)

        node.point_removed()
        prev_parent = node.parent
        if node.is_point_left(point):
            node.child_left = self.remove_node(node.child_left, point)
            if node.child_left is None:
                node = node.child_right
                node.parent = prev_parent

        else:
            node.child_right = self.remove_node(node.child_right, point)

            if node.child_right is None:
                node = node.child_left
                node.parent = prev_parent

        return node

    def _remove_from_leaf(self, node, point):
        if node.num_points() > 1:  # array is duplicate points
            node.point_removed()
            return node

        return None


class TreeNode(object):
    '''Tree Node

    An individual node in a random cut tree.
    '''

    def __init__(self, n, parent, is_left_of_parent=None, feature_data=None, split_data=None, child_left=None, child_right=None, initialize=False, random_features=False, float_min=np.finfo(np.float64).eps * 10000):
        self._n = n
        self.parent = parent
        self.is_left_of_parent = is_left_of_parent
        self.random_features = random_features
        self.float_min = float_min

        self.feature_data = feature_data
        self.child_left = child_left
        self.child_right = child_right
        self._X = None

        self._set_split()

    @property
    def is_leaf(self):
        """ Ensures we correctly check if a node is a leaf. """
        return np.all(self.feature_ranges < self.float_min)

    @property
    def feature_ranges(self):
        return self.feature_maxs - self.feature_mins

    @property
    def feature_mins(self):
        return self.feature_data[0]

    @property
    def feature_maxs(self):
        return self.feature_data[1]

    @feature_mins.setter
    def feature_mins(self, val):
        self.feature_data = (val, self.feature_data[1])

    @feature_maxs.setter
    def feature_maxs(self, val):
        self.feature_data = (self.feature_data[0], val)

    def is_point_left(self, point):
        if point[self.split_feature] < self.split_threshold:
            return True
        return False

    def get_new_node_position(self, feature_ranges, feature_mins):
        if np.all(feature_ranges < self.float_min):
            # Node is a leaf and we don't want to continue down this path
            return (False, False, (None, None))

        split_feature = self._sample_split_feature(feature_ranges)
        split_threshold = self._sample_split_threshold(
            feature_ranges, feature_mins, split_feature)

        alone_left = self.feature_mins[split_feature] > split_threshold

        alone_right = self.feature_maxs[split_feature] < split_threshold

        return (alone_left, alone_right, (split_feature, split_threshold))

    def point_added(self):
        """ Wrapper so the number of points isn't directly exposed """
        self._n += 1

    def num_points(self):
        """ Wrapper to get the number of points stored in a node. """
        return self._n

    def point_removed(self):
        """ Wrapper to decrement the number of points """
        self._n -= 1

    def get_feature_ranges(self, point):
        """ Compute the feature ranges given a new point """
        feature_mins = np.minimum(self.feature_mins, point)
        feature_maxs = np.maximum(self.feature_maxs, point)
        return (feature_mins, feature_maxs)

    def _set_split(self):
        '''set splitting feature and threshold'''
        split_feature = self._sample_split_feature(self.feature_ranges)
        split_threshold = self._sample_split_threshold(
            self.feature_ranges, self.feature_mins, split_feature)
        self.split_feature = split_feature
        self.split_threshold = split_threshold

    def _split(self, X):
        '''split based on feature and threshold'''
        is_left = X[:, self.split_feature] < self.split_threshold
        X_left = X[is_left]
        X_right = X[np.logical_not(is_left)]

        feature_data_left = (np.min(X_left, axis=0), np.max(X_left, axis=0))
        feature_data_right = (np.min(X_right, axis=0), np.max(X_right, axis=0))

        self.child_left = TreeNode(
            n=X_left.shape[0],
            parent=self,
            feature_data=feature_data_left,
            is_left_of_parent=True,
            float_min=self.float_min
        )

        self.child_right = TreeNode(
            n=X_right.shape[0],
            parent=self,
            feature_data=feature_data_right,
            is_left_of_parent=False,
            float_min=self.float_min
        )

        return (self.child_left, self.child_right, X_left, X_right)

    def _sample_split_feature(self, feature_ranges):
        '''sample the feature to split on'''
        if self.random_features:
            is_feature_varying = np.array(
                feature_ranges > self.float_min, dtype=float)
            return np.flatnonzero(np.random.multinomial(1, is_feature_varying / np.sum(is_feature_varying)))[0]
        else:
            if np.sum(feature_ranges) == 0:
                return np.flatnonzero(np.random.multinomial(1, feature_ranges))[0]
            return np.flatnonzero(np.random.multinomial(1, feature_ranges / np.sum(feature_ranges)))[0]

    def _sample_split_threshold(self, feature_ranges, feature_mins, split_feature):
        '''sample the splitting threshold of a node'''
        return np.random.rand() * feature_ranges[split_feature] + feature_mins[split_feature]

    def get_depth(self, x, max_node_depth=None, avg_depth=None, current_depth=0):
        '''calculate number of nodes to isolate a point'''
        if max_node_depth is not None and current_depth >= max_node_depth:
            return avg_depth
        if self.is_leaf:
            return current_depth + average_path_length(self._n)
        elif x[self.split_feature] < self.split_threshold:
            return self.child_left.get_depth(x, max_node_depth, avg_depth, current_depth + 1)
        else:
            return self.child_right.get_depth(x, max_node_depth, avg_depth, current_depth + 1)


def _sample(X, num_samples, replace=False):
    '''take a random sample of X'''
    n = X.shape[0]
    return X[np.random.choice(n, num_samples, replace)]


def _get_max_samples(max_samples, n):
    '''get the number of samples for each tree'''
    if isinstance(max_samples, int):
        if max_samples > n:
            warn("max_samples (%s) is greater than the "
                 "total number of samples (%s). max_samples "
                 "will be set to n_samples for estimation."
                 % (max_samples, n))
            max_samples_ = n
        else:
            max_samples_ = max_samples
    else:  # float
        if not (0. < max_samples <= 1.):
            raise ValueError("max_samples must be in (0, 1]")
        max_samples_ = int(max_samples * n)

    return max_samples_


def harmonic_approx(n):
    '''Returns an approximate value of n-th harmonic number.'''
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209
    return gamma + np.log(n) + 0.5 / n - 1. / (12 * np.square(n)) + 1. / (120 * np.power(n, 4))


def average_path_length(n):
    '''Returns the average path length of an unsuccessful search in a BST'''
    return 2. * harmonic_approx(n - 1) - 2. * (n - 1.) / n if n > 1 else 0
