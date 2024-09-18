# Robust Random Cut Forests

This repo contains an implementation of the `Robust Random Cut Forest` anomaly detection model. This model attempts to find anomalies by seeking out points whose structure is not consistent with the rest of the data set. The `random_cut_forest` folder contains the `RandomCutForest` algorithm while the `notebooks` folder contains Jupyter notebooks showing examples leveraging the module.

## Contributing
If you want to contribute to this repo simply submit a pull request.

## Getting Started

### Installation
To install the package you can do any of the following:

- Run the command `pip install ...`

### Using RobustRandomCutForests
Using a RobustRandomCutForest to classify potential anomalies in your data is simple. Assuming you already have a vector of data stored in `X` you would run the following:

```python
from robust_random_cut_forest import robust_random_cut_forest
forest = robust_random_cut_forest.RobustRandomCutForest()
forest = forest.fit(X)
```

From there you can choose to get the normalized depths of each point within the forest by calling `average_depths` or have the forest label potential anomalies by calling `predict`:

```python
depths = forest.decision_function(X)
labels = forest.predict(X)
```

The function `decision_function` will return an array with numbers ranging from zero to one. The lower the number the more anomalous the point appears (this is how sklearn implements scoring). By default any points that are given a score of `0.3` are labelled as anomalies. To stream new points into your forest simply call the `add_point` method:

```python
# Given an array of points....
for point in points:
    forest.add_point(point)
depths = forest.decision_function(points)
labels = forest.predict(points)
```

## Testing
All tests are written using `pytest`. Simply `pip install pytest` to be able to run tests. All tests are located under the `tests` folder. Any new tests are always welcome!

## Articles
* For more information on Robust Random Cut Forests, see Guha et al.'s 2016 paper
which can be located [here](http://jmlr.org/proceedings/papers/v48/guha16.pdf).
* The original isolation forest paper can be found [here](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).
* Isolation Forests have been implemented in [sklearn](http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.IsolationForest.html)


## Contact
<mr.navdeepgill@gmail.com>

