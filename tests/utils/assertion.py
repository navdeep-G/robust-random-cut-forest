""" Common functions to aid with asserting during tests """


def assert_equals_list(ls_one, ls_two):
    """Calls assert_equal on each elememt of a list.
    Assumes they are both the same length and sorted similarily.
    """
    for idx, elem in enumerate(ls_one):
        assert elem == ls_two[idx]
