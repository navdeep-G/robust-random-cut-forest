""" Assist with printing data to the screen """


def dump_node_data(node):
    """ Dump the data given a node """
    print
    print('Node:')
    print(node.num_points())
    print(node.feature_mins)
    print(node.feature_maxs)
    print

    try:
        print('Left:')
        print(node.child_left.num_points())
        print(node.child_left.feature_mins)
        print(node.child_left.feature_maxs)
        print('is leaf? {}'.format(node.child_left.is_leaf))

    except Exception:
        print('No left child')

    print

    try:
        print('Right:')
        print(node.child_right.num_points())
        print(node.child_right.feature_mins)
        print(node.child_right.feature_maxs)
        print('is leaf? {}'.format(node.child_right.is_leaf))

    except Exception:
        print('No right child')
