from anytree import Node
import numpy as np


class dt_node(Node):
    def __init__(self, id, parent = None):
        Node.__init__(self, id, parent)
        self.id = id  # The node value
        self.name = None
        self.left_node_id = -1   #  Left child
        self.right_node_id = -1  # Right child
        self.feature = -1
        self.threshold = None
        self.values = -1 


def build_tree(tree_, feature_names = None):
    feature = tree_.feature
    threshold = tree_.threshold
    values = tree_.value
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaf[node_id] = True         
    m = tree_.node_count  
    assert (m > 0), "Empty tree"
    
    def extract_data(idx, root = None, feature_names = None):
        i = idx
        assert (i < m), "Error index node"
        if (root is None):
            node = dt_node(i)
        else:
            node = dt_node(i, parent = root)
        if is_leaf[i]:
            node.values = np.argmax(values[i])
        else:
            node.feature = feature[i]
            if (feature_names is not None):
                node.name = feature_names[feature[i]]
            node.threshold = threshold[i]
            node.left_node_id = children_left[i]
            node.right_node_id = children_right[i]
            extract_data(node.left_node_id, node, feature_names) #feat < threshold ( < 0.5 False)
            extract_data(node.right_node_id, node, feature_names) #feat >= threshold ( >= 0.5 True)            
        return node
    root = extract_data(0, None, feature_names)
    return root

def count_nodes(root):
    def count(node):
        if len(node.children):
            return sum([1+count(n) for n in node.children])
        else:
            return 0
    m = count(root) + 1
    return m


def predict_tree(node, sample):
    if (len(node.children) == 0):
        # leaf
        return node.values
    else:
        feature_branch = node.feature
        sample_value = sample[feature_branch]
        assert(sample_value is not None)
        if(sample_value <= node.threshold):
            return predict_tree(node.children[0], sample)
        else:
            return predict_tree(node.children[1], sample)


class Forest:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """
    def __init__(self, rf, feature_names = None):
        self.trees = [ build_tree(dt.tree_, feature_names) for dt in rf.estimators()]
        self.sz = sum([dt.tree_.node_count for dt in rf.estimators()])
        self.md = max([dt.tree_.max_depth for dt in rf.estimators()])
        assert([dt.tree_.node_count for dt in rf.estimators()] == [count_nodes(dt) for dt in self.trees])
        
    def predict_inst(self, inst):
        scores = [predict_tree(dt, inst) for dt in self.trees]
        scores = np.asarray(scores)
        maj = np.argmax(np.bincount(scores))
        return maj
    
    """
    def predict(self, samples):       
        predictions = []
        print("#Trees: ", len(self.trees))
        for sample in np.asarray(samples):
            scores = []
            for i,t in enumerate(self.trees):
                s = predict_tree(t, sample)
                scores.append((s))
            scores = np.asarray(scores)
            predictions.append(scores)
        predictions = np.asarray(predictions)    
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
        return maj
    """
