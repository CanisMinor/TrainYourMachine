class TreeNode():
    def __init__(self, index, parent):
        self.parent = parent
        self.children = []
        self.index = index

    def add_child(self, child, value):
        self.children.append((child, value))



class DecisionTree():
    def __init__(self):
        self.root = TreeNode(0, 0)
        self.nodes = dict
        self.nodes[self.root.index] = self.root
        self.max_index = 0
        self.layers = 0
        self.trained = False


    def add_node(self, parent):
        self.nodes[self.max_index + 1] = TreeNode(self.max_index + 1, parent)
        self.max_index = self.max_index + 1



    def train(self):
        #implement training routine, eventually...
        self.trained = True

