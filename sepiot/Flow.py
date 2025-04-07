class Flow:
    """
    Flow is a class that represents a flow of data through a series of nodes.
    """

    def __init__(self):
        """
        Initialize the Flow object.
        """
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        """
        Add a node to the flow.

        Args:
            node: The node to add.
        """
        self.nodes.append(node)

    def add_edge(self, edge):
        """
        Add an edge to the flow.

        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)