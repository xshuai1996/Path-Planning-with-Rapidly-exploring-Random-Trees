import numpy as np


class RRT():

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.
            :param statee: np.array of new child node's statee
            :returns: child Node object.
            """
            child = RRT.Node(state=state, parent=self)
            self.children.append(child)
            return child


    def __init__(self,
                 start_state,
                 goal_state,
                 dim_ranges,
                 obstacles=[],
                 step_size=0.05,
                 max_iter=1000):
        """
        :param start_state: Array-like representing the start state.
        :param goal_state: Array-like representing the goal state.
        :param dim_ranges: List of tuples representing the lower and upper
            bounds along each dimension of the search space.
        :param obstacles: List of CollisionObjects.
        :param step_size: Distance between nodes in the RRT.
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = RRT.Node(start_state, None)
        self.goal = RRT.Node(goal_state, None)
        self.dim_ranges = dim_ranges
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter

        if (self.start.state.shape != self.goal.state.shape):
            raise AssertionError("Start and Goal states do not match dimension!")

    def build(self):
        """
        Build an RRT.
        :returns: A list of states that create a path from start to
            goal on success.
        """
        for k in range(self.max_iter):
            rand_state = self._get_random_sample()
            near_node = self._get_nearest_neighbor(rand_state)
            new_node = self._extend_sample(rand_state, near_node)
            if new_node and self._check_for_completion(new_node):
                self.goal.parent = new_node
                path = self._trace_path_from_start()
                return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))

    def _get_random_sample(self):
        """
        Uniformly samples the search space.
        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        low_bounds = [i[0] for i in self.dim_ranges]
        high_bounds = [i[1] for i in self.dim_ranges]
        ret = np.random.uniform(low=low_bounds, high=high_bounds)
        return ret

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.
        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        min_dist = float('inf')
        min_node = None
        for node in iter(self.start):
            dist = np.sqrt(np.sum(np.square(node.state - sample)))
            if dist < min_dist:
                min_dist = dist
                min_node = node
        return min_node

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)
        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        direction = (sample - neighbor.state) / np.sqrt(np.sum(np.square(sample - neighbor.state)))
        ext_pos = neighbor.state + direction * self.step_size
        if self._check_for_collision(ext_pos):
            return None
        else:
            new_node = neighbor.add_child(ext_pos)
            return new_node

    def _check_for_completion(self, node):
        """
        Check whether node is within self.step_size distance of the goal.
        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        dist = np.sqrt(np.sum(np.square(node.state - self.goal.state)))
        if dist < self.step_size:
            return True
        else:
            return False

    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.
        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        if node is not None:
            reversed_path = [node]
        else:
            reversed_path = [self.goal]
        while reversed_path[-1].parent is not None:
            reversed_path.append(reversed_path[-1].parent)
        path = [n.state for n in reversed_path]
        return path[::-1]

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.
        :returns: A boolean value indicating that sample is in collision.
        """
        for obs in self.obstacles:
            if obs.in_collision(sample) is True:
                return True
        return False

