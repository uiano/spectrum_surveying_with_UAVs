import numpy as np
import abc
import matplotlib.pyplot as plt
from utilities import FifoUniqueQueue, mat_argmax
from scipy.ndimage import convolve
from skimage.measure import label
import random

from queue import Queue, PriorityQueue
from IPython.core.debugger import set_trace


class RoutePlanner():
    name_on_figs = "(name not set by child class)"

    def __init__(self,
                 grid=None,
                 initial_location=None,
                 dist_between_measurements=None,
                 num_measurement_to_update_destination=None,
                 debug_code=0):

        assert grid
        #assert dist_between_measurements

        self.grid = grid
        self.debug_code = debug_code
        self.dist_between_measurements = dist_between_measurements
        # Necessary to save `initial_location` to reset().
        self.initial_location = initial_location
        self.destination_point = None  # to store latest destination point
        self.num_measurement_to_update_destination = num_measurement_to_update_destination
        self.measurement_count=0
        self.avg_previous_uncertainty = None
        self.reset()

        if self.debug_code == 1:
            self.l_waypoints = [self.previous_waypoint]

            self.l_meas_locs = []
            num_locations = 50
            for ind_location in range(num_locations):
                meas_location = self.next_measurement_location(None)
                self.l_meas_locs.append(meas_location)

            m_waypoints = np.array(self.l_waypoints)
            plt.plot(m_waypoints[:, 0],
                     m_waypoints[:, 1],
                     '+-',
                     label="waypoints",
                     color="black")

            m_meas_locs = np.array(self.l_meas_locs)
            plt.plot(m_meas_locs[:, 0],
                     m_meas_locs[:, 1],
                     'o',
                     label="measurements",
                     color="blue")
            plt.show()

    def reset(self):
        """ Used e. g. in Monte Carlo"""
        # self.measurement_count=0
        if self.initial_location is not None:
            self.previous_waypoint = self.initial_location
        else:
            self.previous_waypoint = self.grid.random_point_in_the_area()

        # Always take a measurement at the starting point. --> already done
        # self.l_next_measurement_locations = [self.previous_waypoint]
        self.remaining_dist = self.dist_between_measurements
        self.l_next_measurement_locations = []
        self.previous_waypoint = None
        self.l_next_waypoints = []

    def next_measurement_location(self, d_map_estimate, m_building_metadata=None):
        self.measurement_count += 1
        while not self.l_next_measurement_locations:
            # Obtain next measurement location
            next_waypoint = self.next_waypoint(d_map_estimate, m_building_metadata=m_building_metadata)

            if self.dist_between_measurements is None:
                # take measurements at grid points
                self.l_next_measurement_locations = [next_waypoint]
                self.previous_waypoint = next_waypoint
                print("No dist", self.previous_waypoint)
            else:
                if self.previous_waypoint is None:
                    self.previous_waypoint = next_waypoint
                    print(self.previous_waypoint)
                    return self.previous_waypoint

                self.l_next_measurement_locations, self.remaining_dist = \
                    self.measurement_locations_from_waypoints(
                        self.previous_waypoint,
                        next_waypoint,
                        self.remaining_dist)

                self.previous_waypoint = next_waypoint

        next_location = self.l_next_measurement_locations.pop(0)

        # Update destination every self.num_measurement_to_update_destination measurements
        if self.num_measurement_to_update_destination and \
                self.measurement_count % self.num_measurement_to_update_destination == 0:
            # self.initial_location = next_location
            self.l_next_waypoints=[]
            self.l_next_measurement_locations = []

        # set_trace()
        # print(f"route_planning: next_location = {next_location}")

        return next_location

    def measurement_locations_from_waypoints(self, previous_waypoint,
                                             next_waypoint, remaining_dist):
        """Args:
        Returns:

        - "l_locations": list of locations between `previous_waypoint`
          and `next_waypoint` if any. First location is at distance
          `remaining_dist` from `previous_waypoint`. `next_waypoint`
          is never included.
        - "remaining_dist": distance to next measurement point after
          the UAV is at point `next_waypoint`.
        """
        dist_waypoints = np.linalg.norm(next_waypoint - previous_waypoint)

        if dist_waypoints <= remaining_dist:
            return [], remaining_dist - dist_waypoints

        num_points = np.ceil(
            (dist_waypoints - remaining_dist) / self.dist_between_measurements)

        v_distances = \
            remaining_dist + np.arange(num_points) * self.dist_between_measurements
        remaining_dist = self.dist_between_measurements - (dist_waypoints -
                                                           v_distances[-1])

        unit_vector = (next_waypoint - previous_waypoint) / dist_waypoints

        l_locations = [
            previous_waypoint + dist * unit_vector for dist in v_distances
        ]

        return l_locations, remaining_dist


    def shortest_path(self, m_node_costs=None, start=None,
                      destination=None, m_building_metadata=None):
        """Bellman-Ford algorithm, aka breath-first search (BFS)
        [bertsekas2005]. Provides a path with minimum cost."
        Arguments:
        `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
        cost of traversing the grid point (i,j). Its dimensions Ny, Nx
        define a rectangular grid.
        Returns:
        `l_path` : None if a path does not exist. Else, a list of grid point
        indices corresponding to a shortest path between `start` and
        `destination`, which are grid node indices. `start` is not
        included in the returned list unless start==destination, but
        `destination` is always included.
        """

        self.m_node_costs = m_node_costs  # Required by other methods

        def is_in_grid(t_inds):
            return (t_inds[0] < m_node_costs.shape[0]) \
                   and (t_inds[1] < m_node_costs.shape[1])

        assert is_in_grid(start)
        assert is_in_grid(destination)

        if start == destination:
            return [destination]

        queue = FifoUniqueQueue()  # OPEN queue in [bertsekas2005]. It
        # contains nodes that are candidates
        # for being in the shortest path to
        # other nodes.
        queue.put(start)
        m_cost = np.full(m_node_costs.shape, fill_value=np.infty, dtype=float)
        m_cost[start] = 0

        # keys: state. values: previous_state
        d_branches = {}
        while not queue.empty():

            current_point = queue.get()
            # print(f"point={current_point}")

            # TODO: _possible_actions should building into account
            for str_action in self._possible_actions(current_point, m_building_metadata):
                next_point = self._next_state(current_point, str_action)

                new_cost = m_cost[current_point] + self.transition_cost(
                    current_point, next_point)

                # UPPER in [bertsekas2005] is m_cost[destination]
                if new_cost < min(m_cost[next_point], m_cost[destination]):

                    d_branches[next_point] = current_point
                    m_cost[next_point] = new_cost

                    if next_point != destination:
                        queue.put(next_point)

        if m_cost[destination] < np.infty:
            if self.debug_code:
                print("Route found")
            state = destination
            l_path = [state]
            while d_branches[state] != start:
                previous_state = d_branches[state]
                l_path.append(previous_state)
                state = previous_state

            return l_path[::-1]

        else:
            """Route not found"""
            # "select random point outside building if there in no route found."
            # l_inidices = np.where(m_building_metadata == 0)
            # ind = np.random.randint(0, int(len(l_inidices[0])))
            # random_point_in_grid_indices = (l_inidices[0][ind], l_inidices[1][ind])
            # # set_trace()
            # print("Route not found")
            # return [random_point_in_grid_indices]
            return None

    def transition_cost(self, point_1, point_2):
        # approximates the integral of the cost

        dist = np.linalg.norm(np.array(point_2) - np.array(point_1))

        cost_1 = self.m_node_costs[point_1[0], point_1[1]]
        cost_2 = self.m_node_costs[point_2[0], point_2[1]]
        return dist / 2 * cost_1 + dist / 2 * cost_2


    def _possible_actions(self, state, m_building_metadata):
        """
        Arguments:
        `state`: tuple with the indices of the considered grid point.
        Returns:
        list of possible actions at state `state`.

        """

        max_row_index = self.m_node_costs.shape[0] - 1
        max_col_index = self.m_node_costs.shape[1] - 1

        l_actions = []
        for str_action in self.d_actions:

            candidate_entry = self._next_state(state, str_action)

            # Check if in grid
            if candidate_entry[0] >= 0 and \
                    candidate_entry[0] <= max_row_index and \
                    candidate_entry[1] >= 0 and \
                    candidate_entry[1] <= max_col_index:
                # TODO: check whether the point is inside the building or not
                if m_building_metadata[candidate_entry] != 1:
                    l_actions.append(str_action)

        return l_actions

    def _next_state(self, state, str_action):

        v_movement = self.d_actions[str_action]
        return (state[0] + v_movement[0], state[1] + v_movement[1])

    def destination_list_to_waypoint_list(self, destination_list,
                                          m_building_metadata):
        """
        Args:
            -`destination_list`: a list of indices of the grid points to be visited

        Returns:
            - `waypoint_list`: a list of adjacent grid points that defines a
            path that traverses all reachable points in `destination_list`.
        """
        # m_building_metadata[1, 0:3], m_building_metadata[0,2] = 1, 1
        # node cost to find the shortest path between waypoints
        m_node_costs = np.ones(shape=m_building_metadata.shape)
        m_node_costs[m_building_metadata == 1] = 0

        # get a matrix with connected region label and region value with
        # maximum connected points.
        m_connected_labels, label_with_max_freq= self.connected_labels(
            m_building_metadata=m_building_metadata)

        # get the first waypoint such that it lies in a region with
        # highest connected grid points outside of buildings.
        waypoint_inds = destination_list.pop(0)
        # check if the first point is outside buildings and in a region with maximum reachable points.
        while m_building_metadata[waypoint_inds] == 1 or \
                m_connected_labels[waypoint_inds]!= label_with_max_freq:
            waypoint_inds = destination_list.pop(0)

        l_inds_waypoints = [waypoint_inds]
        # self.previous_waypoint = self.grid.indices_to_point(l_inds_waypoints[-1])

        while destination_list:
            next_waypoint_inds = destination_list.pop(0)
            # if m_building_metadata[next_waypoint_inds] !=1:
            #     l_inds_waypoints.append(next_waypoint_inds)
            # else:
            # # check if the point is outside buildings
            # # count_waypoint_inside_building = 0
            while m_building_metadata[next_waypoint_inds] == 1:
                if destination_list:
                    next_waypoint_inds = destination_list.pop(0)
                # count_waypoint_inside_building += 1
                else:
                    break
            # point inside building find way_points using shortest path algorithm
            # if count_waypoint_inside_building>=1:

            t_inds_current = l_inds_waypoints[-1]
            t_inds_destination = next_waypoint_inds
            l_shortest_path_inds = self.shortest_path(m_node_costs, t_inds_current,
                                                      t_inds_destination, m_building_metadata)
            if l_shortest_path_inds is not None:
                l_inds_waypoints = l_inds_waypoints + l_shortest_path_inds

            # l_shortest_path_inds = []
            # while not l_shortest_path_inds:
            #     t_inds_current = l_inds_waypoints[-1]
            #     t_inds_destination = next_waypoint_inds
            #
            #     l_shortest_path_inds = self.shortest_path(m_node_costs, t_inds_current,
            #                                               t_inds_destination, m_building_metadata)
            #
            #     # if the point is not reachable, shortest path returns empty l_path_inds
            #     # then find next way point
            #     if l_shortest_path_inds is None:
            #         next_waypoint_inds = destination_list.pop(0)
            #         while m_building_metadata[next_waypoint_inds] == 1:
            #             if not destination_list:
            #                 print("Not reachable way points")
            #                 set_trace()
            #             next_waypoint_inds = destination_list.pop(0)

                # l_inds_waypoints = l_inds_waypoints + l_shortest_path_inds

        l_next_waypoints = [
            self.grid.indices_to_point(inds) for inds in l_inds_waypoints]

        return l_next_waypoints

    def connected_labels(self, m_building_metadata=None,
                         ):
        """
        Returns:
            -`m_connected_labels`: a matrix of size Ny x Nx whose (i,j)-th entry is
                the label of the connected components.

            -`label_with_max_freq`: an integer that represents a label with
                highest connected points or reachable points
        """
        # get the labels for connected regions.
        m_connected_labels, num_labels = label(input=m_building_metadata,
                                               background=1,  # set building locations as a background
                                               return_num=True,
                                               connectivity=2,  # consider diagonal as well
                                               )
        # count the frequency of occurrence of labels except for the background
        v_labels_freq = np.bincount(m_connected_labels[m_connected_labels != 0].flat)

        # label with maximum frequency, i.e. label with maximum reachable points.
        label_with_max_freq = np.argmax(v_labels_freq)

        return m_connected_labels, label_with_max_freq

    @abc.abstractmethod
    def next_waypoint(self, d_map_estimate, m_building_metadata=None):
        pass

    def plot_path(self,
                  start,
                  l_path,
                  axis=None,
                  m_node_costs=None,
                  label="",
                  color="white"):

        if axis is None:
            fig, axis = plt.subplots(nrows=1, ncols=1)

        if m_node_costs is not None:
            im = axis.imshow(
                #            im = plt.imshow(
                m_node_costs,
                # interpolation='spline16',
                cmap='jet',
                # extent=[0, m_node_costs.shape[1], m_node_costs.shape[0], 0],
                extent=np.array(
                    [0, m_node_costs.shape[1], m_node_costs.shape[0], 0]) -
                       0.5,
                # origin="upper"
                vmax=1,  # m_uncertainty_map.max(),
                vmin=0,  # m_uncertainty_map.min()
            )
            # fig.colorbar(im)
            plt.colorbar(im)

        m_path = np.array([start] + l_path)
        axis.plot(
            # plt.plot(
            m_path[:, 1],
            m_path[:, 0],
            '+-',
            label=label,
            color=color,
        )

        axis.plot(
            m_path[0, 1],
            m_path[0, 0],
            'o',
            label=label,
            color=color,
        )

        return axis


# class MinimumCostPlanner(RoutePlanner):
#     name_on_figs = "Min. cost planner"
#
#     d_actions = {
#         "UP": (-1, 0),
#         "DOWN": (1, 0),
#         "LEFT": (0, -1),
#         "RIGHT": (0, 1),
#         "UPLEFT": (-1, -1),
#         "UPRIGHT": (-1, 1),
#         "DOWNLEFT": (1, -1),
#         "DOWNRIGHT": (1, 1),
#     }
#
#     l_next_waypoints = []
#
#     def __init__(self,
#                  metric=None,  # can be "power_variance" or "service_entropy"
#                  grid=None,
#                  smoothing_constant = 0.5,
#                  **kwargs):
#
#         assert metric
#         assert grid
#
#         self.smoothing_constant = smoothing_constant
#         self.metric = metric
#         self.name_on_figs = f"{self.name_on_figs} ({metric})"
#         # number of times at each grid point
#         self.m_visited_points = np.zeros((grid.num_points_y, grid.num_points_x))
#
#         super().__init__(grid=grid, **kwargs)
#
#     def next_waypoint(self, d_map_estimate):
#
#         if not self.l_next_waypoints:
#
#             t_inds_current = self.grid.nearest_gridpoint_inds(
#                 self.previous_waypoint)
#
#             # Choose a destination
#             t_inds_destination, m_uncertainty = self.next_destination(t_inds_current, d_map_estimate)
#             # To plot the destination point
#             self.destination_point = self.grid.indices_to_point(t_inds_destination)
#             # Find shortest path
#             m_node_costs = 1 / (m_uncertainty + 0.01)
#             l_path_inds = self.shortest_path(m_node_costs, t_inds_current,
#                                              t_inds_destination)
#
#             # Turn indices to coordinates
#             self.l_next_waypoints = [
#                 self.grid.indices_to_point(inds) for inds in l_path_inds
#             ]
#
#             if self.debug_code == 2:
#                 self.plot_path(t_inds_current, l_path_inds, m_node_costs=1 / m_node_costs)
#                 plt.show()
#                 # set_trace()
#
#         if not self.l_next_waypoints:
#             set_trace()
#
#         return self.l_next_waypoints.pop(0)
#
#     def next_destination(self, t_inds_current, d_map_estimate):
#         """Returns the indices of the next destination grid point."""
#
#         if self.metric == "power_variance":
#             m_uncertainty = np.sum(d_map_estimate["t_power_map_norm_variance"], 0)
#         elif self.metric == "service_entropy":
#             m_uncertainty = np.sum(d_map_estimate["t_service_map_entropy"], 0)
#             # m_uncertainty = np.max(d_map_estimate["t_service_map_entropy"],0)
#         else:
#             raise Exception("Invalid metric")
#
#         if self.avg_previous_uncertainty is None:
#             self.avg_previous_uncertainty = m_uncertainty
#         else:
#             self.avg_previous_uncertainty = self.smoothing_constant * m_uncertainty + \
#                                             (1-self.smoothing_constant) * self.avg_previous_uncertainty
#
#         m_uncertainty = self.avg_previous_uncertainty
#
#         # Spatial filter
#         kernel = np.ones((3, 3))
#         m_mod_uncertainty = convolve(m_uncertainty, kernel)
#
#         # Modified uncertainty
#         m_mod_uncertainty = m_mod_uncertainty * (1 / (1 + self.m_visited_points))
#         m_mod_uncertainty[t_inds_current] = 0  # prevent remaining in the same point
#
#         t_inds_destination = mat_argmax(m_mod_uncertainty)
#         if t_inds_destination == t_inds_current:
#             set_trace()
#             # print("already at point of maximum uncertainty, choosing next waypoint randomly")
#             # t_inds_destination = (np.random.randint(0, high=m_uncertainty.shape[0]),
#             #                       np.random.randint(0, high=m_uncertainty.shape[0])
#             #                       )
#
#         self.m_visited_points[t_inds_destination] += 1
#
#         return t_inds_destination, m_mod_uncertainty
#
#     def shortest_path(self, m_node_costs=None, start=None, destination=None):
#         """Bellman-Ford algorithm, aka breath-first search (BFS)
#         [bertsekas2005]. Provides a path with minimum cost."
#         Arguments:
#         `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
#         cost of traversing the grid point (i,j). Its dimensions Ny, Nx
#         define a rectangular grid.
#         Returns:
#         `l_path` : None if a path does not exist. Else, a list of grid point
#         indices corresponding to a shortest path between `start` and
#         `destination`, which are grid node indices. `start` is not
#         included in the returned list unless start==destination, but
#         `destination` is always included.
#         """
#
#         self.m_node_costs = m_node_costs  # Required by other methods
#
#         def is_in_grid(t_inds):
#             return (t_inds[0] < m_node_costs.shape[0]) \
#                    and (t_inds[1] < m_node_costs.shape[1])
#
#         assert is_in_grid(start)
#         assert is_in_grid(destination)
#
#         if start == destination:
#             return [destination]
#
#         queue = FifoUniqueQueue()  # OPEN queue in [bertsekas2005]. It
#         # contains nodes that are candidates
#         # for being in the shortest path to
#         # other nodes.
#         queue.put(start)
#         m_cost = np.full(m_node_costs.shape, fill_value=np.infty, dtype=float)
#         m_cost[start] = 0
#
#         # keys: state. values: previous_state
#         d_branches = {}
#         while not queue.empty():
#
#             current_point = queue.get()
#             # print(f"point={current_point}")
#
#             for str_action in self._possible_actions(current_point):
#                 next_point = self._next_state(current_point, str_action)
#
#                 new_cost = m_cost[current_point] + self.transition_cost(
#                     current_point, next_point)
#
#                 # UPPER in [bertsekas2005] is m_cost[destination]
#                 if new_cost < min(m_cost[next_point], m_cost[destination]):
#
#                     d_branches[next_point] = current_point
#                     m_cost[next_point] = new_cost
#
#                     if next_point != destination:
#                         queue.put(next_point)
#
#         if m_cost[destination] < np.infty:
#             if self.debug_code:
#                 print("Route found")
#             state = destination
#             l_path = [state]
#             while d_branches[state] != start:
#                 previous_state = d_branches[state]
#                 l_path.append(previous_state)
#                 state = previous_state
#
#             return l_path[::-1]
#
#         else:
#             set_trace()
#             print("Route not found")
#             return None
#
#     def approximate_bfs_shortest_path(self, m_node_costs=None, start=None, destination=None):
#         """It is an approximate Dijkstra algorithm because we do not revisit
#         nodes. It is assumed that the distance stored for a visited
#         node is the distance of the sortest path. I think this makes
#         sense if our metric is Euclidean distance on a regular graph,
#         but not with node costs.
#         [Daniel] I adapted this from the flying car degree.
#         Arguments:
#         `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
#         cost of traversing the grid point (i,j). Its dimensions Ny, Nx
#         define a rectangular grid.
#         Returns:
#         `l_path` : None if a path does not exist. Else, a list of grid point
#         indices corresponding to a shortest path between `start` and
#         `destination`, which are grid node indices. `start` is not
#         included in the returned list unless start==destination, but
#         `destination` is always included.
#         """
#
#         self.m_node_costs = m_node_costs  # Required by other methods
#
#         def is_in_grid(t_inds):
#             return (t_inds[0] < m_node_costs.shape[0]) \
#                    and (t_inds[1] < m_node_costs.shape[1])
#
#         assert is_in_grid(start)
#         assert is_in_grid(destination)
#
#         if start == destination:
#             return [destination]
#
#         queue = PriorityQueue()
#         queue.put((0, start))
#         s_visited = {start}
#
#         # keys: state. values: previous_state
#         d_branches = {}
#         b_found = False
#         while not queue.empty():
#
#             cost_so_far, current_point = queue.get()
#             # print(f"point={current_point}")
#
#             for str_action in self._possible_actions(current_point):
#                 next_point = self._next_state(current_point, str_action)
#
#                 if next_point not in s_visited:
#                     s_visited.add(next_point)
#                     new_cost = cost_so_far + self.transition_cost(
#                         current_point, next_point)
#                     queue.put((new_cost, next_point))
#                     d_branches[next_point] = current_point
#
#                     if next_point == destination:
#                         b_found = True
#                         print("Route found")
#                         break
#
#             if b_found:
#                 break
#
#         if b_found:
#
#             state = destination
#             l_path = [state]
#             while d_branches[state] != start:
#                 previous_state = d_branches[state]
#                 l_path.append(previous_state)
#                 state = previous_state
#
#             return l_path[::-1]
#
#         else:
#             set_trace()
#             print("Route not found")
#             return None
#
#     def transition_cost(self, point_1, point_2):
#         # approximates the integral of the cost
#
#         dist = np.linalg.norm(np.array(point_2) - np.array(point_1))
#
#         cost_1 = self.m_node_costs[point_1[0], point_1[1]]
#         cost_2 = self.m_node_costs[point_2[0], point_2[1]]
#         return dist / 2 * cost_1 + dist / 2 * cost_2
#
#     def _possible_actions(self, state):
#         """
#         Arguments:
#         `state`: tuple with the indices of the considered grid point.
#         Returns:
#         list of possible actions at state `state`.
#
#         """
#
#         max_row_index = self.m_node_costs.shape[0] - 1
#         max_col_index = self.m_node_costs.shape[1] - 1
#
#         l_actions = []
#         for str_action in self.d_actions:
#
#             candidate_entry = self._next_state(state, str_action)
#
#             # Check if in grid
#             if candidate_entry[0] >= 0 and \
#                     candidate_entry[0] <= max_row_index and \
#                     candidate_entry[1] >= 0 and \
#                     candidate_entry[1] <= max_col_index:
#                 l_actions.append(str_action)
#
#         return l_actions
#
#     def _next_state(self, state, str_action):
#
#         v_movement = self.d_actions[str_action]
#         return (state[0] + v_movement[0], state[1] + v_movement[1])
#
#     def path_cost(self, m_node_costs, start, l_path):
#
#         # Returns the sum of the cost of all nodes in l_path.
#
#         self.m_node_costs = m_node_costs
#
#         cost = 0
#         if len(l_path) == 0:
#             return 0
#         prev_state = start
#         for state in l_path:
#             cost += self.transition_cost(prev_state, state)
#             prev_state = state
#
#         return cost


# class IndependentUniformPlanner(RoutePlanner):
#     name_on_figs = "Indep. Uniform Planner "
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def next_waypoint(self, d_map_estimate):
#         """Returns length-3 vector with the coordinates of a random point in
#         the area (not nec. on the grid)
#         """
#         # random coordinates in the range of x_coords, y_coords, and z_coords
#
#         waypoint = self.grid.random_point_in_the_area()
#
#         if self.debug_code == 1:
#             self.l_waypoints.append(waypoint)
#
#         return waypoint


class IndependentUniformPlanner(RoutePlanner):
    name_on_figs = "Indep. Uniform Planner "
    l_next_waypoints = []
    d_actions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UPLEFT": (-1, -1),
        "UPRIGHT": (-1, 1),
        "DOWNLEFT": (1, -1),
        "DOWNRIGHT": (1, 1),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l_next_measurement_locations = []
        self.previous_waypoint = None #[]

    def next_waypoint(self, d_map_estimate, m_building_metadata):
        """Returns length-3 vector with the coordinates of a random point in
        the area (not nec. on the grid)
        """
        # random coordinates in the range of x_coords, y_coords, and z_coords
        if not self.l_next_waypoints:
            v_inds = self.grid.random_grid_points_inds_outside_buildings(
                num_points=100, m_building_metadata=m_building_metadata)
            inds_row, inds_col = np.unravel_index(v_inds, shape=m_building_metadata.shape)
            l_inds_grid_waypoints = list(zip(inds_row, inds_col))
            # add (0,0) indices too
            l_inds_grid_waypoints = [(0,0), *l_inds_grid_waypoints]
            self.l_next_waypoints = self.destination_list_to_waypoint_list(destination_list=l_inds_grid_waypoints,
                                                   m_building_metadata=m_building_metadata)

            # set initial point to be previous_waypoint
            # self.previous_waypoint = self.l_next_waypoints[0]

        return self.l_next_waypoints.pop(0)


# class GridPlanner(RoutePlanner):
#     name_on_figs = "Grid Planner"
#     l_next_waypoints = []
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def next_waypoint(self, d_map_estimate):
#         if not self.l_next_waypoints:
#             # stay `self.dist_between_measurements/2` away from the limits
#             max_x = self.grid.max_x() - self.dist_between_measurements / 2
#             min_x = self.grid.min_x() + self.dist_between_measurements / 2
#
#             max_y = self.grid.max_y() - self.dist_between_measurements / 2
#             min_y = self.grid.min_y() + self.dist_between_measurements / 2
#
#             # Coordinates of the turning points:
#             eps = 1e-10  # to force arange to include the upper limit if needed
#             v_x_coords = np.arange(min_x, max_x + eps, step=self.dist_between_measurements)
#             # v_y_coords = np.arange(min_y,max_y+eps, step=self.dist_between_measurements)
#             v_y_coords = np.array([max_y, min_y])  # v_y_coords[::-1] # Flip it
#
#             # Now form the sequence of turning points
#             # Just repeat each entry of v_x_coords
#             v_seq_x_coords = np.vstack((v_x_coords, v_x_coords)).T.ravel()
#             # Similar for Y, but with a shift
#             v_seq_y_coords = np.tile(v_y_coords, int(np.ceil(len(v_x_coords) / 2)))
#             v_seq_y_coords = np.vstack((v_seq_y_coords, v_seq_y_coords)).T.ravel()
#             v_seq_y_coords = v_seq_y_coords[0:len(v_seq_x_coords)]
#             # v_seq_y_coords = np.concatenate( ([v_seq_y_coords[-1]] , v_seq_y_coords[0:-1]))
#             v_seq_y_coords = np.concatenate((v_seq_y_coords[1:], [v_seq_y_coords[0]]))
#
#             v_seq_z_coords = np.full(v_seq_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)
#
#             m_points = np.vstack((v_seq_x_coords, v_seq_y_coords, v_seq_z_coords))
#
#             self.l_next_waypoints = list(m_points.T)
#
#         return self.l_next_waypoints.pop(0)


class GridPlanner(RoutePlanner):
    name_on_figs = "Grid Planner"
    l_next_waypoints = []
    d_actions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UPLEFT": (-1, -1),
        "UPRIGHT": (-1, 1),
        "DOWNLEFT": (1, -1),
        "DOWNRIGHT": (1, 1),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l_next_measurement_locations = []
        self.previous_waypoint = None #[]

    def l_inds_all_grid_points(self, grid_dimension):
        """
        Args:
            - `grid_dimension`: a shape of a matrix whose indices are to
            be sorted out.

        Returns:
              -`l_inds_all_grid_points`: a list of grid point indices
        """
        m_grid_inds = np.indices(dimensions=grid_dimension)
        # m_grid_inds is a 2 x Ny x Nx matrix whose fist slab contains
        # row indices and second slab contains column indices
        l_inds_all_grid_points = []
        col_len = grid_dimension[1]
        for ind_col in range(col_len):
            l_inds_col = list(zip(m_grid_inds[0, :, ind_col], m_grid_inds[1, :, ind_col]))
            # if ind_col is odd then reverse the list
            if ind_col % 2 != 0:
                l_inds_col.reverse()

            l_inds_all_grid_points += l_inds_col

        return l_inds_all_grid_points

    def next_waypoint(self, d_map_estimate, m_building_metadata):
        if not self.l_next_waypoints:
            # m_grid_inds = np.indices(dimensions=m_building_metadata.shape)
            # # m_grid_inds is a 2 x Ny x Nx matrix whose fist slab contains
            # # row indices and second slab contains column indices
            # l_inds_all_grid_points = []
            # col_len = m_building_metadata.shape[1]
            # for ind_col in range(col_len):
            #     l_inds_col = list(zip(m_grid_inds[0, :, ind_col], m_grid_inds[1, :, ind_col]))
            #     # if ind_col is odd then reverse the list
            #     if ind_col % 2 != 0:
            #         l_inds_col.reverse()
            #
            #     l_inds_all_grid_points += l_inds_col
            l_inds_all_grid_points = self.l_inds_all_grid_points(
                grid_dimension=m_building_metadata.shape)

            self.l_next_waypoints= self.destination_list_to_waypoint_list(destination_list=l_inds_all_grid_points,
                                                   m_building_metadata=m_building_metadata)

            # set initial point to be previous_waypoint
            # self.previous_waypoint = self.l_next_waypoints[0]

        return self.l_next_waypoints.pop(0)


# class SquareSpiralGridPlanner(RoutePlanner):
#     name_on_figs = "Spiral Grid Planner"
#     l_next_waypoints = []
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def next_waypoint(self, d_map_estimate, m_building_metadata):
#         if not self.l_next_waypoints:
#             # stay `self.dist_between_measurements` away from the limits
#             max_x = self.grid.max_x()
#             min_x = self.grid.min_x()
#
#             max_y = self.grid.max_y()
#             min_y = self.grid.min_y()
#
#             # initial point in top left corner
#             v_x_coords = min_x
#             v_y_coords = max_y
#             count_in_corner_grid = 1
#             count_in_spiral = 1
#             count = 0
#             while 1:
#                 if count_in_corner_grid == 1:
#                     next_v_x_coords = min_x + count_in_spiral * self.dist_between_measurements
#                     next_v_y_coords = max_y - count_in_spiral * self.dist_between_measurements
#                 elif count_in_corner_grid == 2:
#                     next_v_x_coords = max_x - count_in_spiral * self.dist_between_measurements
#                     next_v_y_coords = max_y - count_in_spiral * self.dist_between_measurements
#
#                 elif count_in_corner_grid == 3:
#                     next_v_x_coords = max_x - count_in_spiral * self.dist_between_measurements
#                     next_v_y_coords = min_y + count_in_spiral * self.dist_between_measurements
#                 elif count_in_corner_grid == 4:
#                     next_v_x_coords = min_x + (count_in_spiral + 1) * self.dist_between_measurements
#                     next_v_y_coords = min_y + count_in_spiral * self.dist_between_measurements
#
#                 if count_in_corner_grid % 4 == 0:
#                     count_in_spiral += 1
#                     count_in_corner_grid = 1
#                 else:
#                     count_in_corner_grid += 1
#
#                 count += 1
#                 if count == 200:
#                     break
#
#                 v_x_coords = np.append(v_x_coords, next_v_x_coords)
#                 v_y_coords = np.append(v_y_coords, next_v_y_coords)
#
#             v_z_coords = np.full(v_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)
#
#             m_points = np.vstack((v_x_coords, v_y_coords, v_z_coords))
#
#             self.l_next_waypoints = list(m_points.T)
#
#         return self.l_next_waypoints.pop(0)


class SquareSpiralGridPlanner(RoutePlanner):
    name_on_figs = "Spiral Grid Planner"
    d_actions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UPLEFT": (-1, -1),
        "UPRIGHT": (-1, 1),
        "DOWNLEFT": (1, -1),
        "DOWNRIGHT": (1, 1),
    }
    l_next_waypoints = []
    # l_shortest_next_waypoints = []
    # l_inds_all_squiral_grid_points = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.b_initial_point = True
        self.l_next_measurement_locations = []
        self.previous_waypoint = None

    # def l_inds_of_all_waypoints_in_square_spiral_grid(self,
    #                                                   row_ind_i, col_ind_j,
    #                                                   row_len, col_len):
    #     """Args:
    #         - `row_ind_i`: starting row index value
    #         - `col_ind_j`: starting column index value
    #         - `row_len`: end index + 1 of row
    #         - `col_len`: end index + 1 of col
    #
    #     """
    #
    #     # If i or j lies outside the matrix
    #     if (row_ind_i >= row_len or col_ind_j >= col_len):
    #         return
    #
    #     # Print First Row
    #     # for p in range(row_ind_i, col_len):
    #     #     print(arr[row_ind_i][p], end=" ")
    #     row_inds = row_ind_i * np.ones(shape=(col_len,), dtype=int)
    #     col_inds = np.arange(row_ind_i, col_len, dtype=int)
    #
    #     self.l_inds_all_squiral_grid_points = self.l_inds_all_squiral_grid_points + list(zip(row_inds, col_inds))
    #
    #     # Print Last Column
    #     # for p in range(row_ind_i + 1, row_len):
    #     #     print(arr[p][col_len - 1], end=" ")
    #     row_inds = np.arange(row_ind_i + 1, row_len, dtype=int)
    #     col_inds = (col_len - 1) * np.ones(shape=(row_len,), dtype=int)
    #     self.l_inds_all_squiral_grid_points = self.l_inds_all_squiral_grid_points + list(zip(row_inds, col_inds))
    #
    #     # Print Last Row, if Last and
    #     # First Row are not same
    #     if ((row_len - 1) != row_ind_i):
    #         # for p in range(col_len - 2, col_ind_j - 1, -1):
    #         #     print(arr[row_len - 1][p], end=" ")
    #         row_last_ind = col_len - 2 - (col_ind_j - 1)
    #         row_inds = (row_len - 1) * np.ones(shape=(row_last_ind,), dtype=int)
    #         col_inds = np.arange(col_len - 2, col_ind_j - 1, -1, dtype=int)
    #         self.l_inds_all_squiral_grid_points = self.l_inds_all_squiral_grid_points + list(zip(row_inds, col_inds))
    #
    #     # Print First Column, if Last and First Column are not same
    #     if ((col_len - 1) != col_ind_j):
    #         # for p in range(row_len - 2, row_ind_i, -1):
    #         #     print(arr[p][col_ind_j], end=" ")
    #         col_last_ind = row_len - 2 - row_ind_i
    #         row_inds = np.arange(row_len - 2, row_ind_i, -1, dtype=int)
    #         col_inds = col_ind_j * np.ones(shape=(col_last_ind,), dtype=int)
    #         self.l_inds_all_squiral_grid_points = self.l_inds_all_squiral_grid_points + list(zip(row_inds, col_inds))
    #
    #     self.l_inds_of_all_waypoints_in_square_spiral_grid(row_ind_i + 1,
    #                                                        col_ind_j + 1,
    #                                                        row_len - 1,
    #                                                        col_len - 1)

    # def next_waypoint(self, d_map_estimate, m_building_metadata):
    #     # if not self.l_next_waypoints:
    #     #     # stay `self.dist_between_measurements` away from the limits
    #     #     max_x = self.grid.max_x()
    #     #     min_x = self.grid.min_x()
    #     #
    #     #     max_y = self.grid.max_y()
    #     #     min_y = self.grid.min_y()
    #     #
    #     #     # initial point in top left corner
    #     #     v_x_coords = min_x
    #     #     v_y_coords = max_y
    #     #     count_in_corner_grid = 1
    #     #     count_in_spiral = 1
    #     #     count = 0
    #     #     while 1:
    #     #         if count_in_corner_grid == 1:
    #     #             next_v_x_coords = min_x + count_in_spiral * self.grid.gridpoint_spacing
    #     #             next_v_y_coords = max_y - count_in_spiral * self.grid.gridpoint_spacing
    #     #         elif count_in_corner_grid == 2:
    #     #             next_v_x_coords = max_x - count_in_spiral * self.grid.gridpoint_spacing
    #     #             next_v_y_coords = max_y - count_in_spiral * self.grid.gridpoint_spacing
    #     #
    #     #         elif count_in_corner_grid == 3:
    #     #             next_v_x_coords = max_x - count_in_spiral * self.grid.gridpoint_spacing
    #     #             next_v_y_coords = min_y + count_in_spiral * self.grid.gridpoint_spacing
    #     #         elif count_in_corner_grid == 4:
    #     #             next_v_x_coords = min_x + (count_in_spiral + 1) * self.grid.gridpoint_spacing
    #     #             next_v_y_coords = min_y + count_in_spiral * self.grid.gridpoint_spacing
    #     #
    #     #         if count_in_corner_grid % 4 == 0:
    #     #             count_in_spiral += 1
    #     #             count_in_corner_grid = 1
    #     #         else:
    #     #             count_in_corner_grid += 1
    #     #
    #     #         count += 1
    #     #         if count == 200:
    #     #             break
    #     #
    #     #         v_x_coords = np.append(v_x_coords, next_v_x_coords)
    #     #         v_y_coords = np.append(v_y_coords, next_v_y_coords)
    #     #
    #     #     v_z_coords = np.full(v_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)
    #     #
    #     #     m_points = np.vstack((v_x_coords, v_y_coords, v_z_coords))
    #     #
    #     #     self.l_next_waypoints = list(m_points.T)
    #     if not self.l_inds_all_squiral_grid_points:
    #         self.l_inds_of_all_waypoints_in_square_spiral_grid(
    #             row_ind_i=0,
    #             col_ind_j=0,
    #             row_len=self.grid.num_points_y,
    #             col_len=self.grid.num_points_x
    #         )
    #
    #     if not self.l_next_waypoints:
    #         if not self.l_shortest_next_waypoints:
    #             waypoint_inds = self.l_inds_all_squiral_grid_points.pop(0)
    #             count_waypoint_inside_building = 0
    #             # check if the first point is outside buildings
    #             while m_building_metadata[waypoint_inds] == 1:
    #                 waypoint_inds = self.l_inds_all_squiral_grid_points.pop(0)
    #                 count_waypoint_inside_building += 1
    #                 # print(waypoint_inds)
    #
    #             # if the waypoint is not an initial point and inside the building
    #             # use shortest path to find between previous and current waypoint
    #             if not self.b_initial_point and count_waypoint_inside_building>=1:
    #                 l_path_inds = []
    #                 while not l_path_inds:
    #                     t_inds_current = self.grid.nearest_gridpoint_inds(
    #                     self.previous_waypoint)
    #                     t_inds_destination = waypoint_inds
    #
    #                     m_node_costs = np.ones(shape=m_building_metadata.shape)
    #                     m_node_costs[m_building_metadata==1] = 0
    #
    #                     l_path_inds = self.shortest_path(m_node_costs, t_inds_current,
    #                                                      t_inds_destination, m_building_metadata)
    #
    #                     # if the point is not reachable, shortest path returns empty l_path_inds
    #                     # then find next way point
    #                     if not l_path_inds:
    #                         waypoint_inds = self.l_inds_all_squiral_grid_points.pop(0)
    #                         while m_building_metadata[waypoint_inds] == 1:
    #                             if not self.l_inds_all_squiral_grid_points:
    #                                 print("Not reachable way points")
    #                                 set_trace()
    #                             waypoint_inds = self.l_inds_all_squiral_grid_points.pop(0)
    #
    #                 # Turn indices to coordinates
    #                 self.l_shortest_next_waypoints = [
    #                     self.grid.indices_to_point(inds) for inds in l_path_inds
    #                 ]
    #                 self.l_next_waypoints = [self.l_shortest_next_waypoints.pop(0)]
    #
    #             else:
    #                 if self.b_initial_point:
    #                     self.previous_waypoint = self.grid.indices_to_point(waypoint_inds)
    #                     self.l_next_measurement_locations = [self.previous_waypoint]
    #
    #                 self.l_next_waypoints= [self.grid.indices_to_point(waypoint_inds)]
    #                 self.b_initial_point = False
    #
    #         # list of way_points from shortest path is not empty
    #         else:
    #             self.l_next_waypoints = [self.l_shortest_next_waypoints.pop(0)]
    #
    #     return self.l_next_waypoints.pop(0)

    def l_inds_of_all_waypoints_in_square_spiral_grid(self,
                                                      row_ind_i, col_ind_j,
                                                      row_len, col_len):
        """Args:
            - `row_ind_i`: starting row index value
            - `col_ind_j`: starting column index value
            - `row_len`: end index + 1 of row
            - `col_len`: end index + 1 of col

            Returns:
                `l_inds_all_spiral_grid_points`: a list of all the spiral grid point indices
                outside buildings.

        """

        # If i or j lies outside the matrix
        if (row_ind_i >= row_len or col_ind_j >= col_len):
            return []

        # Print First Row
        # for p in range(row_ind_i, col_len):
        #     print(arr[row_ind_i][p], end=" ")
        row_inds = row_ind_i * np.ones(shape=(col_len,), dtype=int)
        col_inds = np.arange(row_ind_i, col_len, dtype=int)
        l_inds_top_row = list(zip(row_inds, col_inds))

        # Print Last Column
        # for p in range(row_ind_i + 1, row_len):
        #     print(arr[p][col_len - 1], end=" ")
        row_inds = np.arange(row_ind_i + 1, row_len, dtype=int)
        col_inds = (col_len - 1) * np.ones(shape=(row_len,), dtype=int)
        l_inds_right_col = list(zip(row_inds, col_inds))

        # Print Last Row, if Last and
        # First Row are not same
        if ((row_len - 1) != row_ind_i):
            # for p in range(col_len - 2, col_ind_j - 1, -1):
            #     print(arr[row_len - 1][p], end=" ")
            row_last_ind = col_len - 2 - (col_ind_j - 1)
            row_inds = (row_len - 1) * np.ones(shape=(row_last_ind,), dtype=int)
            col_inds = np.arange(col_len - 2, col_ind_j - 1, -1, dtype=int)
            l_inds_bottom_row = list(zip(row_inds, col_inds))

        # Print First Column, if Last and First Column are not same
        if ((col_len - 1) != col_ind_j):
            # for p in range(row_len - 2, row_ind_i, -1):
            #     print(arr[p][col_ind_j], end=" ")
            col_last_ind = row_len - 2 - row_ind_i
            row_inds = np.arange(row_len - 2, row_ind_i, -1, dtype=int)
            col_inds = col_ind_j * np.ones(shape=(col_last_ind,), dtype=int)
            l_inds_left_col = list(zip(row_inds, col_inds))

        l_inds_all_spiral_grid_points = l_inds_top_row + l_inds_right_col + l_inds_bottom_row + l_inds_left_col

        return l_inds_all_spiral_grid_points + self.l_inds_of_all_waypoints_in_square_spiral_grid(row_ind_i + 1,
                                                           col_ind_j + 1,
                                                           row_len - 1,
                                                           col_len - 1)

    def next_waypoint(self, d_map_estimate, m_building_metadata):
        # if not self.l_next_waypoints:
        #     # stay `self.dist_between_measurements` away from the limits
        #     max_x = self.grid.max_x()
        #     min_x = self.grid.min_x()
        #
        #     max_y = self.grid.max_y()
        #     min_y = self.grid.min_y()
        #
        #     # initial point in top left corner
        #     v_x_coords = min_x
        #     v_y_coords = max_y
        #     count_in_corner_grid = 1
        #     count_in_spiral = 1
        #     count = 0
        #     while 1:
        #         if count_in_corner_grid == 1:
        #             next_v_x_coords = min_x + count_in_spiral * self.grid.gridpoint_spacing
        #             next_v_y_coords = max_y - count_in_spiral * self.grid.gridpoint_spacing
        #         elif count_in_corner_grid == 2:
        #             next_v_x_coords = max_x - count_in_spiral * self.grid.gridpoint_spacing
        #             next_v_y_coords = max_y - count_in_spiral * self.grid.gridpoint_spacing
        #
        #         elif count_in_corner_grid == 3:
        #             next_v_x_coords = max_x - count_in_spiral * self.grid.gridpoint_spacing
        #             next_v_y_coords = min_y + count_in_spiral * self.grid.gridpoint_spacing
        #         elif count_in_corner_grid == 4:
        #             next_v_x_coords = min_x + (count_in_spiral + 1) * self.grid.gridpoint_spacing
        #             next_v_y_coords = min_y + count_in_spiral * self.grid.gridpoint_spacing
        #
        #         if count_in_corner_grid % 4 == 0:
        #             count_in_spiral += 1
        #             count_in_corner_grid = 1
        #         else:
        #             count_in_corner_grid += 1
        #
        #         count += 1
        #         if count == 200:
        #             break
        #
        #         v_x_coords = np.append(v_x_coords, next_v_x_coords)
        #         v_y_coords = np.append(v_y_coords, next_v_y_coords)
        #
        #     v_z_coords = np.full(v_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)
        #
        #     m_points = np.vstack((v_x_coords, v_y_coords, v_z_coords))
        #
        #     self.l_next_waypoints = list(m_points.T)

        if not self.l_next_waypoints:
            l_inds_all_spiral_grid_points = self.l_inds_of_all_waypoints_in_square_spiral_grid(
                row_ind_i=0,
                col_ind_j=0,
                row_len=self.grid.num_points_y,
                col_len=self.grid.num_points_x
            )
            # waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            # # check if the first point is outside buildings
            # while m_building_metadata[waypoint_inds] == 1:
            #     waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            #
            # l_inds_waypoints = [waypoint_inds]
            # #self.previous_waypoint = self.grid.indices_to_point(l_inds_waypoints[-1])
            #
            # while l_inds_all_spiral_grid_points:
            #
            #     next_waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            #     # if m_building_metadata[next_waypoint_inds] !=1:
            #     #     l_inds_waypoints.append(next_waypoint_inds)
            #     # else:
            #     # # check if the point is outside buildings
            #     # # count_waypoint_inside_building = 0
            #     while m_building_metadata[next_waypoint_inds] == 1:
            #         next_waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            #         # count_waypoint_inside_building += 1
            #     # point inside building find way_points using shortest path algorithm
            #     # if count_waypoint_inside_building>=1:
            #     l_shortest_path_inds = []
            #     while not l_shortest_path_inds:
            #         t_inds_current = l_inds_waypoints[-1]
            #         t_inds_destination = next_waypoint_inds
            #
            #         m_node_costs = np.ones(shape=m_building_metadata.shape)
            #         m_node_costs[m_building_metadata == 1] = 0
            #
            #         l_shortest_path_inds = self.shortest_path(m_node_costs, t_inds_current,
            #                                          t_inds_destination, m_building_metadata)
            #
            #         # if the point is not reachable, shortest path returns empty l_path_inds
            #         # then find next way point
            #         if not l_shortest_path_inds:
            #             next_waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            #             while m_building_metadata[next_waypoint_inds] == 1:
            #                 if not l_inds_all_spiral_grid_points:
            #                     print("Not reachable way points")
            #                     set_trace()
            #                 next_waypoint_inds = l_inds_all_spiral_grid_points.pop(0)
            #
            #         l_inds_waypoints = l_inds_waypoints + l_shortest_path_inds
            #
            # self.l_next_waypoints = [
            #     self.grid.indices_to_point(inds) for inds in l_inds_waypoints]

            self.l_next_waypoints = self.destination_list_to_waypoint_list(
                destination_list=l_inds_all_spiral_grid_points,
                m_building_metadata=m_building_metadata
            )
            # set initial point to be previous_waypoint
            # self.previous_waypoint = self.l_next_waypoints[0]

        nex_waypoint = self.l_next_waypoints.pop(0)

        return nex_waypoint


class RandomPlanner(RoutePlanner):
    name_on_figs = "Random Planner"
    l_previous_waypoints = None

    def __init__(self,
                 dist_between_waypoints=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert dist_between_waypoints
        self.dist_between_waypoints = dist_between_waypoints

    def next_waypoint(self, d_map_estimate):

        max_x = self.grid.max_x()
        min_x = self.grid.min_x()

        max_y = self.grid.max_y()
        min_y = self.grid.min_y()
        slope_in_radian = np.random.choice(np.arange(0, 2 * np.pi, np.pi / 6))  # list of angle spaced by 30 degree

        # slope_in_radian = np.random.uniform(0, 2 * np.pi)

        if self.l_previous_waypoints is None:
            waypoint = self.grid.random_point_in_the_area()
            self.l_previous_waypoints = waypoint

        else:
            x_coord = self.l_previous_waypoints[0] + self.dist_between_waypoints * np.cos(slope_in_radian)

            if x_coord > max_x:
                x_coord = max_x
            elif x_coord < min_x:
                x_coord = min_x
            else:
                x_coord = x_coord

            y_coord = self.l_previous_waypoints[1] + self.dist_between_waypoints * np.sin(slope_in_radian)
            if y_coord > max_y:
                y_coord = max_y
            elif y_coord < min_y:
                y_coord = min_y

            z_coord = self.l_previous_waypoints[2]

            waypoint = np.array([x_coord, y_coord, z_coord])
            self.l_previous_waypoints = waypoint

        return waypoint


class MinimumCostPlanner(RoutePlanner):
    name_on_figs = "Min. Cost Planner"

    d_actions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UPLEFT": (-1, -1),
        "UPRIGHT": (-1, 1),
        "DOWNLEFT": (1, -1),
        "DOWNRIGHT": (1, 1),
    }

    l_next_waypoints = []

    def __init__(self,
                 metric=None,  # can be "power_variance" or "service_entropy"
                 grid=None,
                 smoothing_constant=1.0,
                 cost_factor=1.0,
                 cost_func="Reciprocal",
                 **kwargs):

        assert metric
        assert grid

        self.smoothing_constant = smoothing_constant
        self.cost_factor = cost_factor
        self.cost_func = cost_func
        self.metric = metric
        self.name_on_figs = f"{self.name_on_figs} ({metric})"
        # number of times at each grid point
        self.m_visited_points = np.zeros((grid.num_points_y, grid.num_points_x))

        super().__init__(grid=grid, **kwargs)
        self.l_next_measurement_locations = []
        self.previous_waypoint = None
        self.l_all_next_waypoints = []

    # def next_measurement_location(self, d_map_estimate, m_building_metadata):
    #     self.measurement_count += 1
    #     if self.l_next_measurement_locations:
    #         # Obtain next measurement location
    #         next_location = self.l_next_measurement_locations.pop(0)
    #
    #     else:
    #         next_location = self.next_waypoint(d_map_estimate, m_building_metadata)
    #         self.previous_waypoint = next_location
    #
    #         # Update destination at every self.num_measurement_to_update_destination measurements
    #         if self.num_measurement_to_update_destination and \
    #                 self.measurement_count % self.num_measurement_to_update_destination == 0:
    #             self.initial_location = next_location
    #             self.l_next_waypoints = []
    #             self.reset()
    #     return next_location

    def next_measurement_location(self, d_map_estimate, m_building_metadata=None):
        self.measurement_count += 1
        # For the first measurement, outside of building
        if self.previous_waypoint is None:
            # get a matrix with connected region label and region value with
            # maximum connected points.
            m_connected_labels, label_with_max_freq = self.connected_labels(
                m_building_metadata=m_building_metadata)

            # choose random point from the region with highest reachable points
            m_building_metadata[m_connected_labels != label_with_max_freq] = 1

            # choose (0,0) as initial point if it is outside of building
            # and in a region with maximum connectivity.
            if m_building_metadata[0, 0] != 1:
                self.previous_waypoint = self.grid.indices_to_point((0, 0))
            else:
                self.previous_waypoint = self.grid.random_points_in_the_grid_outside_buildings(
                    m_building_metadata=m_building_metadata)[0]
            # self.previous_waypoint = self.grid.indices_to_point((0, 0))
            return self.previous_waypoint

        while not self.l_next_measurement_locations:
            # Obtain next measurement location
            next_waypoint = self.next_waypoint(d_map_estimate, m_building_metadata=m_building_metadata)

            if self.dist_between_measurements is None:
                # take measurements at grid points
                self.l_next_measurement_locations = [next_waypoint]
                self.previous_waypoint = next_waypoint

            else:
                self.l_next_measurement_locations, self.remaining_dist = \
                    self.measurement_locations_from_waypoints(
                        self.previous_waypoint,
                        next_waypoint,
                        self.remaining_dist)

                self.previous_waypoint = next_waypoint

        next_location = self.l_next_measurement_locations.pop(0)

        # Update destination every self.num_measurement_to_update_destination measurements
        if self.num_measurement_to_update_destination and \
                self.measurement_count % self.num_measurement_to_update_destination == 0:
            # self.initial_location = next_location
            self.l_next_waypoints=[]
            self.l_next_measurement_locations = []

        # set_trace()
        # print(f"route_planning: next_location = {next_location}")

        return next_location

    def next_waypoint(self, d_map_estimate, m_building_metadata):

        if not self.l_next_waypoints:

            t_inds_current = self.grid.nearest_gridpoint_inds(
                self.previous_waypoint)
            # TODO: t_inds_destination and m_uncertainty should consider building locations
            # Choose a destination
            t_inds_destination, m_uncertainty = self.next_destination(t_inds_current, d_map_estimate, m_building_metadata)

            # To plot the destination point
            self.destination_point = self.grid.indices_to_point(t_inds_destination)
            # Find shortest path
            # m_node_costs = 1 / (m_uncertainty + 0.01)
            if self.cost_func is "Reciprocal":
                m_node_costs = (1-self.cost_factor) * 1 + self.cost_factor * 1 / (m_uncertainty + 0.01)
            elif self.cost_func is "Exponential":
                m_node_costs = (1 - self.cost_factor) * 1 + self.cost_factor * np.exp(-m_uncertainty)
            elif self.cost_func is "Threshold":
                m_node_costs = np.zeros(m_uncertainty.shape)
                # m_node_costs[m_uncertainty < (0.5 * m_uncertainty.max())] = 1
                # get the threshold value such that 70% of the points are below it.
                uncertainty_threshold = np.percentile(m_uncertainty, 85)
                # set the cost to be 1 for uncertainty less than threshold value.
                m_node_costs[m_uncertainty < uncertainty_threshold] = 1
            elif self.cost_func is "Shortest":
                m_node_costs = 1 + np.zeros(shape=m_uncertainty.shape)
            else:
                raise NotImplementedError

            # m_node_costs = np.exp(-m_uncertainty)
            l_path_inds = self.shortest_path(m_node_costs, t_inds_current,
                                             t_inds_destination, m_building_metadata)

            if l_path_inds is None:
                """if route is not found choose a random 
                point outside of buildings"""
                self.l_next_waypoints = [self.grid.random_points_in_the_grid_outside_buildings(
                    m_building_metadata=m_building_metadata)[0]]

            else:
                # Turn indices to coordinates
                self.l_next_waypoints = [
                    self.grid.indices_to_point(inds) for inds in l_path_inds
                ]

            if self.debug_code == 2:
                self.plot_path(t_inds_current, l_path_inds, m_node_costs=1 / m_node_costs)
                plt.show()
                # set_trace()

        if not self.l_next_waypoints:
            set_trace()
        # self.l_all_next_waypoints = self.l_next_waypoints
        return self.l_next_waypoints.pop(0)

    def next_destination(self, t_inds_current, d_map_estimate, m_building_metadata):
        """Returns the indices of the next destination grid point."""

        if self.metric == "power_variance":
            m_uncertainty = np.sum(d_map_estimate["t_power_map_norm_variance"], 0)
        elif self.metric == "service_entropy":
            m_uncertainty = np.sum(d_map_estimate["t_service_map_entropy"], 0)
            # m_uncertainty = np.max(d_map_estimate["t_service_map_entropy"],0)
        else:
            raise Exception("Invalide metric")

        if self.avg_previous_uncertainty is None:
            self.avg_previous_uncertainty = m_uncertainty
        else:
            self.avg_previous_uncertainty = self.smoothing_constant * m_uncertainty + \
                                            (1-self.smoothing_constant) * self.avg_previous_uncertainty

        m_uncertainty = self.avg_previous_uncertainty

        # Spatial filter
        kernel = np.ones((3, 3))
        # kernel = np.random.uniform(size=(3, 3))
        # kernel = np.array([[1,2,1], [2,4,2],[1,2,1]])
        m_mod_uncertainty = convolve(m_uncertainty, kernel/np.sum(kernel))
        # m_mod_uncertainty = m_uncertainty
        # Modified uncertainty
        m_mod_uncertainty = m_mod_uncertainty * (1 / (1 + self.m_visited_points))
        m_mod_uncertainty[t_inds_current] = 0  # prevent remaining in the same point
        # TODO: get the destination point outside the building locations
        # set the m_mod_uncertainty values to 0 at building locations
        m_mod_uncertainty_building_zero = np.where(m_building_metadata == 1,
                                                   0, m_mod_uncertainty)
        # t_inds_destination = mat_argmax(m_mod_uncertainty)

        # get a matrix with connected region label and region value with
        # maximum connected points.
        m_connected_labels, label_with_max_freq= self.connected_labels(
            m_building_metadata=m_building_metadata)

        # choose destination point from the region with highest reachable points
        # and with highest uncertainty
        m_mod_uncertainty_building_zero[m_connected_labels != label_with_max_freq] = 0

        t_inds_destination = mat_argmax(m_mod_uncertainty_building_zero)

        if t_inds_destination == t_inds_current:
            set_trace()
            # print("already at point of maximum uncertainty, choosing next waypoint randomly")
            # t_inds_destination = (np.random.randint(0, high=m_uncertainty.shape[0]),
            #                       np.random.randint(0, high=m_uncertainty.shape[0])
            #                       )

        self.m_visited_points[t_inds_destination] += 1

        return t_inds_destination, m_mod_uncertainty

    def shortest_path(self, m_node_costs=None, start=None, destination=None, m_building_metadata=None):
        """Bellman-Ford algorithm, aka breath-first search (BFS)
        [bertsekas2005]. Provides a path with minimum cost."
        Arguments:
        `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
        cost of traversing the grid point (i,j). Its dimensions Ny, Nx
        define a rectangular grid.
        Returns:
        `l_path` : None if a path does not exist. Else, a list of grid point
        indices corresponding to a shortest path between `start` and
        `destination`, which are grid node indices. `start` is not
        included in the returned list unless start==destination, but
        `destination` is always included.
        """

        self.m_node_costs = m_node_costs  # Required by other methods

        def is_in_grid(t_inds):
            return (t_inds[0] < m_node_costs.shape[0]) \
                   and (t_inds[1] < m_node_costs.shape[1])

        assert is_in_grid(start)
        assert is_in_grid(destination)

        if start == destination:
            return [destination]

        queue = FifoUniqueQueue()  # OPEN queue in [bertsekas2005]. It
        # contains nodes that are candidates
        # for being in the shortest path to
        # other nodes.
        queue.put(start)
        m_cost = np.full(m_node_costs.shape, fill_value=np.infty, dtype=float)
        m_cost[start] = 0

        # keys: state. values: previous_state
        d_branches = {}
        while not queue.empty():

            current_point = queue.get()
            # print(f"point={current_point}")

            for str_action in self._possible_actions(current_point, m_building_metadata):
                next_point = self._next_state(current_point, str_action)

                new_cost = m_cost[current_point] + self.transition_cost(
                    current_point, next_point)

                # UPPER in [bertsekas2005] is m_cost[destination]
                if new_cost < min(m_cost[next_point], m_cost[destination]):

                    d_branches[next_point] = current_point
                    m_cost[next_point] = new_cost

                    if next_point != destination:
                        queue.put(next_point)

        if m_cost[destination] < np.infty:
            if self.debug_code:
                print("Route found")
            state = destination
            l_path = [state]
            while d_branches[state] != start:
                previous_state = d_branches[state]
                l_path.append(previous_state)
                state = previous_state

            return l_path[::-1]

        else:
            "select random point outside building if there is no route found."
            # l_inidices = np.where(m_building_metadata == 0)
            # ind = np.random.randint(0, int(len(l_inidices[0])))
            # random_point_in_grid_indices = (l_inidices[0][ind], l_inidices[1][ind])
            # # set_trace()
            # print("Route not found")
            # return [random_point_in_grid_indices]
            return None

    def transition_cost(self, point_1, point_2):
        # approximates the integral of the cost
        dist = np.linalg.norm(np.array(point_2) - np.array(point_1))

        cost_1 = self.m_node_costs[point_1[0], point_1[1]]
        cost_2 = self.m_node_costs[point_2[0], point_2[1]]
        return dist / 2 * cost_1 + dist / 2 * cost_2

    def _possible_actions(self, state, m_building_metadata):
        """
        Arguments:
        `state`: tuple with the indices of the considered grid point.
        Returns:
        list of possible actions at state `state`.

        """

        max_row_index = self.m_node_costs.shape[0] - 1
        max_col_index = self.m_node_costs.shape[1] - 1

        l_actions = []
        for str_action in self.d_actions:

            candidate_entry = self._next_state(state, str_action)
            # Check if in grid
            if candidate_entry[0] >= 0 and \
                    candidate_entry[0] <= max_row_index and \
                    candidate_entry[1] >= 0 and \
                    candidate_entry[1] <= max_col_index:
                # TODO: check whether the point is inside the building or not
                if m_building_metadata[candidate_entry] != 1:
                    l_actions.append(str_action)

        return l_actions

    def _next_state(self, state, str_action):

        v_movement = self.d_actions[str_action]
        return (state[0] + v_movement[0], state[1] + v_movement[1])


class UniformRandomSamplePlanner(RoutePlanner):
    name_on_figs = "Uni. Random Sample Planner "

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_waypoint(self, d_map_estimate, m_building_metadata):
        """Returns length-3 vector with the coordinates of a random point in
        the area (not nec. on the grid)
        """
        # random coordinates in the range of x_coords, y_coords, and z_coords

        # waypoint = self.grid.random_point_in_the_area()
        waypoint = self.grid.random_points_in_the_grid_outside_buildings(m_building_metadata=m_building_metadata)[0]
        return waypoint

    def next_measurement_location(self, m_uncertainty, m_building_metadata=None):

        next_location = self.next_waypoint(m_uncertainty, m_building_metadata)
        return next_location

def tests():
    def random_point(num_points_x, num_points_y):

        return (np.random.randint(low=0, high=num_points_y),
                np.random.randint(low=0, high=num_points_x))

    num_points_x = 20
    num_points_y = 10

    rp = MinimumCostPlanner(grid=1,
                            dist_between_measurements=1,
                            initial_location=(0, 0))
    m_node_costs = np.ones((num_points_y, num_points_x))
    # m_node_costs = m_node_costs + np.triu(m_node_costs)
    for row in range(num_points_y):
        for col in range(num_points_x):
            m_node_costs[row, col] = np.exp(-((row - num_points_y / 2) ** 2) /
                                            (num_points_y) ** 2 -
                                            ((col - num_points_x / 2) ** 2) /
                                            (num_points_x / 2) ** 2)

    while True:
        # start = (1, 6)  #
        start = random_point(num_points_x, num_points_y)
        # destination = (11, 6)
        destination = random_point(num_points_x, num_points_y)

        # approximately minimum cost
        l_path_cost_appr = rp.approximate_bfs_shortest_path(m_node_costs=m_node_costs,
                                                            start=start,
                                                            destination=destination)
        cost_appr = rp.path_cost(m_node_costs, start, l_path_cost_appr)
        print(f"Cost of approximate min cost path {cost_appr}")

        # shortest path in distance
        l_path_nodes = rp.approximate_bfs_shortest_path(m_node_costs=np.ones(
            (num_points_y, num_points_x)),
            start=start,
            destination=destination)

        print(f"Cost min dist path {rp.path_cost(m_node_costs, start, l_path_nodes)}")

        # minimum cost
        l_path_bf = rp.shortest_path(m_node_costs=m_node_costs,
                                     start=start,
                                     destination=destination)

        cost_bf = rp.path_cost(m_node_costs, start, l_path_bf)
        print(f"Cost BF path {cost_bf}")

        axis = rp.plot_path(start,
                            l_path_cost_appr,
                            m_node_costs=m_node_costs,
                            label="approx cost",
                            color="white")

        rp.plot_path(start,
                     l_path_nodes,
                     axis=axis,
                     label="nodes",
                     color="blue")

        rp.plot_path(start,
                     l_path_bf,
                     axis=axis,
                     label="BF",
                     color="red")

        if cost_appr < cost_bf:
            set_trace()

        plt.show()
    # set_trace()


if __name__ == "__main__":
    tests()

# TESTs
# m_node_costs = np.ones((num_points_y, num_points_x))
# m_node_costs[:, 5:] = 2
# start = (1, 6)  #random_point(num_points_x, num_points_y)
# destination = (11, 6)  #random_point(num_points_x, num_points_y)