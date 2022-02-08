#!/usr/bin/python3
from itertools import product

import numpy as np
import numpy.random as nr
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util import tau


class PercolationLattice:
    eps = 1e-12

    def __init__(self, dim=None, p=0.75, random_state=None,
                 xdim=3, ydim=None, zdim=None):
        if dim is not None:
            try:
                self._xdim = dim[0]
                self._ydim = dim[1]
                self._zdim = dim[2]
            except IndexError:
                self._xdim = dim[0]
                self._ydim = dim[0]
                self._zdim = dim[0]
        else:
            self._xdim = xdim
            self._ydim = ydim
            self._zdim = zdim

        if random_state is None:
            self.random_state = nr.randint(1, 2**32 - 1)
        else:
            self.random_state = random_state
        nr.seed(seed=self.random_state)
        self._p = p
        self.reset_lattice()

    def reset_lattice(self):
        self._lattice_graph = nx.Graph()
        self._lattice_graph.graph['xdim'] = self._xdim
        self._lattice_graph.graph['ydim'] = self._ydim
        self._lattice_graph.graph['zdim'] = self._zdim
        self._lattice_graph.graph['Prob'] = self._p
        self.construct_lattice()

    def construct_lattice(self):
        range_x = range(1, self._xdim + 1)
        range_y = range(1, self._ydim + 1)
        range_z = range(1, self._zdim + 1)

        for x, y, z, in product(range_x, range_y, range_z):
            self._lattice_graph.add_node(tau(x, y, z))
            self._lattice_graph.nodes[tau(x, y, z)]['X'] = x
            self._lattice_graph.nodes[tau(x, y, z)]['Y'] = y
            self._lattice_graph.nodes[tau(x, y, z)]['Z'] = z

            self._lattice_graph.nodes[tau(x, y, z)]['Checked'] = False
            random_num = nr.random()
            formation_value = self.get_formation_value(random_num)
            self._lattice_graph.nodes[tau(x, y, z)]['value'] = formation_value

    def get_formation_value(self, number: float) -> float:
        if number <= self._p ** 2:
            formation_value = 3
        elif number <= self._p:
            formation_value = 2
        elif number <= 2 * self._p - self._p ** 2:
            formation_value = 1
        else:
            formation_value = 0
        return formation_value

    def get_lattice_graph(self):
        return self._lattice_graph

    def connect_lattice(self):
        for node in self._lattice_graph.nodes():
            self.connect(node)

    def connect(self, node):
        self._lattice_graph.nodes[node]['Checked'] = True
        # Keeping the convention arbitrary for generality.
        self.determine_shell(node)
        for neighbour in self._lattice_graph.nodes[node]['shell']:
            # For every node in the 'shell' of nearest neighbouring nodes.
            self.determine_shell(neighbour)
            # For any sort of fusion, need to consider the probability.
            if node in self._lattice_graph.nodes[neighbour]['shell']:
                # This condition tests for a direct (non-diagonal) fusion.
                if not self._lattice_graph.nodes[neighbour]['Checked']:
                    rand = nr.random()
                    if rand <= self._p:
                        self._lattice_graph.add_edge(node, neighbour)
                continue
            else:
                connected = False
                """Not found a suiting node to connect to yet, so this BOOLEAN variable represents the connection status"""
                prior = node
                posterior = neighbour
                jump_count = 1
                while not connected:
                    """CHANGE CONVENTION OF WISE_CHAIN FUNCTION HERE"""
                    jump_count += 1
                    chain_decider = self.wise_chain(prior, posterior)
                    """Decides the next node to consider for a fusion to in the event of a diagonal connection"""
                    if (not chain_decider[1]) or (chain_decider[0] is None):
                        # Hit a boundary, no diagonal fusion possible
                        connected = True
                        break
                    else:
                        prior, posterior = posterior, chain_decider[0]
                        self.determine_shell(posterior)
                        if prior in self._lattice_graph.nodes[posterior]['shell']:
                            """Checks if the diagonal path has found its destination in the next node"""
                            if self._lattice_graph.nodes[posterior]['Checked']:
                                """Checks if this particular fusion has been considered before, if so, theres no need to make
                                another edge.
                                """
                                connected = True
                            else:
                                connected = True
                                rand = nr.random()
                                if rand <= self._p**jump_count:
                                    self._lattice_graph.add_edge(node,
                                                                 posterior)
                            break
                        else: 
                            continue
                        
    def determine_shell(self, node):
        """
        Instantiates a shell of immediately neighbouring points that
        correspond to the fusion connections in a brickwork pattern. This is 
        done by a positive and negative x-direction connection, and then
        alternation each consecutive node in any of three directions, between
        positive and negative connections in y-direction and z-direction.
        
        Parameters
        ----------
        
        node : dict
            Node in the networkx graph.
        """
        shell = []
        nodex = self._lattice_graph.nodes[node]['X']
        nodey = self._lattice_graph.nodes[node]['Y']
        nodez = self._lattice_graph.nodes[node]['Z']
        value = self._lattice_graph.nodes[node]['value']
        # Factor to determine the alternating brickwork lattice connections.
        factor = (-1)**(nodey + nodez +nodex)
        # The x-dimension is not affected by alternating factor.
        if 1 < nodex < self._xdim:
            if value in [3, 2]:
                alt_ = 1 if value == 3 else -1
                shell.append(tau(nodex + alt_, nodey, nodez))
                shell.append(tau(nodex - alt_, nodey, nodez))
        elif nodex == self._xdim:
            if value in [3, 2]:
                shell.append(tau(nodex - 1, nodey, nodez))
        else:
            if value in [3, 2]:
                shell.append(tau(nodex + 1, nodey, nodez))
        # The y-dimension is affected by alternating factor.
        if 1 < nodey < self._ydim:
            if value in [3, 1]:
                shell.append(tau(nodex, nodey + factor, nodez))
        elif nodey == self._ydim:
            if factor == -1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey - 1, nodez))
        else:
            if factor == 1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey + 1, nodez))
        # The z-dimension is affected by alternating factor
        if 1 < nodez < self._zdim:
            if value in [3, 1]:
                shell.append(tau(nodex, nodey, nodez + factor))
        elif nodez == self._zdim:
            if factor == -1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey, nodez - 1))
        else:
            if factor == 1:
                if value in [3, 1]:
                    shell.append(tau(nodex, nodey, nodez + 1))
        self._lattice_graph.nodes[node]['shell'] = shell

    def wise_chain(self, node, neighbour):
        """Wise_chain function which works on the exact same principles as the one above except
           convention 2 is applied where  1 (+ve x connection) & 2 (-ve x connection)
           are coupled. Also 3 (alternating y connection) & 4 (alternating z connection) are coupled.
           Same notation and decision flow is used here.
        """ 
        # Considers the neighbouring 'mate' of the current node.
        x_nn = self._lattice_graph.nodes[neighbour]['X']
        y_nn = self._lattice_graph.nodes[neighbour]['Y']
        z_nn = self._lattice_graph.nodes[neighbour]['Z']
        alt_factor = (-1)**(x_nn + y_nn + z_nn)
        continue_chain = True
        # There is no next unless convinced otherwise.
        next_node = None
        if x_nn == self._lattice_graph.nodes[node]['X'] + 1:
            # If +ve 1 in the x-direction
            if x_nn == self._xdim:
                continue_chain = False
            else:
                next_node = tau(x_nn + 1, y_nn, z_nn)
        elif x_nn == self._lattice_graph.nodes[node]['X'] - 1:
            # If -ve 1 in the x-direction
            if x_nn == 1:
                # If hitting the edge of the lattice
                continue_chain = False
            else:
                next_node = tau(x_nn - 1, y_nn, z_nn)

        elif int(np.abs(self._lattice_graph.nodes[node]['Y'] - y_nn)) == 1:
            # If a positive of negative difference in the y-direction.
            end_check_1 = bool(alt_factor == 1 and z_nn == self._zdim)
            end_check_2 = bool(alt_factor == -1 and z_nn == 1)
            if end_check_1 or end_check_2:
                continue_chain = False
            else:
                next_node = tau(x_nn, y_nn, z_nn + alt_factor)

        elif int(np.abs(self._lattice_graph.nodes[node]['Z'] - z_nn)) == 1:
            # If a positive of negative difference in the z-direction.
            end_check_1 = bool(alt_factor == 1 and y_nn == self._ydim)
            end_check_2 = bool(alt_factor == -1 and y_nn == 1)
            if end_check_1 or end_check_2:
                continue_chain = False
            else:
                next_node = tau(x_nn, y_nn + alt_factor, z_nn)

        return [next_node, continue_chain]

    def assess_percolation(self, premade=False):
        """Not for use with extendable lattices or overlap windows"""
        front_nodes = []
        back_nodes = []
        # Create two panels representing the nodes at the front and
        # the back of the lattice
        for y, z in product(range(1, self._ydim + 1),
                            range(1, self._zdim + 1)):
            front_nodes.append(tau(self._xdim, y , z))
            back_nodes.append(tau(1, y, z))
        if not premade:
            for node in self._lattice_graph.nodes():
                self.connect(node)
        for node1, node2 in product(back_nodes, front_nodes):
            # Detects any percolating path between nodes at the front or back.
            if node2 in nx.node_connected_component(self._lattice_graph,
                                                    node1):
                return True
        # Return False if no connections found.
        return False

    def edge_count(self):
        # edge_counts = [x, y, z, diagonal]
        edge_counts = [0, 0, 0, 0]
        for edge in self._lattice_graph.edges():
            n1, n2 = edge[0], edge[1]
            xdiff = np.abs(self._lattice_graph.nodes[n1]['X'] -
                           self._lattice_graph.nodes[n2]['X'])
            ydiff = np.abs(self._lattice_graph.nodes[n1]['Y'] -
                           self._lattice_graph.nodes[n2]['Y'])
            zdiff = np.abs(self._lattice_graph.nodes[n1]['Z'] -
                           self._lattice_graph.nodes[n2]['Z'])
            metric = [int(xdiff), int(ydiff), int(zdiff)]
            if sum(metric) == 1:
                indx = metric.index(1)
                edge_counts[indx] += 1
            else:
                edge_counts[3] += 1
        return edge_counts
    
    
    def average_cluster(self):
        clusters = 0
        clusters_sq = 0
        total_no = 0.0
        for node in self._lattice_graph.nodes():
            len_clust = len(nx.node_connected_component(self.__latt, node))
            if len_clust != 1:
                clusters += 1
                clusters_sq += len_clust
                total_no += 1./len_clust
        if clusters == 0:
            return [0, 1, 1]
        else:
            mean = clusters/total_no
            mean_sq = clusters_sq/total_no
            std = np.sqrt(np.abs(mean_sq-mean**2))
            return mean, std, total_no

    def panels(self):
        """RARELY USED (IGNORE THIS FUNCTION)
           Sometimes used to find front/back set of nodes from which to deduce a percolation"""
        front, back = [], []
        for z in range(1, self._zdim + 1):
            for y in range(1, self._ydim + 1):
                front.append(tau(self._xdim, y, z))
                back.append(tau(1, y, z))
        return (back, front)
    
def visualise(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xxs, xxh, xxf = [], [], []
    yys, yyh, yyf = [], [], []
    zzs, zzh, zzf = [], [], []
    for node in graph.nodes():
        if graph.nodes[node]['value'] == 3:
            xxs.append(graph.nodes[node]['X'])
            yys.append(graph.nodes[node]['Y'])
            zzs.append(graph.nodes[node]['Z'])
        elif graph.nodes[node]['value'] in [2, 1]:
            xxh.append(graph.nodes[node]['X'])
            yyh.append(graph.nodes[node]['Y'])
            zzh.append(graph.nodes[node]['Z'])
        else:
            xxf.append(graph.nodes[node]['X'])
            yyf.append(graph.nodes[node]['Y'])
            zzf.append(graph.nodes[node]['Z'])
    Axes3D.scatter(ax, xxs, yys, zzs, s = 12, color = 'k')
    Axes3D.scatter(ax, xxh, yyh, zzh, s = 12, color = 'g')
    Axes3D.scatter(ax, xxf, yyf, zzf, s = 12, color = 'r')
    ax.set_xlim((1, graph.graph['xdim']))
    ax.set_ylim((1, graph.graph['ydim']))
    ax.set_zlim((1, graph.graph['zdim']))
    for edge in graph.edges():
        x1, x2 = graph.nodes[edge[0]]['X'], graph.nodes[edge[1]]['X']
        y1, y2 = graph.nodes[edge[0]]['Y'], graph.nodes[edge[1]]['Y']
        z1, z2 = graph.nodes[edge[0]]['Z'], graph.nodes[edge[1]]['Z']
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'm-')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
            
        
    
            