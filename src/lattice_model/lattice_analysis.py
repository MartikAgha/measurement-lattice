from collections import defaultdict
from functools import partial

import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count

from .lattice_model import PercolationLattice
from .util import fermi_scaling_function, tau


class LatticeAnalysisRepeater:
    calculations_types = ['avg_cluster', 'fwd_bond_frac',
                          'planar_bond_frac', 'max_cluster']
    max_random_number = 2**32 - 1

    def __init__(self, xdim, ydim=None, zdim=None, parallel=False,
                 default_repeats=100, random_state=1):
        """
        LatticeAnalysisRepeater: Performing averages of calculations.
        :param xdim: size of grid along x-axis.
        :param ydim: size of grid along y-axis.
        :param zdim: size of grid along z-axis.
        :param parallel: Parallelise the calculation for repeats.
        :param default_repeats: Default number of repeats
        :param random_state: Integer to seed the random generator.
        """
        if ydim is None or zdim is None:
            if isinstance(xdim, (int, )):
                self._dim_tuple = (xdim, xdim, xdim)
            else:
                self._dim_tuple = (xdim[0], xdim[1], xdim[2])
        else:
            self._dim_tuple = (xdim, ydim, zdim)
        self._default_repeats = default_repeats
        self._data = defaultdict()
        self._parallel = parallel
        self._ncpus = cpu_count()
        self._random_state = random_state
        np.random.seed(random_state)

    def possible_calculations(self):
        """List possible calculations to perform."""
        return self.calculations_types

    def perform_analysis(self,
                         calculations=None,
                         probability=None,
                         n_repeats=None):
        """
        Perform statistical analysis on Lattice models to determine metrics
        from the possible calculations in this model.
        :param calculations: list of calculations to do
        :param probability: float or array of floats for probabilities to
                            measure at. If None then defaults to 0.5
        :param n_repeats: integer number of repeats for each probability
        :return:
        """
        # Set up datastream
        data = {}
        data_agg = {}
        data_err = {}
        if calculations is None:
            calculations = self.possible_calculations()
        else:
            for calc_type in calculations:
                if not calc_type in self.possible_calculations():
                    msg = "Calculation \'%s\' doesn\'t exist." % calc_type
                    raise ValueError(msg)

        if probability is None:
            probabilities = [0.5]
        else:
            if isinstance(probability, float):
                probabilities = [probability]
            elif isinstance(probability, (np.ndarray, list,)):
                probabilities = np.array(probability)
            else:
                msg = "\'%s\' should be a float, array or list." % probability
                raise TypeError(msg)

        repeat_no = self._default_repeats if n_repeats is None else n_repeats

        for p_id, p in enumerate(probabilities):
            if self._parallel:
                procs = cpu_count() if self._ncpus is None else self._ncpus
                pool = Pool(procs)
                func = partial(self.get_statistics_parallel,
                               statistics_list=calculations)
                outcomes = pool.map(func, list(p*np.ones(repeat_no)))
                prob_data = {id: dct for id, dct in enumerate(outcomes)}
            else:
                prob_data = {}
                for repeat_id in probabilities:
                    lattice = PercolationLattice(dim=self._dim_tuple,
                                                 p=p)
                    is_percolation = lattice.assess_percolation()
                    sim_data = {'percolation': float(is_percolation), 'p': p}
                    for calc_type in calculations:
                        statistic = self.get_statistic(lattice, calc_type)
                        sim_data[calc_type] = statistic
                    prob_data[repeat_id] = sim_data
            df = pd.DataFrame(prob_data).transpose()
            means = df.mean(axis=0)
            stds = df.std(axis=0)
            data[p_id] = {'p': p, 'data': df}
            data_agg[p_id] = {c: agg for c, agg in means.items()}
            data_err[p_id] = {c: agg for c, agg in stds.items()}

        self._data = data
        self._aggregate_data = pd.DataFrame(data_agg).transpose()
        self._error_data = pd.DataFrame(data_err).transpose()

    def get_statistics_parallel(self, p, statistics_list=None):
        """
        Statistics function to be used to obtain statistics in parallell.
        :param p: Fusion probability.
        :param statistics_list: Which statistics to calculate.
        :return:
        """
        lattice = PercolationLattice(dim=self._dim_tuple,
                                     p=p)
        is_percolation = lattice.assess_percolation()
        sim_data = {'percolation': float(is_percolation), 'p': p}
        for calc_type in statistics_list:
            statistic = self.get_statistic(lattice, calc_type)
            sim_data[calc_type] = statistic
        return sim_data

    def get_statistic(self, lattice: PercolationLattice,
                      calculation_type: str) -> float:
        """
        Determine statistics of a lattice
        :param lattice: PercolationLattice object which has been connected.
        :param calculation_type: List of statistics to calculate.
        :return: value of the statistic.
        """
        lattice_graph = lattice.get_lattice_graph()
        if calculation_type == 'avg_cluster':
            cluster_sizes = []
            for node in lattice_graph.nodes():
                if len(nx.node_connected_component(lattice_graph, node)) != 1:
                    cluster_sizes.append(
                        len(nx.node_connected_component(lattice_graph, node))
                    )
                else:
                    continue
            if len(cluster_sizes) > 0:
                statistic = np.mean(cluster_sizes)
            else:
                statistic = 0
        elif calculation_type == 'max_cluster':
            cluster_sizes = []
            for node in lattice_graph.nodes():
                if len(nx.node_connected_component(lattice_graph, node)) != 1:
                    cluster_sizes.append(
                        len(nx.node_connected_component(lattice_graph, node))
                    )
                else:
                    continue
            if len(cluster_sizes) > 0:
                statistic = np.max(cluster_sizes)
            else:
                statistic = 0
        elif calculation_type == 'fwd_bond_frac':
            n_fwd_bonds = 0
            for edge in lattice_graph.edges():
                if lattice_graph.nodes[edge[0]]['X'] != lattice_graph.nodes[edge[1]]['X']:
                    n_fwd_bonds += 1
                else:
                    continue
            if len(lattice_graph.edges()) == 0.0:
                statistic = 0
            else:
                statistic = float(n_fwd_bonds) / len(lattice_graph.edges())
        elif calculation_type == 'planar_bond_frac':
            planar = 0
            for edge in lattice_graph.edges():
                if lattice_graph.nodes[edge[0]]['X'] == lattice_graph.nodes[edge[1]]['X']:
                    planar += 1
                else:
                    continue
            if len(lattice_graph.edges()) == 0.0:
                statistic = 0
            else:
                statistic = float(planar)/len(lattice_graph.edges())
        else:
            statistic = None
        return statistic

    def get_data(self, agg=True):
        """
        Return the data in a pandas format
        :param agg: return means opposed to individual measurement.
        :return: pandas.DataFrame
        """
        if agg:
            return self._aggregate_data
        else:
            return self._data

    def get_error_data(self):
        """Get errors of the calculations"""
        return self._error_data

    def obtain_random_value(self):
        """Random number generator of class."""
        return np.random.randint(0, self.max_random_number)

    def determine_threshold(self,
                            probability=None,
                            n_repeats=None):
        """
        Determine the threshold probability of the model for percolation.
        :param probability: list of probabilities to sample at.
        :param n_repeats: number of repeats at each probability.
        :return: threshold probability where percolation happens at.
        """
        if probability is None:
            probabilities = np.linspace(0.1, 0.9, 41)
        else:
            if isinstance(probability, (np.ndarray, list,)):
                probabilities = np.array(probability)
            else:
                msg = "\'%s\' should be a float, array or list." % probability
                raise TypeError(msg)

        repeat_no = self._default_repeats if n_repeats is None else n_repeats
        percolations = []
        for p in probabilities:
            outcomes = []
            for i in range(repeat_no):
                lattice = PercolationLattice(dim=self._dim_tuple,
                                             p=p)
                is_percolation = lattice.assess_percolation()
                percolates = float(is_percolation)
                outcomes.append(percolates)
            percolations.append(np.mean(outcomes))

        results = spo.curve_fit(fermi_scaling_function,
                                probabilities,
                                percolations, p0=[0.5, 1])
        return results[0][0]

    def average_path_length(self, p, n_repeats=None):
        """
        Calculates the average path length at a certain probability.
        :param p: Probability to assess at.
        :param n_repeats: Number of repeats to perform.
        :return: average path length
        """
        repeat_no = self._default_repeats if n_repeats is None else n_repeats
        lengths = []
        for rep in range(repeat_no):
            lattice = PercolationLattice(dim=self._dim_tuple,
                                         p=p)
            is_percolation = lattice.assess_percolation()
            lattice_graph = lattice.get_lattice_graph()
            lattice_graph.add_node('Back')
            lattice_graph.add_node('Front')
            xdim = lattice_graph.graph['xdim']
            ydim = lattice_graph.graph['ydim']
            zdim = lattice_graph.graph['zdim']

            if is_percolation:
                for y in range(1, ydim + 1):
                    for z in range(1, zdim + 1):
                        lattice_graph.add_edge(tau(1, y, z), 'Back')
                        lattice_graph.add_edge(tau(xdim, y, z), 'Front')
                length = len(nx.shortest_path(lattice_graph,
                                              'Back',
                                              'Front')) - 2
                lengths.append(length)
            else:
                continue
        return [np.mean(lengths), sps.sem(lengths)]
