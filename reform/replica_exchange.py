"""
replica_exchange.py
Implementation of core functionality of replica exchange and definition of the interface of multiple replicas.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class MultiTReplicas(ABC):
    """An abstract class defining the interfaces for replica exchange functionality to act on."""
    _temps: List[float]  # temperatures of replicas
    _N: int  # number of replicas
    _k: float  # Boltzmann constant, should be compatible in unit with the potential function

    @property
    def num_replicas(self) -> int:
        return self._N

    @property
    def boltzmann_constant(self) -> float:
        return self._k

    def get_temp(self, index: int) -> float:
        """Return the preset temperature for replica #index."""
        return self._temps[index]

    def _check_index(self, index: int) -> None:
        """Check if the given index is in range [0, _N)."""
        if type(index) is not int or index < 0 or index >= self._N:
            raise IndexError(f"Given index {index} does not correspond to any replica.")

    @abstractmethod
    def get_potential(self, index: int) -> float:
        """Return the potential energy of replica #index."""
        return 0.0

    @abstractmethod
    def exchange_pair(self, pair: Tuple[int, int]):
        """Perform exchange of the given two replicas. Scale the velocities when necessary."""
        pass


class ReplicaExchange:
    """Core class for replica exchange functionalities.
    """
    _n_attempts: int
    _n_exchanges: int

    def __init__(self, context: MultiTReplicas, proposing_mode="one_at_a_time"):
        self.context = context
        self.N_replicas = self.context.num_replicas
        # set the mode of proposing the pairs
        if proposing_mode == "even_odd_alternative":
            self._one_at_a_time = False
            self._even_pairs = [(i, i + 1) for i in np.arange(self.N_replicas)[:-1:2]]
            self._odd_pairs = [(i, i + 1) for i in np.arange(self.N_replicas)[1:-1:2]]
            self._propose_even_pairs = True
        else:
            # proposing one random neighboring pair at a time
            self._one_at_a_time = True
        self._n_attempts = 0
        self._n_exchanges = 0

    @property
    def num_attempts(self) -> int:
        return self._n_attempts

    @property
    def num_exchanges(self) -> int:
        return self._n_exchanges

    @property
    def exchange_rate(self) -> float:
        if self._n_attempts:
            return self._n_exchanges / self._n_attempts
        else:
            return 0.

    # defining the pair datatype
    Pair = Tuple[int, int]

    def _choose_one(self) -> Pair:
        """Propose one random neighboring pair."""
        place = np.random.rand() * (self.N_replicas - 1)
        return int(np.floor(place)), int(np.ceil(place))

    def _proposing_pairs(self) -> List[Pair]:
        if self._one_at_a_time:
            return [self._choose_one()]
        else:
            # we will propose the even and odd pairs alternatively
            if self._propose_even_pairs:
                self._propose_even_pairs = False
                return self._even_pairs
            else:
                self._propose_even_pairs = True
                return self._odd_pairs

    def _check_pairs(self, proposed_pairs: List[Pair]) -> List[Pair]:
        k_B = self.context.boltzmann_constant
        pairs2go: List[ReplicaExchange.Pair] = []  # for collecting accepted pairs
        # filter the proposed pairs by the criterion
        xs = np.random.rand(len(proposed_pairs))
        for i, pair in enumerate(proposed_pairs):
            a, b = pair
            u_a, u_b = self.context.get_potential(a), self.context.get_potential(b)
            inv_t_a, inv_t_b = 1 / (k_B * self.context.get_temp(a)), 1 / (k_B * self.context.get_temp(b))
            p_pair = np.exp((u_a - u_b) * (inv_t_a - inv_t_b))
            if xs[i] < p_pair:
                pairs2go.append(pair)
        return pairs2go

    def perform_exchange(self) -> None:
        proposed_pairs = self._proposing_pairs()
        self._n_attempts += len(proposed_pairs)
        pairs_to_exchange = self._check_pairs(proposed_pairs)
        for pair in pairs_to_exchange:
            self.context.exchange_pair(pair)
        self._n_exchanges += len(pairs_to_exchange)
