"""
test_replica_exchange.py
Tests for validating the core functionality of replica exchange.
"""

from typing import List, Tuple
import numpy as np
from reform.replica_exchange import MultiTReplicas, ReplicaExchange


class ExampleMultiTContext(MultiTReplicas):
    """An abstract class for the target interfaces of the replica exchange functionality."""

    def __init__(self, temps: List[float], k_B: float = 1.0):
        self._temps = temps
        self._N = len(temps)
        self._k = k_B  # Boltzmann constant, should be compatible in unit with the potential function
        self._potentials = np.zeros(self._N)

    def set_potentials(self, potentials: List[float]) -> None:
        assert len(potentials) == self._N
        self._potentials = potentials

    def get_potential(self, num: int) -> float:
        """Return the potential energy of replica #num."""
        return self._potentials[num]

    def exchange_pair(self, pair: Tuple[int, int]):
        """Perform exchange of the given two replicas. Scale the velocities when necessary."""
        # print("Context: the following pair is exchanged: {:d}, {:d}".format(pair[0], pair[1]))
        pass


class TestReplicaExchangeProposing:
    def test_choose_one(self):
        re = ReplicaExchange(ExampleMultiTContext(list(range(3))), proposing_mode="one_at_a_time")
        results = []
        for i in range(50):
            results += re._proposing_pairs()
        assert len(results) == 50, "Not proposing one at a time!"
        # when everything's going correctly, the probability of not passing this assertion is 2 * 0.5^50 \approx 0
        assert set(results) == {(0, 1), (1, 2)}, "Pairs not correct!"

    def test_even_odd_alternative(self):
        re = ReplicaExchange(ExampleMultiTContext(list(range(4))), proposing_mode="even_odd_alternative")
        for i in range(50):
            if i % 2 == 0:
                # now we should get even pairs
                assert re._proposing_pairs() == [(0, 1), (2, 3)], "Even pairs not correct!"
            else:
                assert re._proposing_pairs() == [(1, 2)], "Odd pairs not correct!"


class TestReplicaExchangeExchanging:
    def test_should_always_exchange(self):
        context = ExampleMultiTContext([1., 2., 4.])
        context.set_potentials([0., 0., 0.])
        re = ReplicaExchange(context, proposing_mode="one_at_a_time")
        for i in range(50):
            re.perform_exchange()
        assert re.exchange_rate == 1., "There's something wrong with the criterion."

    def test_exchange(self):
        """This is a test case where the exchange rate should be around 1/e."""
        context = ExampleMultiTContext([1., 2., 4.])
        context.set_potentials([-2., 0., 4.])
        re = ReplicaExchange(context, proposing_mode="one_at_a_time")
        for i in range(1000):
            re.perform_exchange()
        assert np.abs(re.exchange_rate - 1. / np.e) < 0.05, "There's something wrong with the criterion."
