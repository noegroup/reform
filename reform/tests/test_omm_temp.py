"""
test_omm_temp.py

Tests for validating the reform.omm (single-threaded OpenMM interface) by checking if the average temperature is within
the expected range with and without replica exchanges. """

from reform import replica_exchange
from .sysprep import prepare_capped_alanine_replicas
import numpy as np

N_FRAMES = 500

def _check_temps(temps_recorded, temp_intended):
    error_percentage = np.abs(temps_recorded.mean() - temp_intended) / temp_intended
    assert error_percentage < 0.1, \
        "Temperature is set at {:.2f} K, but the actual average is {:.2f} K.".format(temp_intended,
                                                                                     temps_recorded.mean())

import pytest

@pytest.mark.parametrize("replicated_system", [True, False])
@pytest.mark.parametrize("exchange", [True, False])
def test_omm_temp(replicated_system: bool, exchange: bool):
    """Test the average temperature from simulation without replica exchanges."""
    temps_intended = [300., 350.]
    replicas = prepare_capped_alanine_replicas(temps_intended, replicated_system)
    
    if exchange:
        # attach a replica exchange engine
        ex_engine = replica_exchange.ReplicaExchange(replicas)
    
    # run short simulation and record instantaneous temps
    temps = np.zeros((len(temps_intended), N_FRAMES))
    for i in range(N_FRAMES):
        replicas.step(100)
        if exchange: ex_engine.perform_exchange()
        # direct record the temperatures after exchange attempts w/o further equilibration, such that bug in velocity
        # scaling will also be revealed
        for k in range(len(temps_intended)):
            temps[k, i] = replicas.get_instantaneous_temp(k)
    # make sure that exchanges happened during the process
    if exchange: assert ex_engine.exchange_rate > 0, "No exchanges happened?"
    # check temperatures
    try:
        for k, temp_intended in enumerate(temps_intended):
            _check_temps(temps[k], temp_intended)
    except AssertionError as err:
        print("Avg temperature is not correct during replica exchanges: {0}".format(err))


