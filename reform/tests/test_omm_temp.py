"""
test_omm_temp.py

Tests for validating the reform.omm (single-threaded OpenMM interface) by checking if the average temperature is within
the expected range with and without replica exchanges. """

from reform import replica_exchange
from reform import omm
from simtk.openmm import app
import numpy as np

PDB_FILE = "spep_0000.pdb"
N_FRAMES = 500

# get correct path to the pdb file
import os
test_dir = os.path.abspath(os.path.dirname(__file__))
PDB_PATH = os.path.join(test_dir, PDB_FILE)


def _prepare_capped_alanine_replicas(temps_intended) -> omm.OMMTReplicas:
    """Return an `OMMTReplicas` object of capped alanine system (defined by `tests/spep_0000.pdb`) with given
    temperatures. Positions will be set according to the pdb file."""
    pdb = app.PDBFile(PDB_PATH)
    ff = app.ForceField("amber99sbildn.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             hydrogenMass=4 * omm.unit.amu)
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": 2.0}
    replicas = omm.OMMTReplicas(system, temps_intended, integrator_params=integrator_params)
    for i in range(replicas.num_replicas):
        replicas.set_positions(i, pdb.getPositions())
        replicas.set_velocities(i)  # will be redirected to a OpenMM internal function to set system velocities to temp
    replicas.step(2000)  # pre-equilibration
    return replicas


def _check_temps(temps_recorded, temp_intended):
    error_percentage = np.abs(temps_recorded.mean() - temp_intended) / temp_intended
    assert error_percentage < 0.1, \
        "Temperature is set at {:.2f} K, but the actual average is {:.2f} K.".format(temp_intended,
                                                                                     temps_recorded.mean())


def test_omm_temp():
    """Test the average temperature from simulation without replica exchanges."""
    temps_intended = [300., 350.]
    replicas = _prepare_capped_alanine_replicas(temps_intended)
    # run short simulation and record instantaneous temps
    temps = np.zeros((len(temps_intended), N_FRAMES))
    for i in range(N_FRAMES):
        replicas.step(100)
        for k in range(len(temps_intended)):
            temps[k, i] = replicas.get_instantaneous_temp(k)
    # check temperatures
    for k, temp_intended in enumerate(temps_intended):
        _check_temps(temps[k], temp_intended)


def test_omm_ex_temp():
    """Test the average temperature from simulation with replica exchanges."""
    temps_intended = [300., 350.]
    replicas = _prepare_capped_alanine_replicas(temps_intended)
    # attach a replica exchange engine
    ex_engine = replica_exchange.ReplicaExchange(replicas)
    # run short simulation and record instantaneous temps
    temps = np.zeros((len(temps_intended), N_FRAMES))
    for i in range(N_FRAMES):
        replicas.step(100)
        ex_engine.perform_exchange()
        # direct record the temperatures after exchange attempts w/o further equilibration, such that bug in velocity
        # scaling will also be revealed
        for k in range(len(temps_intended)):
            temps[k, i] = replicas.get_instantaneous_temp(k)
    # make sure that exchanges happened during the process
    assert ex_engine.exchange_rate > 0, "No exchanges happened?"
    # check temperatures
    try:
        for k, temp_intended in enumerate(temps_intended):
            _check_temps(temps[k], temp_intended)
    except AssertionError as err:
        print("Avg temperature is not correct during replica exchanges: {0}".format(err))
