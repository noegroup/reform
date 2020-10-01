"""
test_simu_utils.py

Tests for the simulation interface. """

from reform import replica_exchange
from reform import omm
from reform import simu_utils
from simtk.openmm import app
import numpy as np


N_FRAMES = 500


def _prepare_capped_alanine_replicas(temps_intended) -> simu_utils.MultiTSimulation:
    """Return an `MultiTSimulation` object of capped alanine system (defined by `tests/spep_0000.pdb`) with given
    temperatures. Positions will be set according to the pdb file. """
    n_replicas = len(temps_intended)
    pdb = app.PDBFile("spep_0000.pdb")
    ff = app.ForceField("amber99sbildn.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             hydrogenMass=4 * omm.unit.amu)
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": 2.0}
    simu = simu_utils.MultiTSimulation(system, temps_intended, interface="single_threaded",
                                       integrator_params=integrator_params, verbose=False)
    simu.set_positions([pdb.getPositions()] * n_replicas)
    simu.minimize_energy()
    simu.set_velocities_to_temp()
    simu.run(2000)  # pre-equilibration
    return simu

def test_initiate_simulation():
    temps_intended = [300., 350.]
    simu = _prepare_capped_alanine_replicas(temps_intended)
