"""
test_simu_utils.py

Tests for the simulation interface. """

from reform import replica_exchange
from reform import omm
from reform import simu_utils
from simtk.openmm import app
import numpy as np

PDB_FILE = "spep_0000.pdb"
MODEL_NAME = "central.pt"
N_FRAMES = 10
EXCHANGE_INTERVAL = 100
RECORDING_INTERVAL = 200

# get correct path to the pdb file
import os
test_dir = os.path.abspath(os.path.dirname(__file__))
PDB_PATH = os.path.join(test_dir, PDB_FILE)
MODEL_PATH = os.path.join(test_dir, MODEL_NAME)


def _prepare_capped_alanine_simu_torch(temps_intended, torch_force_func, interface="single_threaded") -> simu_utils.MultiTSimulation:
    """Return an `MultiTSimulation` object of capped alanine system (defined by `tests/spep_0000.pdb`) with given
    temperatures. Positions will be set according to the pdb file. """
    n_replicas = len(temps_intended)
    pdb = app.PDBFile(PDB_PATH)
    ff = app.ForceField("amber99sbildn.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             hydrogenMass=4 * omm.unit.amu)
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": 2.0}

    # load torch force
    f = torch_force_func(MODEL_PATH)
    
    if interface == "single_threaded":
        system.addForce(f)
        simu = simu_utils.MultiTSimulation(system, temps_intended, interface=interface, platform="CUDA",
                                           integrator_params=integrator_params, verbose=False)
    elif interface == "replicated_system":
        simu = simu_utils.MultiTSimulation(system, temps_intended, replicated_system_additional_forces=[f],
                                           interface=interface, platform="CUDA",
                                           integrator_params=integrator_params, verbose=False)
    simu.set_positions([pdb.getPositions()] * n_replicas)
    # simu.minimize_energy() # not suitable for NN force
    simu.set_velocities_to_temp()
    simu.run(2000)  # pre-equilibration
    return simu

import pytest
@pytest.mark.parametrize("interface", ["replicated_system", "single_threaded"])
def test_full_simulation_simu_w_torch(interface):
    try:
        from openmmtorch import TorchForce
    except ImportError as e:
        pytest.skip("No openmm-torch package installed. Skipping.")
        return  # module doesn't exist, skip the test.

    temps_intended = [300., 350.]
    simu = _prepare_capped_alanine_simu_torch(temps_intended, TorchForce, interface)
    total_steps = N_FRAMES * EXCHANGE_INTERVAL
    simu.register_regular_hook(simu_utils.ReplicaExchangeHook(), EXCHANGE_INTERVAL)
    recorder_hook = simu_utils.NpyRecorderHook("/tmp/00100.npy", int(total_steps / RECORDING_INTERVAL), 100)
    simu.register_regular_hook(recorder_hook, RECORDING_INTERVAL)
    #simu.print_regular_hooks()
    simu.run(int(total_steps / 2))
    simu.save_chkpt("/tmp/00100_chk.npz")
    simu.load_chkpt("/tmp/00100_chk.npz")
    simu.run(total_steps - int(total_steps / 2))
    recorder_hook.save()
    recorded = np.load("/tmp/00100.npy")
    assert len(recorded.shape) == 4, "Unrecognized shape for recorded trajectories."
    assert recorded.shape[0] == len(temps_intended), "Wrong number of replicas in recorded trajectories."

