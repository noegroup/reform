from reform import omm
from reform import omm_replicated
from simtk.openmm import app

PDB_FILE = "spep_0000.pdb"

# get correct path to the pdb file
import os
test_dir = os.path.abspath(os.path.dirname(__file__))
PDB_PATH = os.path.join(test_dir, PDB_FILE)


def prepare_capped_alanine_replicas(temps_intended, replicated=True) -> omm.OMMTReplicas:
    """Return an `OMMTReplicas` object of capped alanine system (defined by `tests/spep_0000.pdb`) with given
    temperatures. Positions will be set according to the pdb file. Using replicated systems when `replicated`==True."""
    pdb = app.PDBFile(PDB_PATH)
    ff = app.ForceField("amber99sbildn.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             hydrogenMass=4 * omm.unit.amu)
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": 2.0}
    
    # establish the system and set positions
    if replicated:
        replicas = omm_replicated.OMMTReplicas_replicated(system, temps_intended,
                                                          integrator_params=integrator_params)
        replicas.set_positions_all([pdb.getPositions()] * len(temps_intended))
    else:
        replicas = omm.OMMTReplicas(system, temps_intended, integrator_params=integrator_params)
        for i in range(replicas.num_replicas):
            replicas.set_positions(i, pdb.getPositions())
    # set velocities
    for i in range(replicas.num_replicas):
        replicas.set_velocities(i)  # will be redirected to a OpenMM internal function to set system velocities to temp
    replicas.step(2000)  # pre-equilibration
    return replicas
