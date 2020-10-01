"""
omm.py
Implementation of the MultiTReplicas with a pool of OpenMM contexts (single-threaded OpenMM interface).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from reform.replica_exchange import MultiTReplicas


class OMMTReplicas(MultiTReplicas):
    _contexts: List[omm.Context]  # OpenMM contexts for all replicas
    _n_DoF: int  # number of degrees of freedom

    def __init__(self, system: omm.System, temps: List[float],
                 integrator_params: dict = {"integrator": "Langevin", "friction_in_inv_ps": 1.0,
                                            "time_step_in_fs": 2.0},
                 platform: str = "CPU", platform_prop={}):
        self._system = system
        self._n_DoF = self._get_n_dof()
        self._temps = temps
        self._N = len(temps)
        self._k = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA) \
            .value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

        self._integrator_params = integrator_params
        self._platform = omm.Platform.getPlatformByName(platform)
        self._platform_prop = platform_prop
        self._integrators, self._contexts = self._create_contexts()

    def _get_n_dof(self) -> int:
        n_dof = self._system.getNumParticles() * 3 - self._system.getNumConstraints()
        has_CMM_remover = False
        for f in self._system.getForces():
            if type(f) is omm.CMMotionRemover:
                has_CMM_remover = True
        if has_CMM_remover:
            n_dof -= 3  # when center of mass is constrained
        return n_dof

    def _create_contexts(self):
        integrators = []
        contexts = []
        if self._integrator_params["integrator"] != "Langevin":
            raise NotImplementedError(f"Integrator {self._integrator_params['integrator']} is not implemented.")
        friction = self._integrator_params["friction_in_inv_ps"] / unit.picosecond
        time_step = self._integrator_params["time_step_in_fs"] * unit.femtoseconds
        for i in range(self._N):
            integrator = omm.LangevinIntegrator(self._temps[i] * unit.kelvin, friction, time_step)
            if "constraint_tolerance" in self._integrator_params.keys():
                integrator.setConstraintTolerance(self._integrator_params["constraint_tolerance"])
            else:
                integrator.setConstraintTolerance(0.00001)
            context = omm.Context(self._system, integrator, self._platform, self._platform_prop)
            integrators.append(integrator)
            contexts.append(context)
        return integrators, contexts

    def set_positions(self, index: int, positions):
        """Set positions for replica #index. `positions` should be array-like and corresponding to the number of
        particles in the system. """
        self._check_index(index)
        self._contexts[index].setPositions(positions)

    def minimize_energy(self, index: int, tolerance=10*unit.kilojoule/unit.mole, max_iterations: int = 0):
        """Use openmm.LocalEnergyMinimizer to perform a local energy minimization. `max_iteration` should >=0,
        =0 means that the minimization will continue until the potential energy converges within the given `tolerance`.
        """
        self._check_index(index)
        assert max_iterations >= 0, "Invalid number for iterations, should be non-negative."
        omm.LocalEnergyMinimizer.minimize(self._contexts[index], tolerance, max_iterations)

    def set_velocities(self, index: int, velocities=None):
        """Set velocities for replica #index. If `velocities` is None, then a set of random velocities according to
        desired temperatures will be set. Otherwise, `velocities` should be array-like and corresponding to the
        number of particles in the system. """
        self._check_index(index)
        if velocities:
            self._contexts[index].setVelocities(velocities)
        else:
            self._contexts[index].setVelocitiesToTemperature(self._temps[index])

    def get_potential(self, index: int) -> float:
        """Return the potential energy of replica #index."""
        self._check_index(index)
        state: omm.State = self._contexts[index].getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _vel_scaling_factor(self, ori_index, target_index) -> float:
        """Return a scaling factor for exchanging velocities from ."""
        if self._integrator_params["integrator"] != "Langevin":
            raise NotImplementedError(f"Scaling with integrator {self._integrator_params['integrator']} is not yet "
                                      f"supported.")
        return np.sqrt(self._temps[target_index] / self._temps[ori_index])

    def exchange_pair(self, pair: Tuple[int, int]):
        """Perform exchange of the given two replicas. Scale the velocities when necessary."""
        a, b = pair
        self._check_index(a)
        self._check_index(b)
        state_a = self._contexts[a].getState(getPositions=True, getVelocities=True)
        state_b = self._contexts[b].getState(getPositions=True, getVelocities=True)

        self._contexts[b].setPositions(state_a.getPositions())
        self._contexts[b].setVelocities(state_a.getVelocities() * self._vel_scaling_factor(a, b))
        self._contexts[a].setPositions(state_b.getPositions())
        self._contexts[a].setVelocities(state_b.getVelocities() * self._vel_scaling_factor(b, a))

    def step(self, steps: int):
        """Run each integrator for `steps` steps."""
        for i in range(self._N):
            self._integrators[i].step(steps)

    def get_state(self, index: int, getPositions=False, getVelocities=False, getForces=False, getEnergy=False) \
            -> omm.State:
        """Return a simtk.openmm.State object corresponding to replica #index"""
        self._check_index(index)
        state = self._contexts[index].getState(getPositions=getPositions, getVelocities=getVelocities,
                                               getForces=getForces, getEnergy=getEnergy)
        return state

    def get_states(self, getPositions=False, getVelocities=False, getForces=False, getEnergy=False) -> List[omm.State]:
        """Return a list of simtk.openmm.State object corresponding to all replicas"""
        return [self._contexts[i].getState(getPositions=getPositions, getVelocities=getVelocities,
                                           getForces=getForces, getEnergy=getEnergy) for i in range(self._N)]

    def get_instantaneous_temp(self, index: int) -> float:
        """Return the instantaneous temperature of replica #index"""
        state = self.get_state(getEnergy=True)
        e_k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        temp = e_k * 2 / self._n_DoF / self._k
        return temp

    def get_positions(self, index: int) -> unit.Quantity:
        """Return the positions of context #index in OpenMM internal form."""
        return self.get_state(index, getPositions=True).getPositions(asNumpy=True)

    def get_velocities(self, index: int) -> unit.Quantity:
        """Return the velocities of context #index in OpenMM internal form."""
        return self.get_state(index, getVelocities=True).getVelocities(asNumpy=True)

    def get_all_positions_as_numpy(self, length_unit: str = "nm") -> np.ndarray:
        """Return a numpy array of shape [self._N, number_of_particles, 3] containing particle positions from all
        replicas in given `length_unit` (can be "nm" or "angstrom")."""
        if length_unit == "nm":
            l_unit = unit.nano * unit.meter
        elif length_unit == "angstrom":
            l_unit = unit.angstrom
        else:
            raise ValueError("Unknown unit!")
        states = self.get_states(getPositions=True)
        output = []
        for state in states:
            output.append(state.getPositions(asNumpy=True).value_in_unit(l_unit))
        return np.array(output)

    def save_states(self, filepath: str = "./omm_chkpt.npz"):
        """Save the current positions and velocities into a NumPy binary archive at given `filepath`."""
        posis = np.array([self.get_positions(i)._value for i in range(self._N)])
        velos = np.array([self.get_velocities(i)._value for i in range(self._N)])
        np.savez(filepath, set_temps=self._temps, positions=posis, velocities=velos)

    def load_states(self, filepath: str = "./omm_chkpt.npz", check_temps=True):
        """Load positions and velocities from a NumPy binary archive at given `filepath` to contexts."""
        chkpt = np.load(filepath)
        if check_temps:
            assert np.allclose(chkpt["set_temps"], self._temps), "Preset temperatures in the checkpoint are different" \
                                                                 " from current OMMTReplicas object."
        assert self._N == len(chkpt["positions"]) and self._N == len(chkpt["velocities"]), \
            "Checkpoint file is invalid: number of replicas is not consistent inside the file."
        assert len(chkpt["set_temps"]) == len(self._temps), "Checkpoint file is incompatible: number of replicas is " \
                                                            "not consistent with the current OMMTReplicas object."
        assert chkpt["positions"].shape[2] == self._system.getNumParticles(), "Number of particles in the checkpoint " \
                                                                              "file is inconsistent with the current " \
                                                                              "OpenMM system."
        # after checking we can load the positions and velocities
        for i in range(self._N):
            self.set_positions(i, chkpt["positions"][i])
            self.set_velocities(i, chkpt["velocities"][i])
