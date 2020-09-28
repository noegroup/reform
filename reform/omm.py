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

    def get_instantaneous_temp(self, index: int) -> float:
        """Return the instantaneous temperature of replica #index"""
        self._check_index(index)
        state: omm.State = self._contexts[index].getState(getEnergy=True)
        e_k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        temp = e_k * 2 / self._n_DoF / self._k
        return temp
