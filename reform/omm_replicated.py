"""
omm_replicate.py
Implementation of the OMMTReplicas with replicated OpenMM contexts. This enables batch-evaluation of systems with certain neural-network force field.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from reform.omm import OMMTReplicas
from reform.replicated_system import ReplicatedSystem, get_custom_langevin_integrator


class OMMTReplicas_replicated(OMMTReplicas):
    def __init__(self, system: omm.System, temps: List[float],
                 integrator_params: dict = {"integrator": "Langevin", "friction_in_inv_ps": 1.0,
                                            "time_step_in_fs": 2.0},
                 platform: str = "CPU", platform_prop={}):
        # check and remove CMMotionRemover if it exists
        # since we don't use it
        no_cmm = None
        for i, f in enumerate(system.getForces()):
            if isinstance(f, omm.openmm.CMMotionRemover):
                no_cmm = i
                break
        if no_cmm is not None:
            system.removeForce(no_cmm)
        
        super(OMMTReplicas_replicated, self).__init__(system, temps, integrator_params, platform,
                                                      platform_prop)
        """
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
        """

    def _get_n_dof(self) -> int:
        n_dof = self._system.getNumParticles() * 3 - self._system.getNumConstraints()
        # replicated systems don't have CMM removers
        return n_dof

    def _create_contexts(self):
        """Modified for the replicated system:
        for simulation there is only one context and only one integrator for the replicated system, which contains
        all replicas.
        For replica exchange and energy computation, we need to calculate the potential energies. At current stage,
        we rely on a separate context for single-replica system to evaluate this energy. Therefore, here we also initiate
        a """
        
        if self._integrator_params["integrator"] != "Langevin":
            raise NotImplementedError(f"Integrator {self._integrator_params['integrator']} is not implemented.")

        # initiate the single-replica context (for potential evaluation)
        friction = self._integrator_params["friction_in_inv_ps"] / unit.picosecond
        time_step = self._integrator_params["time_step_in_fs"] * unit.femtoseconds
        self._single_integrator = omm.LangevinIntegrator(self._temps[0] * unit.kelvin, friction, time_step)
        if "constraint_tolerance" in self._integrator_params.keys():
            self._single_integrator.setConstraintTolerance(self._integrator_params["constraint_tolerance"])
        else:
            self._single_integrator.setConstraintTolerance(0.00001)
        self._single_context = omm.Context(self._system, self._single_integrator, self._platform, self._platform_prop)
        self._single_context_positions_set = False
        
        # initiate the replicated context (for simulation)
        self._replicated_system = ReplicatedSystem.replicate_system(self._system, self._N)
        temperatures_per_dof = np.concatenate([np.ones((self._system.getNumParticles(), 3)) * temp for temp in self._temps])
        integrator = get_custom_langevin_integrator(temperatures_per_dof, self._integrator_params["friction_in_inv_ps"],
                                                    self._integrator_params["time_step_in_fs"] * 0.001)
        #integrator = omm.LangevinIntegrator(self._temps[1] * unit.kelvin, friction, time_step)
        if "constraint_tolerance" in self._integrator_params.keys():
            integrator.setConstraintTolerance(self._integrator_params["constraint_tolerance"])
        else:
            integrator.setConstraintTolerance(0.00001)
        context = omm.Context(self._replicated_system, integrator, self._platform, self._platform_prop)
        
        ## we use an array form to keep some api consistency with the base class
        integrators = [integrator]
        contexts = [context]
        return integrators, contexts


    def get_state(self, index: int, getPositions=False, getVelocities=False, getForces=False, getEnergy=False) \
            -> omm.State:
        """Not efficient for the replicated system. Deprecated."""
        return NotImplemented

    def get_states(self, getPositions=False, getVelocities=False, getForces=False, getEnergy=False) -> List[omm.State]:
        """Not efficient for the replicated system. Deprecated."""
        return NotImplemented

    def get_rep_state(self, getPositions=False, getVelocities=False, getForces=False, getEnergy=False) -> omm.State:
        return self._contexts[0].getState(getPositions=getPositions, getVelocities=getVelocities,
                                          getForces=getForces, getEnergy=getEnergy)

    def get_positions_all(self) -> unit.Quantity:
        return self.get_rep_state(getPositions=True).getPositions(asNumpy=True).reshape((self._N, -1, 3))
    
    def get_all_positions_as_numpy(self, length_unit: str = "nm") -> np.ndarray:
        """Return a numpy array of shape [self._N, number_of_particles, 3] containing particle positions from all
        replicas in given `length_unit` (can be "nm" or "angstrom")."""
        if length_unit == "nm":
            l_unit = unit.nano * unit.meter
        elif length_unit == "angstrom":
            l_unit = unit.angstrom
        else:
            raise ValueError("Unknown unit!")
        output = self.get_positions_all().value_in_unit(l_unit)
        return output
    
    def set_positions_all(self, positions_all):
        self._contexts[0].setPositions(np.array(positions_all).reshape((-1, 3)))
        if not self._single_context_positions_set:
            self._single_context.setPositions(self.get_positions(0))
            self._single_context_positions_set = True

    def get_positions(self, index: int) -> unit.Quantity:
        """Return the positions of context #index in OpenMM internal form."""
        return self.get_positions_all()[index]

    def set_positions(self, index: int, positions):
        """Set positions for replica #index. `positions` should be array-like and corresponding to the number of
        particles in the system. """
        self._check_index(index)
        positions_all = self.get_positions_all()
        positions_all[index, :, :] = positions
        self.set_positions_all(positions_all)

    def get_velocities_all(self) -> unit.Quantity:
        return self.get_rep_state(getVelocities=True).getVelocities(asNumpy=True).reshape((self._N, -1, 3))
    
    def set_velocities_all(self, velocities_all):
        self._contexts[0].setVelocities(np.array(velocities_all).reshape((-1, 3)))

    def get_velocities(self, index: int) -> unit.Quantity:
        """Return the velocities of context #index in OpenMM internal form."""
        return self.get_velocities_all()[index]
    
    def set_velocities(self, index: int, velocities=None):
        """Set velocities for replica #index. If `velocities` is None, then a set of random velocities according to
        desired temperatures will be set. Otherwise, `velocities` should be array-like and corresponding to the
        number of particles in the system. """
        self._check_index(index)
        if velocities is not None:
            velocities_all = self.get_velocities_all()
            velocities_all[index, :, :] = velocities
            self.set_velocities_all(velocities_all)
        else:
            self._single_context.setVelocitiesToTemperature(self._temps[index])
            velocities = self._single_context.getState(getVelocities=True).getVelocities(asNumpy=True)
            self.set_velocities(index, velocities)

    def minimize_energy_all(self, tolerance=10*unit.kilojoule/unit.mole, max_iterations: int = 0):
        """Use openmm.LocalEnergyMinimizer to perform a local energy minimization. `max_iteration` should >=0,
        =0 means that the minimization will continue until the potential energy converges within the given `tolerance`.
        """
        assert max_iterations >= 0, "Invalid number for iterations, should be non-negative."
        omm.LocalEnergyMinimizer.minimize(self._contexts[0], tolerance, max_iterations)

    def minimize_energy(self, index: int, tolerance=10*unit.kilojoule/unit.mole, max_iterations: int = 0):
        """Use openmm.LocalEnergyMinimizer to perform a local energy minimization. `max_iteration` should >=0,
        =0 means that the minimization will continue until the potential energy converges within the given `tolerance`.
        """
        self._check_index(index)
        assert max_iterations >= 0, "Invalid number for iterations, should be non-negative."
        self._single_context.setPositions(self.get_positions(index))
        omm.LocalEnergyMinimizer.minimize(self._single_context, tolerance, max_iterations)
        minimized_posi = self._single_context.getState(getPositions=True).getPositions(asNumpy=True)
        self.set_positions(index, minimized_posi)

    def get_potential(self, index: int) -> float:
        """Return the potential energy of replica #index."""
        self._check_index(index)
        self._single_context.setPositions(self.get_positions(index))
        state: omm.State = self._single_context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def exchange_pair(self, pair: Tuple[int, int]):
        """Perform exchange of the given two replicas. Scale the velocities when necessary."""
        a, b = pair
        self._check_index(a)
        self._check_index(b)

        posi_all = self.get_positions_all()._value
        velo_all = self.get_velocities_all()._value
        middle = np.empty((posi_all.shape[1], posi_all.shape[2]), dtype=posi_all.dtype)
        # swap positions
        middle[:, :] = posi_all[b]
        posi_all[b, :, :] = posi_all[a, :, :]
        posi_all[a, :, :] = middle
        #print(posi_all)
        # swap velocities with scaling
        middle[:, :] = velo_all[b]
        velo_all[b, :, :] = velo_all[a, :, :] * self._vel_scaling_factor(a, b)
        velo_all[a, :, :] = middle * self._vel_scaling_factor(b, a)
        
        self.set_positions_all(posi_all)
        self.set_velocities_all(velo_all)

    def step(self, steps: int):
        """Run the replicated integrator for `steps` steps."""
        #for i in range(self._N):
        #    self._integrators[i].step(steps)
        self._integrators[0].step(steps)

    def get_instantaneous_temp(self, index: int) -> float:
        """Return the instantaneous temperature of replica #index"""
        self._check_index(index)
        self._single_context.setPositions(self.get_positions(index))
        self._single_context.setVelocities(self.get_velocities(index))
        state = self._single_context.getState(getEnergy=True)
        e_k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        temp = e_k * 2 / self._n_DoF / self._k
        return temp

    def save_states(self, filepath: str = "./omm_chkpt.npz"):
        """Save the current positions and velocities into a NumPy binary archive at given `filepath`."""
        posis = self.get_positions_all()._value
        velos = self.get_velocities_all()._value
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
        assert chkpt["positions"].shape[1] == self._system.getNumParticles(), "Number of particles in the checkpoint " \
                                                                              "file is inconsistent with the current " \
                                                                              "OpenMM system."
        # after checking we can load the positions and velocities
        self.set_positions_all(chkpt["positions"])
        self.set_velocities_all(chkpt["velocities"])

