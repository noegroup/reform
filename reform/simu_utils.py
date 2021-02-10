"""
simu_utils.py
Implementation of some useful functions/data structures to help multi-context simulations, inspired by
`simtk.app.Simulation`.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from reform.omm import OMMTReplicas
from reform import replica_exchange


class SimulationHook(ABC):
    """Abstract class for defining functions to act on multiple contexts. It can be used to define:
    - a state reader/recorder (e.g., trajectory saver), or
    - something to change the simulation state (e.g., replica-exchanger).
    """

    @abstractmethod
    def action(self, context: OMMTReplicas) -> None:
        """This callback function will be called by a MultiTSimulation object.
        The multiple context will be passed to parameter `context`."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a description of self for more informative printing/debugging."""
        pass


class ReplicaExchangeHook(SimulationHook):
    """Perform replica exchange when called.
    """
    
    def __init__(self):
        self._ex_engine = None
    
    def action(self, context: OMMTReplicas) -> None:
        if self._ex_engine == None:
            # initiate the exchange engine, if it's not yet initiated
            self._ex_engine = replica_exchange.ReplicaExchange(context)
        self._ex_engine.perform_exchange()
    
    def __str__(self) -> str:
        return "Hook for replica exchange action."
    

class NpyRecorderHook(SimulationHook):
    """Recording the trajectory when called.
    """
    
    def __init__(self, saving_path, maximum_length, saving_interval=1000, dtype=np.float32):
        assert dtype is np.float32 or np.float64, "Unrecognized dtype, should be either numpy.float32 or numpy.float64!"
        self._dtype = dtype
        self._saving_path = saving_path
        self._max_len = maximum_length
        self._save_int = saving_interval
        self._count = -1
    
    def action(self, context: OMMTReplicas) -> None:
        curr_posis = context.get_all_positions_as_numpy()
        if self._count < 0:
            # initiate the array for storage, if it's not yet initiated
            self._count = 0
            self._n_replica = curr_posis.shape[0]
            self._n_atom = curr_posis.shape[1]
            self._traj = np.empty((self._n_replica, self._max_len, self._n_atom, 3), dtype=self._dtype)
        if self._count < self._max_len:
            self._traj[:, self._count, :, :] = curr_posis
            self._count += 1
        else:
            self.save()
            raise MemoryError("Buffer full! Current trajectory is stored at %s." % self._saving_path)
        if self._count % self._save_int == 0:
            self.save()
    
    def save(self) -> None:
        np.save(self._saving_path, self._traj[:, :self._count, :, :])
    
    def __del__(self):
        self.save()
    
    def __str__(self) -> str:
        return "Hook for storing trajectory to a npy file."


class MultiTSimulation:
    _context: OMMTReplicas
    _regular_hooks: List[Tuple[int, SimulationHook]]
    _verbose: bool
    _current_step: int
    _positions_set: bool

    def __init__(self, system: omm.System, temps: List[float], interface: str="single_threaded",
                 integrator_params: dict = {"integrator": "Langevin", "friction_in_inv_ps": 1.0,
                                            "time_step_in_fs": 2.0},
                 platform: str = "CPU", platform_prop=None, verbose=True):
        if platform_prop is None:
            platform_prop = {}
        if interface == "single_threaded":
            # reference implementation, single threaded, slow, but should work in any situation
            self._context = OMMTReplicas(system, temps, integrator_params, platform, platform_prop)
        elif interface == "replicated_system":
            # TODO: replica exchange simulation with multiple temperatures can also be implemented by replica systems
            raise NotImplementedError("TODO")
        elif interface == "":
            # TODO: this supports parallelization on multiple GPUs
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError("Unknown OpenMM interface.")

        self._regular_hooks = []
        self._update_interval_counter()
        self._verbose = verbose
        self._current_step = 0
        self._positions_set = False

    def register_regular_hook(self, hook: SimulationHook, interval: int):
        assert interval > 0, "Invalid interval: it should be a positive integer."
        self._regular_hooks.append((interval, hook))
        self._update_interval_counter()
        num = len(self._regular_hooks)
        if self._verbose:
            print("Hook #{:d}: {:s} is registered.".format(num - 1, str(hook)))

    def print_regular_hooks(self):
        for i, (interval, hook) in enumerate(self._regular_hooks):
            print("Hook #{:d}: {:s}, at an interval of {:d} time steps.".format(i, str(hook), interval))

    def remove_regular_hook(self, index: int) -> SimulationHook:
        assert 0 <= index < len(self._regular_hooks), "Given regular hook index does not exist!"
        _, hook = self._regular_hooks.pop(index)
        if self._verbose:
            print("Hook #{:d}: {:s} is removed.".format(index, str(hook)))
        self._update_interval_counter()
        return hook

    def _check_out_regular_hooks(self):
        """See if any of the regular hooks should be run. If yes, then call its `.action` method."""
        for i, (interval, hook) in enumerate(self._regular_hooks):
            if self._current_step % interval == 0:
                # time to call its callback function
                hook.action(self._context)
                if self._verbose:
                    print("Hook #{:d}: {:s} is called at Step {:d}.".format(i, str(hook), self._current_step))

    def minimize_energy(self, tolerance=10*unit.kilojoule/unit.mole, max_iterations: int = 100):
        """Minimize the potential energies locally for all replicas.
        Mimics the `minimizeEnergy` method of `simtk.openmm.app.Simulation`.
        `tolerance` sets the criterion for energy convergence.
        `max_iteration` sets the max number of iterations, if =0 then the minimization will continue until convergence.
        """
        assert self._positions_set, "System positions are not yet set."
        for i in range(self._context.num_replicas):
            self._context.minimize_energy(i, tolerance, max_iterations)

    def set_positions(self, positions):
        """Setting positions for all replicas. `positions` should be a list/np.ndarray that have the same number of
         elements as the number of replicas in the simulation context. Each element should correspond to the number of
         particles in the underlying MD system."""
        assert len(positions) == self._context.num_replicas, "Invalid shape of position input. Expecting the same " \
                                                             "amount as the number of replicas"\
                                                             "({:d}).".format(self._context.num_replicas)
        for i, posi in enumerate(positions):
            self._context.set_positions(i, posi)
        self._positions_set = True

    def set_velocities_to_temp(self):
        """Assign random velocities according to Maxwell-Boltzmann distribution according to the intended temperature.
        """
        assert self._positions_set, "System positions are not yet set."
        for i in range(self._context.num_replicas):
            self._context.set_velocities(i)

    def run(self, steps: int):
        """Running `steps` steps of simulation on all underlying replicas with the consideration of attached hooks."""
        assert self._positions_set, "System positions are not yet set."
        intended_stop = self._current_step + steps
        if self._verbose:
            print("{:d} steps (Step {:d} -> Step {:d}) will be run.".format(steps, self._current_step, intended_stop))
        # main loop for running simulations and checking hook intervals
        while True:
            next_steps = self._get_next_run_steps(intended_stop)
            if not next_steps:
                break
            else:
                self._context.step(next_steps)
                self._current_step += next_steps
                self._check_out_regular_hooks()
        if self._verbose:
            print("{:d} steps (Step {:d} -> Step {:d}) will be run.".format(steps, intended_stop - steps,
                                                                            self._current_step))

    def reset_step_counter(self) -> int:
        """Reset the step counter and return the current number.
        Can be useful for production run after equilibration."""
        current_step = self._current_step
        self._current_step = 0
        return current_step

    def _update_interval_counter(self):
        """Decide whether to use the the GCD strategy or calculate remaining steps for each time."""
        max_interval = 1000  # can be anything meaningfully as long as it's not too large that slows response of UI
        intervals = [interval for (interval, _) in self._regular_hooks]
        if intervals:
            gcd = np.gcd.reduce(intervals)
            if gcd > max_interval:
                self._interval_gcd = max_interval  # otherwise it's too large and blocks the UI
            elif gcd < 10 or gcd < min(intervals) / len(intervals):
                self._interval_gcd = 0  # it's not worthwhile in this case to use the GCD as simulation interval
        else:
            self._interval_gcd = max_interval  # when there's no hook in the simulation

    def _get_next_run_steps(self, intended_stop):
        """Calculate how many steps to go."""
        if self._interval_gcd:
            # using the greatest-common-divider as proposed
            steps_until_next_stop = self._interval_gcd - self._current_step % self._interval_gcd
        else:
            # consider all possible stops because of hook intervals
            remaining_steps = []
            for (interval, _) in self._regular_hooks:
                steps_until_next_hook = interval - self._current_step % interval
                remaining_steps.append(steps_until_next_hook)
            steps_until_next_stop = min(remaining_steps)
        # now check if our `intended_stop` arrives earlier than the calculated stop
        return min(steps_until_next_stop, intended_stop - self._current_step)
