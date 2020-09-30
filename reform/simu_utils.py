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


class MultiTSimulation:
    _context: OMMTReplicas
    _regular_hooks: List[Tuple[int, SimulationHook]]
    _verbose: bool
    _current_step: int

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

    def run(self, steps: int):
        """Running `steps` steps of simulation on all underlying replicas with the consideration of attached hooks."""
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
