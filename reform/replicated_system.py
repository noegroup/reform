"""
replicated_system.py
Systems for batch evaluation. (low-level APIs)
Originally implemented as part of the `openmmsystems` project by Andreas Kraemer.
Yaoyi Chen added the CHARMM support and integrated it to the `reform` package.
"""

import copy
import numpy as np

from simtk import unit
from simtk.openmm import (
    openmm,
    System,
    LocalCoordinatesSite,
    OutOfPlaneSite,
    TwoParticleAverageSite,
    ThreeParticleAverageSite,
    HarmonicBondForce,
    HarmonicAngleForce,
    PeriodicTorsionForce,
    NonbondedForce,
    CustomBondForce,
    # for supporting CHARMM22*
    CustomTorsionForce,
    CMAPTorsionForce,
)

from simtk.openmm.app import Topology

## dropping dependencies on openmmsystems
# from openmmsystems.util import OpenMMSystemsException
# from openmmsystems.base import BaseSystem

__all__ = ["ReplicatedSystem", "get_custom_langevin_integrator"]


#class ReplicatedSystem(BaseSystem):
class ReplicatedSystem():
    """
    (Original descriptions)
    Encapsules an openmm.System that contains multiple replicas of one system to enable batch computations.
    This class mimics the OpenMMSystem API. The implementation only works for specific forces, since
    forces of the replicated system have to be tailored so that the replicas are independent.

    Attributes
    ----------
    base_system : OpenMMSystem
        The base system that should be replicas.
    n_replicas : int
        Number of replicas to be stored in the replicated system.
    enable_energies : bool
        Whether to enable energy evaluations in batch. This option slows down the computation,
        since force objects have to be assigned to single replicas. This method enables energy
        evaluations via force groups (one force group per replica) but slows down force computations
        and propagation considerably. It also limits the maximal number of replicas to 32 (the max
        number of force groups OpenMM allows in one system). Therefore, `enable_energies=True` is not recommended.

    Notes
    -----
    Most methods in this class are static in order to enable conversion of single openmm objects (such as
    System, Topology, ...) as well as OpenMMSystem instances.

    Examples
    --------
    Replicate an openmm.System:
    >>> from openmmtools.testsystems import AlanineDipeptideImplicit
    >>> system = AlanineDipeptideImplicit().system
    >>> system_10batches = ReplicatedSystem.replicate_system(system, n_replicas=10, enable_energies=False)

    Replicate an openmmsystems.OpenMMSystem:
    >>> from openmmsystems import OpenMMToolsTestSystem
    >>> s = OpenMMToolsTestSystem("AlanineDipeptideImplicit")
    >>> s_10batches = ReplicatedSystem(s, n_replicas=10, enable_energies=False)
    >>> print(s_10batches.system, s_10batches.topology, s_10batches.positions)
    """

    ''' removing the dependencies and incomplete methods
    def __init__(self, base_system: BaseSystem, n_replicas: int, enable_energies: bool=False):
        super(ReplicatedSystem, self).__init__()
        assert n_replicas > 0
        self._base_system = base_system
        # replicate
        self._system = self.replicate_system(base_system.system, n_replicas, enable_energies)
        self._topology = self.replicate_topology(base_system.topology, n_replicas)
        self._positions = self.replicate_positions(base_system.positions)
        # set system parameters
        for parameter, default in self._parameter_defaults.items():
            self.system_parameter(parameter, getattr(base_system, parameter), default)
        self.base_system_name = self.system_parameter("base_system_name", base_system.name, "")
        self.n_replicas = self.system_parameter("n_replicas", n_replicas, None)
        self.enable_energies = self.system_parameter("enable_energies", enable_energies, None)
    
    @property
    def system(self):
        return self._system

    @staticmethod
    def replicate_positions(positions):
        """Replicate particle positions."""
        # TODO
        if type(positions) is list:
            pass
        else:
            pass
        return NotImplemented

    @staticmethod
    def replicate_topology(base_topology: Topology, n_replicas: int):
        """Replicate an OpenMM Topology."""
        topology = Topology()
        # TODO
        return NotImplemented
    '''

    @staticmethod
    def replicate_system(base_system: System, n_replicas: int, enable_energies=False):
        """Replicate an OpenMM System."""
        system = System()
        n_particles = base_system.getNumParticles()
        # particles
        for j in range(n_replicas):
            for i in range(n_particles):
                system.addParticle(base_system.getParticleMass(i))
                if system.isVirtualSite(i):
                    vs = system.getVirtualSite(i)
                    vs_copy = ReplicatedSystem._replicate_virtual_site(vs, n_particles, j)
                    system.setVirtualSite(i + j * n_particles, vs_copy)
        # constraints
        for j in range(n_replicas):
            for i in range(base_system.getNumConstraints()):
                p1, p2, distance = base_system.getConstraintParameters(i)
                system.addConstraint(p1 + j * n_particles, p2 + j * n_particles, distance)
        # properties
        system.setDefaultPeriodicBoxVectors(*(base_system.getDefaultPeriodicBoxVectors()))
        # forces
        for force in base_system.getForces():
            forcename = force.__class__.__name__
            methodname = f"_replicate_{forcename}"
            assert hasattr(ReplicatedSystem, methodname), f"Replicating {forcename} not implemented."
            replicate_force_method = getattr(ReplicatedSystem, methodname)
            replicated_forces = replicate_force_method(force, n_particles, n_replicas, enable_energies)
            for f in replicated_forces:
                system.addForce(f)
        return system

    @staticmethod
    def _replicate_virtual_site(vs, n_particles, replica):
        if isinstance(vs, LocalCoordinatesSite):
            args = []
            for i in range(vs.getNumParticles()):
                args.append(vs.getParticle(i) + replica * n_particles)
            args.append(vs.getOriginWeights())
            args.append(vs.getXWeights())
            args.append(vs.getYWeights())
            args.append(vs.getLocalPosition())
            return LocalCoordinatesSite(*args)
        elif isinstance(vs, OutOfPlaneSite):
            args = []
            for i in range(vs.getNumParticles()):
                args.append(vs.getParticle(i) + replica * n_particles)
            args.append(vs.getWeight12())
            args.append(vs.getWeight13())
            args.append(vs.getWeightCross())
            return OutOfPlaneSite(*args)
        elif isinstance(vs, TwoParticleAverageSite):
            return TwoParticleAverageSite(
                vs.getParticle(0) + replica * n_particles,
                vs.getParticle(1) + replica * n_particles,
                vs.getWeight(0),
                vs.getWeight(1)
            )
        elif isinstance(vs, ThreeParticleAverageSite):
            return ThreeParticleAverageSite(
                vs.getParticle(0) + replica * n_particles,
                vs.getParticle(1) + replica * n_particles,
                vs.getParticle(2) + replica * n_particles,
                vs.getWeight(0),
                vs.getWeight(1),
                vs.getWeight(2)
            )
        else:
            raise OpenMMSystemsException(f"Unknown virtual site type: {type(vs)}.")

    @staticmethod
    def _replicate_HarmonicBondForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = HarmonicBondForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                replicated_force.addBond(p1 + j * n_particles, p2 + j * n_particles, length, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = HarmonicBondForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_HarmonicAngleForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = HarmonicAngleForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                replicated_force.addAngle(
                    p1 + j * n_particles,
                    p2 + j * n_particles,
                    p3 + j * n_particles,
                    angle, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = HarmonicAngleForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_PeriodicTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = PeriodicTorsionForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, angle, mult, k = force.getTorsionParameters(i)
                replicated_force.addTorsion(
                    p1 + j * n_particles,
                    p2 + j * n_particles,
                    p3 + j * n_particles,
                    p4 + j * n_particles,
                    angle, mult, k)
            if enable_energies:
                # create a new force object for each replicate
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = PeriodicTorsionForce()
                replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CustomTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = CustomTorsionForce(force.getEnergyFunction())
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for i in range(force.getNumGlobalParameters()):
            replicated_force.addGlobalParameter(force.getGlobalParameterName(i),
                    force.getGlobalParameterDefaultValue(i))
        for i in range(force.getNumPerTorsionParameters()):
            replicated_force.addPerTorsionParameter(force.getPerTorsionParameterName(i))
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, params = force.getTorsionParameters(i)
                replicated_force.addTorsion(p1 + j * n_particles,
                        p2 + j * n_particles,
                        p3 + j * n_particles,
                        p4 + j * n_particles,
                        params)
            if enable_energies:
                return NotImplemented
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CMAPTorsionForce(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        replicated_force = CMAPTorsionForce()
        replicated_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
        for i in range(force.getNumMaps()):
            size, energy = force.getMapParameters(i)
            replicated_force.addMap(size, energy)
        for j in range(n_replicas):
            for i in range(force.getNumTorsions()):
                map_, *abs_ = force.getTorsionParameters(i)
                new_abs = [a_or_b + j * n_particles for a_or_b in abs_]
                replicated_force.addTorsion(map_, *new_abs)
            if enable_energies:
                return NotImplemented
        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_NonbondedForce(force, n_particles, n_replicas, enable_energies):
        nonbonded_method = force.getNonbondedMethod()
        if nonbonded_method == NonbondedForce.NoCutoff:
            return ReplicatedSystem._replicate_nonbonded_as_custom_bond_force(
                force,
                n_particles,
                n_replicas,
                enable_energies
            )
        else:
            return NotImplemented

    @staticmethod
    def _replicate_nonbonded_as_custom_bond_force(force, n_particles, n_replicas, enable_energies):
        replicated_forces = []
        energy_string = "qiqj * ONE_4PI_EPS0 / r + 4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
        ONE_4PI_EPS0 = 138.935456

        def prep_force(force=force, energy_string=energy_string):
            f = CustomBondForce(energy_string)
            f.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
            f.addPerBondParameter("qiqj")
            f.addPerBondParameter("epsilon")
            f.addPerBondParameter("sigma")
            return f

        replicated_force = prep_force()
        exceptions = {}
        for i in range(force.getNumExceptions()):
            p1, p2, qiqj, sigma, epsilon = force.getExceptionParameters(i)
            pair = (p1, p2) if p1 < p2 else (p2, p1)
            exceptions[pair] = (qiqj, sigma, epsilon)
        parameters = {}
        for i in range(force.getNumParticles()):
            q, sigma, epsilon = force.getParticleParameters(i)
            parameters[i] = (q, sigma, epsilon)
        assert force.getNumExceptionParameterOffsets() == 0
        assert force.getNumParticleParameterOffsets() == 0
        for j in range(n_replicas):
            for p1 in range(force.getNumParticles()):
                for p2 in range(p1+1, force.getNumParticles()):
                    if (p1,p2) in exceptions:
                        qiqj, sigma, epsilon = exceptions[(p1, p2)]
                        if (
                                (abs(qiqj.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                                or
                                (abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                        ):
                            replicated_force.addBond(p1 + j*n_particles, p2 + j*n_particles, [qiqj, epsilon, sigma])
                    else:
                        q1, sigma1, epsilon1 = parameters[p1]
                        q2, sigma2, epsilon2 = parameters[p2]
                        qiqj = q1*q2
                        sigma = 0.5 * (sigma1 + sigma2)
                        epsilon = np.sqrt(epsilon1 * epsilon2)
                        if (
                                (abs(qiqj.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                                or
                                (abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 1e-10)
                        ):
                            replicated_force.addBond(p1 + j*n_particles, p2 + j*n_particles, [qiqj, epsilon, sigma])
            if enable_energies:
                replicated_force.setForceGroup(j)
                replicated_forces.append(replicated_force)
                replicated_force = prep_force()

        if len(replicated_forces) == 0:
            replicated_forces.append(replicated_force)
        return replicated_forces

    @staticmethod
    def _replicate_CMMotionRemover(force, n_particles, n_replicas, enable_energies):
        return [] # we don't use the CMMotionRemover in. Part of the reason is that it's not easy to implement.

    @staticmethod
    def _replicate_CustomGBForce(force, n_particles, n_replicas, enable_energies):
        raise NotImplementedError("Not implemented in the replicated system mode. Please check out the single threaded version instead.")


def get_custom_langevin_integrator(temperatures_per_dof_in_K, friction_in_inv_ps=1.0, time_step_in_ps=0.002):
    kB = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA) \
             .value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    integrator = openmm.CustomIntegrator(time_step_in_ps)
    integrator.addGlobalVariable("a", np.exp(-friction_in_inv_ps*time_step_in_ps))
    integrator.addGlobalVariable("b", np.sqrt(1-np.exp(-2*friction_in_inv_ps*time_step_in_ps)))
    integrator.addPerDofVariable("kT", 0.)
    integrator.setPerDofVariableByName("kT", kB*temperatures_per_dof_in_K)
    integrator.addPerDofVariable("x1", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v + dt*f/m");
    integrator.addConstrainVelocities()
    integrator.addComputePerDof("x", "x + 0.5*dt*v");
    integrator.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    integrator.addComputePerDof("x", "x + 0.5*dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v + (x-x1)/dt")
    integrator.setKineticEnergyExpression("m*v1*v1/2; v1=v+0.5*dt*f/m")
    return integrator
"""

def get_custom_langevin_integrator(temperatures_per_dof_in_K, friction_in_inv_ps=1.0, time_step_in_ps=0.002):
    integrator = openmm.CustomIntegrator(time_step_in_ps)
    integrator.addPerDofVariable("temperatures", 0.)
    integrator.setPerDofVariableByName("temperatures", temperatures_per_dof_in_K)
    integrator.addGlobalVariable("friction", friction_in_inv_ps)
    integrator.addGlobalVariable("vscale", 0)
    integrator.addGlobalVariable("fscale", 0)
    integrator.addPerDofVariable("noisescales", 0)
    integrator.addPerDofVariable("x0", 0)
    integrator.addUpdateContextState()
    integrator.addComputeGlobal("vscale", "exp(-dt*friction)")
    integrator.addComputeGlobal("fscale", "(1-vscale)/friction")
    integrator.addComputePerDof("noisescales", "sqrt(kT*(1-vscale*vscale)); kT=0.008314472*temperatures")
    integrator.addComputePerDof("x0", "x")
    integrator.addComputePerDof("v", "vscale*v + fscale*f/m + noisescales*gaussian/sqrt(m)")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "(x-x0)/dt")
    integrator.setKineticEnergyExpression("m*v1*v1/2; v1=v+0.5*dt*f/m")
    return integrator
"""
