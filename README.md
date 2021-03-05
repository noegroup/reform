Replica Exchange FOR openmM (REFORM)
==============================
<!---[//]: # (Badges)--->
<!---[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/Replica Exchange FOR openmM (REFORM).svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/Replica Exchange FOR openmM (REFORM))--->
<!---[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Replica Exchange FOR openmM (REFORM)/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Replica Exchange FOR openmM (REFORM)/branch/master)--->


A simple implementation of replica exchange MD simulations for OpenMM.

### Core ideas of parallel tempering/replica exchange
Run simulations of multiple replicas of a MD system at different temperatures in parallel. Attempt to exchange configurations (atomic positions and scaled velocities) every certain time interval.

Steps for each attempt of exchanging:
- Propose a pair (or multiple pairs) of replicas for checking.
- For each proposed pair (assuming the replicas are A and B), extract from the simulation context the corresponding potential energies `U_A` and `U_B`. Then calculate the swapping probability `p=e^{(\beta_A - \beta_B)(U_A-U_B)}`, where `\beta_A=1/(k_B \times T_A)` is the inverse temperature and similarly for `\beta_B`. After that, generate a random number from U(0, 1) and test if it is smaller than p. This serves as the acceptance criterion for the swapping similar to the Metropolis-Hastings algorithm.
- Perform the swapping(s) between the replicas when the proposal(s) got accepted. In case when Langevin integrator is used, the velocities should be scaled by a factor `\sqrt{\frac{\beta_{old}}{\beta_{new}}}`

Physcially, swappings following the above process will not affect the fact that each replica samples the Boltzmann distribution of the conformational space of the system at its temperature. However, the exchange can help the simulation at lower temperature to faster get out of local energy minima, thus exploring the conformational space more efficiently. One need to note that the kinetics are not preserved when replica exchange method is used.

#### References and continue reading:
- Some general introduction: Earl et al., [Phys. Chem. Chem. Phys., 2005,7, 3910-3916](https://doi.org/10.1039/B509983H).
- Scaling of the velocities during swapping: Mori et al., [J. Phys. Soc. Jpn. 79, 074001 (2010)](https://doi.org/10.1143/JPSJ.79.074001).
- Influence of parameters in REMD: Iwai et al., [Biophys Physicobiol. 2018; 15: 165–172](https://dx.doi.org/10.2142%2Fbiophysico.15.0_165).

### Design goals and road maps
- [x] An abstract class for holding and enabling access to multiple replicas.
- [x] Core replica exchange functionalities.
- [x] Implementation of the multiple replicas with OpenMM.
- [x] Interfaces to the users.
- [x] Some test systems.
- [x] Implementation of multiple replicas with the `replicated_systems` (speedups for small systems and for neural network force fields).
- [ ] Implementation of multiple replicas with concurrency in Python.

### Copyright

Copyright (c) 2020-2021, noegroup


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
