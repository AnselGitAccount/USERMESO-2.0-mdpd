# <sub>user</sub>**MESO 2.0**

<sub>user</sub>**MESO 2.0** is an updated version of <sub>user</sub>**MESO**, which is a GPU-accelerated extension package to **LAMMPS**.

<sub>user</sub>**MESO** (https://github.com/yhtang/MESOwas) was developed by Yu-Hang Tang to simulate molecular dynamics, classic dissipative particle dynamics, and smoothed particle dynamics. It integrates several algorithmic innovations that take advantage of CUDA devices:
- An atomic-free warp-synchronous neighbor list construction algorithm;
- A 2-level particle reordering scheme, which aligns with the cell list lattice boundaries for generating strictly monotonic neighbor list;
- A locally transposed neighbor list;
- Redesigned non-branching transcendental functions ($\sin$, $\cos$, pow, $\log$, $\exp$, etc.);
- An overlapped pairwise force evaluation and halo exchange using CUDA streams for hiding the communication and the kernel launch latency;
- Radix sorting with GPU stream support;
- Pairwise random number generation based on per-timestep binary particle signatures and the prepriority Tiny Encryption Algorithm.

As an upgrade version, <sub>user</sub>**MESO 2.0** injects new capabilities into <sub>user</sub>**MESO** . Now it is possible to simulate advection, diffusion, and reaction processes with dissipative particle dynamics (tDPD). Another major upgrade is the ability to simulate red blood cells. Combining tDPD and the red blood cell model, the simulation of the chemical-releasing process from the red blood cells becomes realizable.  

The details regarding code implementation can be found at https://arxiv.org/abs/1611.06163

<br>
## Compilation Guide
> cd <working_copy>/src
>
> make yes-molecule
>
> make yes-user-meso
>
> make meso ARCH=[sm_30|sm_35|sm_52|sm_60|...]

<br>
## Running a simple example
Simulation of a red blood cell in fluid.
> cd <working_copy>/exmaple/simple
>
> ../../src/lmp_meso -in tDPD_RBC_spec1_Single_GPU.in

<br>
## Single-node benchmark
Benchmark of RBC suspension in a single node. The simulations of different system volumes are timed.
> cd <working_copy>/example/single_node_benchmark
>
> ./run_file.sh

<br>
## Example Simulation Visualization
| Chemical-release of Red Blood Cells in a Microfluidic Device |     
|:-------------------------------------------------------------|
|<img src="visualizations/chemical_release_RBC_device.png">|

<br>
## File Description
In addition to the source files in <sub>user</sub>**MESO** (program summary URL: http://cpc.cs.qub.ac.uk/summaries/AETN_v1_0.html), the following source files are included in <sub>user</sub>**MESO 2.0** package.

#### Transport dissipative particle dynamics
The files in this section are essential to the simulation of transport dissipative particle dynamics.
**atom_vec_tdpd_atomic_meso.cu/.h**
> These files contain the tdpd class declaration and implementation.

**pair_tdpd_meso.cu/.h**
> These files compute the pairwise interactions which includes forces and concentration fluxes.

**fix_nve_tdpd_meso.cu/.h**
> These files performs constant energy and volume integration to update position, velocity, and concentration for atoms in each timestep.

#### Red blood cell computation
The files in this section are solely needed to compute red blood cell dynamics.
**atom_vec_tdpd_rbc_meso.cu/.h**
> These files contain the tdpd class declaration and implementation that are specifically designed for red blood cells.

**angle_area_volume_meso.cu/.h**
> These files compute the angle term in the red blood cell model.

**bond_wlc_pow_all_visc_meso.cu/.h**
> These files compute the bond term in the red blood cell model.

**dihedral_bend_meso.cu/.h**
> These files compute the dihedral term in the red blood cell model.

#### Auxiliary
**compute_concent_tdpd_meso.ch/.h**
> These files sum the concentration of each species over all particles.

**dump_tdpd_meso.cu/.h**
> These files prints the coordinates and concentrations to an output file.

**fix_addconf_tdpd_meso.cu/.h**
> These files increases the concentration by a constant.

**fix_resetconc_tdpd_meso.cu/.h**
> These files reset the concentration to a value.
