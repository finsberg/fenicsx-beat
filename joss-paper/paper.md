---
title: 'fenicsx-beat - An Open Source Simulation Framework for Cardiac Electrophysiology'
tags:
  - cardiac electrophysiology
  - monodomain
  - ecg
  - python
  - FEniCSx
  - FEniCS
  - partial differential equations
  - finite element method
authors:
  - name: Henrik Finsberg
    orcid: 0000-0001-6489-8858
    corresponding: true
    affiliation: 1

affiliations:
 - name: Simula Research Laboratory, Oslo, Norway
   index: 1

date: 20 March 2024
bibliography: paper.bib

---

# Summary
`fenicsx-beat` is an open-source Python software package for simulating cardiac electrophysiology, built upon the FEniCSx finite element framework [@ORBi-bf72337a-a760-487c-84eb-292ea4cebe75]. It provides tools to solve the Monodomain model, a widely used model for electrical signal propagation in cardiac tissue, coupled with systems of ordinary differential equations (ODEs) that represent complex ionic models governing cell membrane dynamics. Designed for researchers in computational cardiology, `fenicsx-beat` leverages the high-performance capabilities of FEniCSx to enable efficient and scalable simulations of cardiac electrical activity on various computational platforms.


# Statement of need
Computational modeling plays an increasingly vital role in cardiac electrophysiology research, offering insights into mechanisms of heart rhythm disorders (arrhythmias), evaluating therapeutic strategies like drug effects or ablation, and paving the way towards personalized medicine and "digital twin" concepts. Mathematical models, ranging from detailed ionic interactions at the subcellular level to electrical wave propagation at the tissue and organ level, allow for quantitative investigation of physiological and pathological conditions that are difficult or impossible to study experimentally.

The Bidomain model is often considered the most physiologically detailed representation of cardiac tissue electrophysiology, but its computational cost can be prohibitive for large-scale or long-duration simulations. The Monodomain model, a simplification valid under certain assumptions about tissue conductivity, offers a computationally less expensive alternative while still capturing essential propagation dynamics accurately in many scenarios, such as studies of activation sequences or basic arrhythmia mechanisms. Solving these models typically involves coupling the reaction-diffusion PDE (Monodomain or Bidomain) with a system of stiff ODEs describing the ionic currents across the cell membrane (the ionic model) [@sundnes2007computing].

The FEniCSx project provides a modern, powerful, and performant platform for solving PDEs using the finite element method, featuring a high-level Python interface for ease of use and a C++ backend for efficiency and scalability. There is a need for specialized cardiac simulation tools that integrate seamlessly with this evolving ecosystem. While related tools based on the legacy FEniCS library exist, such as `cbcbeat` [@Rognes2017] and the electro-mechanics solver `simcardems` [@Finsberg2023], `fenicsx-beat` specifically targets the FEniCSx platform, providing researchers already using FEniCSx with a dedicated and readily integrable tool for cardiac electrophysiology simulations based on the Monodomain model.


## Functionality
`fenicsx-beat` facilitates the simulation of electrical wave propagation in cardiac tissue by solving the Monodomain equation coupled with a system of ODEs representing a chosen ionic model. The core mathematical problem is a reaction-diffusion system:

$$
\chi (C_m \frac{\partial v}{\partial t} + I_{\text{ion}}  - I_{\text{stim}}) = \nabla \cdot (\sigma\nabla v)
$$

$$
\frac{\partial s}{\partial t} = f(s, v, t)
$$


where $v$ is the transmembrane potential, $s$ represents the state variables of the ionic model, $C_m$ is the membrane capacitance, $I_{\text{ion}}$ is the total ionic current, $I_{\text{stim}}$ is the applied stimulus current, $\sigma$ is the conductivity tensor, and $f(s, v, t)$ is the system of ODEs defining the ionic model. The Monodomain equation describes the diffusion of the transmembrane potential $v$ in the tissue, while the ODE system captures the dynamics of the ionic currents across the cell membrane. The coupling between these two components is essential for accurately simulating cardiac electrophysiology.

The software leverages the FEniCSx library for the spatial discretization of the PDE component using the finite element method. Variational forms are expressed using the Unified Form Language (UFL), and the software utilizes the high-performance assembly and solution capabilities of DOLFINx. This allows for simulations on complex geometries using various element types and supports parallel execution via MPI. DOLFINx interfaces with robust external libraries, notably PETSc[@petsc-user-ref], for scalable linear algebra operations and solvers.

## Comparison with Other Software
The field of computational cardiac electrophysiology benefits from several open-source simulation packages. `fenicsx-beat` distinguishes itself by being built natively on the modern FEniCSx framework, targeting users who leverage this platform for its flexibility in solving PDEs with the finite element method.

Within the FEniCS ecosystem, `fenicsx-beat` can be seen as a successor or counterpart to `cbcbeat`, which provided similar Monodomain/Bidomain capabilities but was based on the legacy FEniCS library. Other FEniCS/FEniCSx-based tools focus on different physics: `simcardems` couples electrophysiology with solid mechanics, `pulse` [@Finsberg2019] focusing solely on cardiac mechanics, and `Ambit` [@Hirschvogel2024] is a newer multi-physics solver primarily for cardiac mechanics and fluid-structure interaction (FSI), although future electrophysiology capabilities are envisioned. `fenicsx-beat` provides a dedicated, up-to-date electrophysiology solver within this FEniCSx environment.

Compared to established standalone simulators like openCARP [@plank2021opencarp] or Chaste[@Cooper2020], `fenicsx-beat` offers tighter integration with FEniCSx libraries. openCARP is a powerful, widely used simulator with its own optimized C++ core and a dedicated Python workflow tool (carputils) for managing complex simulations. Chaste (Cancer, Heart And Soft Tissue Environment) is a C++ library designed for a broad range of computational biology problems, including cardiac electrophysiology.

In summary, `fenicsx-beat` occupies a valuable niche by providing a modern, Python-interfaced, FEniCSx-native tool for Monodomain cardiac electrophysiology simulations. Its primary strength lies in its seamless integration with the FEniCSx platform, making it an attractive choice for researchers utilizing FEniCSx for multi-physics cardiac modeling.

# Acknowledgements
This work has been financially supported by Simula Research Laboratory and by the European Union’s Horizon 2020 research and innovation program (grant number: 101016496 (SimCardioTest)).

# References
