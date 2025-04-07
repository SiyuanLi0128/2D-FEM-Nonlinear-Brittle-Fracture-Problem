Developed by Siyuan Li

This solver implements a Finite Element Method (FEM) framework for iteratively solving nonlinear solid mechanics problems, with specialized capabilities for modeling brittle fracture. The core formulation leverages the second invariant of the deviatoric stress tensor (J₂) as the driving force for fracture propagation.

Key Features:

Nonlinear material behavior solver with iterative solution schemes

J₂-based fracture criterion for brittle failure modeling

Extensible architecture for implementing additional constitutive models

Educational foundation for advanced computational solid mechanics

Implementation Details:

MeshPy (2022.1.3) 

Shapely (2.0.7)

Primary Application:
Demonstrates ductile-to-brittle transition modeling through stress-driven fracture evolution, suitable for both research and educational purposes in computational fracture mechanics.
