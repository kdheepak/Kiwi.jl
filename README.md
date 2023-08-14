# KiwiConstraintSolver

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kdheepak.github.io/KiwiConstraintSolver.jl/dev/)
[![Build Status](https://github.com/kdheepak/KiwiConstraintSolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kdheepak/KiwiConstraintSolver.jl/actions/workflows/CI.yml?query=branch%3Amain)

KiwiConstraintSolver.jl is port of the efficient C++ implementation of the Cassowary constraint solving algorithm from [kiwi.cpp](https://github.com/nucleic/kiwi).

This package contains an implementation of the Cassowary constraint solving algorithm, based upon the work by G.J. Badros et al. in 2001.
This algorithm is designed primarily for use constraining elements in user interfaces, but works well for many constraints that use floats.
Constraints are linear combinations of the problem variables.
The notable features of Cassowary that make it ideal for user interfaces are that it is incremental (i.e. you can add and remove constraints at runtime and it will perform the minimum work to update the result) and that the constraints can be violated if necessary, with the order in which they are violated specified by setting a "strength" for each constraint.
This allows the solution to gracefully degrade, which is useful for when a user interface needs to compromise on its constraints in order to still be able to display something.

## References

- https://github.com/nucleic/kiwi
- https://github.com/yglukhov/kiwi
- https://github.com/schell/casuarius
