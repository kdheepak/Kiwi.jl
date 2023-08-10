```@meta
CurrentModule = KiwiConstraintSolver
```

# [KiwiConstraintSolver](https://github.com/kdheepak/KiwiConstraintSolver.jl).

```@example
using KiwiConstraintSolver

s = Solver()
x = Variable()
y = Variable()

add_constraint(s, x == 20)
add_constraint(s, x + 2 == y + 10)
update_variables(s)

x.value, y.value
```

This module contains an implementation of the Cassowary constraint solving algorithm, based upon the work by G.J. Badros et al. in 2001.
This algorithm is designed primarily for use constraining elements in user interfaces, but works well for many constraints that use floats.
Constraints are linear combinations of the problem variables.
The notable features of Cassowary that make it ideal for user interfaces are that it is incremental (i.e. you can add and remove constraints at runtime and it will perform the minimum work to update the result) and that the constraints can be violated if necessary, with the order in which they are violated specified by setting a "strength" for each constraint.
This allows the solution to gracefully degrade, which is useful for when a user interface needs to compromise on its constraints in order to still be able to display something.

```@index

```

```@autodocs
Modules = [KiwiConstraintSolver]
```
