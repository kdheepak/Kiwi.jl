"""
This module contains an implementation of the Cassowary constraint solving algorithm, based upon the work by G.J. Badros et al. in 2001. 
This algorithm is designed primarily for use constraining elements in user interfaces, but works well for many constraints that use floats.
Constraints are linear combinations of the problem variables. 
The notable features of Cassowary that make it ideal for user interfaces are that it is incremental (i.e. you can add and remove constraints at runtime and it will perform the minimum work to update the result) and that the constraints can be violated if necessary, with the order in which they are violated specified by setting a "strength" for each constraint.
This allows the solution to gracefully degrade, which is useful for when a user interface needs to compromise on its constraints in order to still be able to display something.
"""
module KiwiConstraintSolver

using DataStructures
using InlineTest

import Base.isequal, Base.hash
import Base: +, -, *, /, ==, <=, >=

function create_strength(a::Real, b::Real, c::Real, w::Real=1.0)
  result = 0.0
  result += max(0.0, min(1000.0, a * w)) * 1000000.0
  result += max(0.0, min(1000.0, b * w)) * 1000.0
  result += max(0.0, min(1000.0, c * w))
  return result
end

const ϵ = 1e-8

is_approx_zero(value::Float64) = ≈(value, 0.0; atol=ϵ)

const REQUIRED = create_strength(1000.0, 1000.0, 1000.0)
const STRONG = create_strength(1.0, 0.0, 0.0)
const MEDIUM = create_strength(0.0, 1.0, 0.0)
const WEAK = create_strength(0.0, 0.0, 1.0)

function clamp_strength(value::Real)
  return clamp(value, 0, REQUIRED)
end

"""
SymbolKind:

- :INVALID
- :EXTERNAL
- :SLACK
- :ERROR
- :DUMMY
"""
const KiwiSymbolKind = Symbol

mutable struct KiwiSymbol
  value::UInt32
  kind::KiwiSymbolKind
end

KiwiSymbol() = KiwiSymbol(0, :INVALID)

isequal(s1::KiwiSymbol, s2::KiwiSymbol) = s1.value == s2.value && s1.kind == s2.kind
hash(s::KiwiSymbol) = hash(string(hash(s.value)) * string(hash(s.kind)))

kind(s::KiwiSymbol) = s.kind
invalid(s::KiwiSymbol) = s.value == 0
valid(s::KiwiSymbol) = s.value != 0

mutable struct Variable
  name::String
  value::Float64
  function Variable(name::String, value::Real=0.0)
    return new(name, value)
  end
end

isequal(v1::Variable, v2::Variable) = v1.name == v2.name
hash(v::Variable) = hash(v.name)

function Variable(value::Real=0.0)
  return Variable(string(gensym(:anon)), value)
end

function name(v::Variable)
  return v.name
end

function Base.show(io::IO, v::Variable)
  print(io, "name: ", name(v), ", value: ", v.value)
end

mutable struct Term
  variable::Variable
  coefficient::Float64
  function Term(variable::Variable, coefficient::Real=1.0)
    return new(variable, coefficient)
  end
end

isequal(t1::Term, t2::Term) = isequal(t1.variable, t2.variable)
hash(t::Term) = hash(t.variable)

function value(t::Term)
  return t.coefficient * t.variable.value
end

function Base.show(io::IO, t::Term)
  print(io, "variable: (", t.variable, "), coefficient: ", t.coefficient)
end

mutable struct Expression
  terms::Vector{Term}
  constant::Float64
  function Expression(terms::Vector{Term}, constant::Real=0.0)
    return new(terms, constant)
  end
end

isequal(e1::Expression, e2::Expression) = isequal(e1.terms, e2.terms)
hash(e::Expression) = hash(e.terms)

function Expression(constant::Real=0.0)
  return Expression(Term[], constant)
end

function Expression(term::Term, constant::Real=0.0)
  return Expression([term], constant)
end

function Expression(v::Variable)
  return Expression(Term(v))
end

function value(e::Expression)
  return e.constant + sum(value.(e.terms))
end

function is_constant(e::Expression)
  return isempty(e.terms)
end

"""Mutates this expression by multiplying it by minus one."""
function negate(e::Expression)
  e.constant = -e.constant
  for (i, t) in enumerate(e.terms)
    e.terms[i] = -1 * deepcopy(t)
  end
end

function Base.show(io::IO, e::Expression)
  print(io, "is_constant: ", is_constant(e), ", constant: ", e.constant)
  if !is_constant(e)
    print(io, " terms: [")
    print(io, join(e.terms, ", "))
    print(io, "]")
  end
end

"""
RelationalOperator:

- `<=`
- `>=`
- `==`
"""
const RelationalOperator = Symbol

function reduce(e::Expression)
  vars = DefaultDict{Variable,Float64}(0.0)
  for term in e.terms
    v = term.variable
    vars[v] += term.coefficient
  end
  reducedTerms = Term[]
  for (variable, coef) in vars
    push!(reducedTerms, Term(variable, coef))
  end
  return Expression(reducedTerms, e.constant)
end

mutable struct Constraint
  expression::Expression
  strength::Float64
  weight::Float64
  op::RelationalOperator
  function Constraint(e::Expression, strength::Real, op::RelationalOperator)
    return new(reduce(e), clamp_strength(strength), 1, op)
  end
end

isequal(c1::Constraint, c2::Constraint) = isequal(c1.expression, c2.expression) && c1.op == c2.op
hash(c::Constraint) = hash(string(hash(c.expression)) * string(hash(c.op)))

function Constraint(e::Expression, op::RelationalOperator)
  return Constraint(e, REQUIRED, op)
end

function Constraint(other::Constraint, strength::Real)
  return Constraint(other.expression, other.op, strength)
end

function Base.show(io::IO, c::Constraint)
  print(io, "expression: (", c.expression, "), strength: ", c.strength, ", operator: ", c.op)
end

mutable struct Row
  constant::Float64
  cells::Dict{KiwiSymbol,Float64}
end

function Row(constant::Real=0.0)
  return Row(constant, Dict{KiwiSymbol,Float64}())
end

function Row(other::Row)
  return Row(other.constant, deepcopy(other.cells))
end

invalid(r::Row) = isempty(r.cells)

function add!(r::Row, value::Real)
  r.constant += value
  return r.constant
end

"""
Insert a symbol into the row with a given coefficient.

If the symbol already exists in the row, the coefficient will be
added to the existing coefficient. If the resulting coefficient
is zero, the symbol will be removed from the row.
"""
function insert!(r::Row, symbol::KiwiSymbol, coefficient::Real=1.0)
  coefficient += get(r.cells, symbol, 0)
  if is_approx_zero(coefficient)
    delete!(r.cells, symbol)
  else
    r.cells[symbol] = coefficient
  end
end

"""
Insert a row into this row with a given coefficient.

The constant and the cells of the other row will be multiplied by
the coefficient and added to this row. Any cell with a resulting
coefficient of zero will be removed from the row.
"""
function insert!(r::Row, other::Row, coefficient::Real=1.0)
  r.constant += other.constant * coefficient
  for (s, v) in other.cells
    insert!(r, s, v * coefficient)
  end
end

"""
Remove the given symbol from the row.
"""
function remove!(r::Row, symbol::KiwiSymbol)
  delete!(r.cells, symbol)
end

"""
Reverse the sign of the constant and all cells in the row.
"""
function reverse_sign!(r::Row)
  r.constant = -r.constant
  foreach(keys(r.cells)) do s
    r.cells[s] = -r.cells[s]
  end
end

"""
Solve the row for the given symbol.

This method assumes the row is of the form a * x + b * y + c = 0
and (assuming solve for x) will modify the row to represent the
right hand side of x = -b/a * y - c / a. The target symbol will
be removed from the row, and the constant and other cells will
be multiplied by the negative inverse of the target coefficient.

The given symbol *must* exist in the row.
"""
function solve!(r::Row, symbol::KiwiSymbol)
  coefficient = -1.0 / r.cells[symbol]
  delete!(r.cells, symbol)
  r.constant *= coefficient
  foreach(keys(r.cells)) do s
    r.cells[s] *= coefficient
  end
end

"""
Solve the row for the given symbols.

This method assumes the row is of the form x = b * y + c and will
solve the row such that y = x / b - c / b. The rhs symbol will be
removed from the row, the lhs added, and the result divided by the
negative inverse of the rhs coefficient.

The lhs symbol *must not* exist in the row, and the rhs symbol
*must* exist in the row.
"""
function solve!(r::Row, lhs::KiwiSymbol, rhs::KiwiSymbol)
  insert!(r, lhs, -1.0)
  solve!(r, rhs)
end

"""
Get the coefficient for the given symbol.

If the symbol does not exist in the row, zero will be returned.
"""
function coefficient_for(r::Row, symbol::KiwiSymbol)
  return get(r.cells, symbol, 0)
end

"""Test whether a row is composed of all dummy variables."""
all_dummies(row::Row) = all(kind(k) == :DUMMY for k in keys(row.cells))

"""
Substitute a symbol with the data from another row.

Given a row of the form a * x + b and a substitution of the
form x = 3 * y + c the row will be updated to reflect the
expression 3 * a * y + a * c + b.

If the symbol does not exist in the row, this is a no-op.
"""
function substitute!(r::Row, symbol::KiwiSymbol, row::Row)
  coefficient = pop!(r.cells, symbol, nothing)
  if !isnothing(coefficient)
    insert!(r, row, coefficient)
    true
  else
    false
  end
end

"""
Get the first SLACK or ERROR symbol in the row.

If no such symbol is present, and Invalid symbol will be returned.
"""
function any_pivotable_symbol(row::Row)
  for (k, v) in row.cells
    if k in [:SLACK, :ERROR]
      return k
    end
  end
  return KiwiSymbol() # Return invalid symbol
end

"""
Compute the entering variable for a pivot operation.

This method will return first symbol in the objective function which
is non-dummy and has a coefficient less than zero. If no symbol meets
the criteria, it means the objective function is at a minimum, and an
invalid symbol is returned.
"""
function get_entering_symbol(objective::Row)
  for (k, v) in objective.cells
    if kind(k) != :DUMMY && v < 0
      return k
    end
  end
  return KiwiSymbol()
end

struct EditConstraintException <: Exception end
struct DuplicateConstraintException <: Exception end
struct UnsatisfiableConstraintException <: Exception end
struct UnknownConstraintException <: Exception end
struct DuplicateEditVariableException <: Exception end
struct BadRequiredStrengthException <: Exception end
struct UnknownEditVariableException <: Exception end
struct InternalSolverException <: Exception end

mutable struct Tag
  marker::KiwiSymbol
  other::KiwiSymbol
end

Tag() = Tag(KiwiSymbol(), KiwiSymbol())

mutable struct EditInfo
  tag::Tag
  constraint::Constraint
  constant::Float64
end

function EditInfo(constraint::Constraint, tag::Tag, constant::Real)
  return EditInfo(tag, constraint, constant)
end

mutable struct Solver
  cns::OrderedDict{Constraint,Tag}
  rows::OrderedDict{KiwiSymbol,Row}
  vars::OrderedDict{Variable,KiwiSymbol}
  edits::OrderedDict{Variable,EditInfo}
  infeasible_rows::Vector{KiwiSymbol} # should never contain EXTERNAL
  objective::Row
  artificial::Row
  id_tick::UInt32
end

function Solver()
  return Solver(OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), [], Row(), Row(), UInt32(0))
end

function KiwiSymbol(s::Solver, kind::KiwiSymbolKind)
  s.id_tick += 1
  KiwiSymbol(s.id_tick, kind)
end

"""
Add a constraint to the solver.

Errors:

- DuplicateConstraint: The given constraint has already been added to the solver.
- UnsatisfiableConstraint: The given constraint is required and cannot be satisfied.
"""
function add_constraint(s::Solver, constraint::Constraint)
  if haskey(s.cns, constraint)
    throw(DuplicateConstraintException())
  end

  # Creating a row causes symbols to be reserved for the variables in the constraint.
  # If this method exits with an exception, then solver must be reinitialized.
  tag = Tag()
  row = create_row(s, constraint, tag)
  if invalid(tag.marker)
    error("Internal error. Got $tag for constraint $constraint")
  end
  subject = choose_subject(row, tag)

  #  If chooseSubject could not find a valid entering symbol, one
  #  last option is available if the entire row is composed of
  #  dummy variables. If the constant of the row is zero, then
  #  this represents redundant constraints and the new dummy
  #  marker can enter the basis. If the constant is non-zero,
  #  then it represents an unsatisfiable constraint.
  if invalid(subject) && all_dummies(row)
    if !(is_approx_zero(row.constant))
      throw(UnsatisfiableConstraintException())
    else
      subject = tag.marker
    end
  end

  # If an entering symbol still isn't found, then the row must
  # be added using an artificial variable. If that fails, then
  # the row represents an unsatisfiable constraint.
  if invalid(subject)
    if !add_with_artificial_variable(s, row)
      throw(UnsatisfiableConstraintException())
    end
  else
    solve!(row, subject)
    substitute!(s, subject, row)
    s.rows[subject] = row
  end

  s.cns[constraint] = tag

  # Optimizing after each constraint is added performs less
  # aggregate work due to a smaller average system size. It
  # also ensures the solver remains in a consistent state.
  optimize!(s, s.objective)
end

"""
Remove a constraint from the solver.

Errors:

- UnknownConstraint: The given constraint has not been added to the solver.
"""
function remove_constraint(s::Solver, constraint::Constraint)
  tag = get(s.cns, constraint, nothing)
  if isnothing(tag)
    throw(UnknownConstraintException(""))
  else
    delete!(s.cns, constraint)
  end

  # Remove the error effects from the objective function
  # *before* pivoting, or substitutions into the objective
  # will lead to incorrect solver results.
  remove_constraint_effects(s, constraint, tag)

  # If the marker is basic, simply drop the row. Otherwise,
  # pivot the marker into the basis and then drop the row.
  row = get(s.rows, tag.marker, nothing)
  if !isnothing(row)
    delete!(s.rows, tag.marker)
  else
    row = get_marker_leaving_row(s, tag.marker)
    @assert !isnothing(row), "internal solver error"

    leaving = KiwiSymbol()
    for (sym, v) in s.rows
      if v == row
        leaving = sym
      end
    end

    @assert !invalid(leaving) "internal solver error"

    delete!(s.rows, leaving)
    solve!(row, leaving, tag.marker)
    substitute!(s, tag.marker, row)
  end

  # Optimizing after each constraint is removed ensures that the
  # solver remains consistent. It makes the solver api easier to
  # use at a small tradeoff for speed.
  optimize!(s, s.objective)
end

"""
Remove the effects of an error marker on the objective function.
"""
function remove_marker_effects(s::Solver, marker::KiwiSymbol, strength::Real)
  row = get(s.rows, marker, nothing)
  if isnothing(row)
    insert!(s.objective, marker, -strength)
  else
    insert!(s.objective, row, -strength)
  end
end

"""
Remove the effects of a constraint on the objective function.
"""
function remove_constraint_effects(s::Solver, constraint::Constraint, tag::Tag)
  if kind(tag.marker) == :ERROR
    remove_marker_effects(s, tag.marker, constraint.strength)
  elseif kind(tag.other) == :ERROR
    remove_marker_effects(s, tag.other, constraint.strength)
  end
end


"""
Compute the leaving row for a marker variable.

This method will return an iterator to the row in the row map
which holds the given marker variable. The row will be chosen
according to the following precedence:

1) The row with a restricted basic varible and a negative coefficient
    for the marker with the smallest ratio of -constant / coefficient.

2) The row with a restricted basic variable and the smallest ratio
    of constant / coefficient.

3) The last unrestricted row which contains the marker.

If the marker does not exist in any row, the row map end() iterator
will be returned. This indicates an internal solver error since
the marker *should* exist somewhere in the tableau.
"""
function get_marker_leaving_row(solver::Solver, marker::KiwiSymbol)
  r1 = Inf
  r2 = Inf

  first = nothing
  second = nothing
  third = nothing
  for (s, candidateRow) in solver.rows
    c = coefficient_for(candidateRow, marker)
    if c == 0.0
      continue
    end

    if kind(s) == :EXTERNAL
      third = candidateRow
    elseif c < 0
      r = -candidateRow.constant / c
      if r < r1
        r1 = r
        first = candidateRow
      end
    else
      r = candidateRow.constant / c
      if r < r2
        r2 = r
        second = candidateRow
      end
    end
  end

  if !isnothing(first)
    return first
  elseif !isnothing(second)
    return second
  end
  return third
end

"""Test whether a constraint has been added to the solver."""
has_constraint(s::Solver, constraint::Constraint) = constraint in keys(s.cns)

"""
Add an edit variable to the solver.

This method should be called before the `suggest_value` method is
used to supply a suggested value for the given edit variable.

Errors:

- DuplicateEditVariableException: The given edit variable has already been added to the solver.
- BadRequiredStrengthException: The given strength is >= required.
"""
function add_edit_variable(s::Solver, variable::Variable, strength::Real)
  if variable in keys(s.edits)
    throw(DuplicateEditVariableException(""))
  end

  strength = clamp_strength(strength)

  if strength == REQUIRED
    throw(BadRequiredStrengthException(""))
  end

  constraint = Constraint(Expression(Term(variable)), strength, :(==))

  add_constraint(s, constraint)

  info = EditInfo(constraint, get(s.cns, constraint, nothing), 0.0)
  s.edits[variable] = info
end

"""
Remove an edit variable from the solver.

Error:

- UnknownEditVariable: The given edit variable has not been added to the solver.
"""
function remove_edit_variable(s::Solver, variable::Variable)
  edit = get(s.edits, variable, nothing)
  if isnothing(edit)
    throw(UnknownEditVariableException(""))
  end

  try
    remove_constraint(s, edit.constraint)
  catch e
    if e isa UnknownConstraintException
      @warn "Exception caught" exception = (e, catch_backtrace())
    end
  end

  delete!(s.edits, variable)
end

function remove_all_edit_variables(s::Solver)
  for (k, edit) in s.edits
    try
      remove_constraint(s, edit.constraint)
    catch e
      if e isa UnknownConstraintException
        @warn "Exception caught" exception = (e, catch_backtrace())
      end
    end
  end

  empty!(s.edits)
end

"""Test whether an edit variable has been added to the solver."""
has_edit_variable(s::Solver, variable::Variable) = variable in keys(s.edits)

"""
Compute the entering symbol for the dual optimize operation.

This method will return the symbol in the row which has a positive
coefficient and yields the minimum ratio for its respective symbol
in the objective function. The provided row *must* be infeasible.
If no symbol is found which meats the criteria, an invalid symbol
is returned.
"""
function get_dual_entering_symbol(solver::Solver, row::Row)
  ratio = Inf
  result = KiwiSymbol()
  for (s, currentCell) in row.cells
    if kind(s) != :DUMMY
      if currentCell > 0
        coefficient = coefficient_for(solver.objective, s)
        r = coefficient / currentCell
        if r < ratio
          ratio = r
          result = s
        end
      end
    end
  end
  return result
end

"""
Optimize the system using the dual of the simplex method.

The current state of the system should be such that the objective
function is optimal, but not feasible. This method will perform
an iteration of the dual simplex method to make the solution both
optimal and feasible.

Errors:

- InternalSolverException: The system cannot be dual optimized.
"""
function dual_optimize!(s::Solver)
  while length(s.infeasible_rows) > 0
    leaving = pop!(s.infeasible_rows)
    row = get(s.rows, leaving, nothing)
    if !isnothing(row) && row.constant < 0
      entering = get_dual_entering_symbol(s, row)
      @assert !(invalid(entering)) "internal solver error"

      delete!(s.rows, leaving)
      solve!(row, leaving, entering)
      substitute!(s, entering, row)
      s.rows[entering] = row
    end
  end
end

"""
Suggest a value for the given edit variable.

This method should be used after an edit variable as been added to
the solver in order to suggest the value for that variable.

Errors:

`UnknownEditVariable`: The given edit variable has not been added to the solver.
"""
function suggest_value(s::Solver, variable::Variable, value::Real)
  info = get(s.edits, variable, nothing)
  if isnothing(info)
    throw(UnknownEditVariableException(""))
  end

  delta = value - info.constant
  info.constant = value

  # Check first if the positive error variable is basic.
  row = get(s.rows, info.tag.marker, nothing)
  if !isnothing(row)
    if add!(row, -delta) < 0
      push!(s.infeasible_rows, info.tag.marker)
    end
    dual_optimize!(s)
    return
  end

  # Check next if the negative error variable is basic.
  row = get(s.rows, info.tag.other, nothing)
  if !isnothing(row)
    if add!(row, delta) < 0
      push!(s.infeasible_rows, info.tag.other)
    end
    dual_optimize!(s)
    return
  end

  # Otherwise update each row where the error variables exist.
  for (sym, currentRow) in s.rows
    coefficient = coefficient_for(currentRow, info.tag.marker)
    if coefficient != 0 && add!(currentRow, delta * coefficient) < 0 && kind(sym) != :EXTERNAL
      push!(s.infeasible_rows, sym)
    end
  end

  dual_optimize!(s)
end

"""Update the values of the external solver variables."""
function update_variables(s::Solver)
  for (variable, sym) in s.vars
    row = get(s.rows, sym, nothing)
    if isnothing(row)
      variable.value = 0
    else
      variable.value = row.constant
    end
  end
end

function get_var_symbol(s::Solver, variable::Variable)
  result = get(s.vars, variable, KiwiSymbol())
  if invalid(result)
    result = KiwiSymbol(s, :EXTERNAL)
    s.vars[variable] = result
  end
  return result
end

"""
Create a new Row object for the given constraint.

The terms in the constraint will be converted to cells in the row.
Any term in the constraint with a coefficient of zero is ignored.
This method uses the `getVarSymbol` method to get the symbol for
the variables added to the row. If the symbol for a given cell
variable is basic, the cell variable will be substituted with the
basic row.

The necessary slack and error variables will be added to the row.
If the constant for the row is negative, the sign for the row
will be inverted so the constant becomes positive.

The tag will be updated with the marker and error symbols to use
for tracking the movement of the constraint in the tableau.
"""
function create_row(s::Solver, constraint::Constraint, tag::Tag)
  expression = constraint.expression
  row = Row(expression.constant)

  # Substitute the current basic variables into the row.
  for term in expression.terms
    if !(is_approx_zero(term.coefficient))
      symbol = get_var_symbol(s, term.variable)
      other_row = get(s.rows, symbol, nothing)
      if isnothing(other_row)
        insert!(row, symbol, term.coefficient)
      else
        insert!(row, other_row, term.coefficient)
      end
    end
  end

  # Add the necessary slack, error, and dummy variables.
  if constraint.op in [:(<=), :(>=)]
    coefficient = (constraint.op == :(<=) ? 1.0 : -1.0)
    slack = KiwiSymbol(s, :SLACK)
    tag.marker = slack
    insert!(row, slack, coefficient)

    if constraint.strength < REQUIRED
      error = KiwiSymbol(s, :ERROR)
      tag.other = error
      insert!(row, error, -coefficient)
      insert!(s.objective, error, constraint.strength)
    end

  elseif constraint.op == :(==)
    if constraint.strength < REQUIRED
      errplus = KiwiSymbol(s, :ERROR)
      errminus = KiwiSymbol(s, :ERROR)
      tag.marker = errplus
      tag.other = errminus
      insert!(row, errplus, -1.0)
      insert!(row, errminus, 1.0)
      insert!(s.objective, errplus, constraint.strength)
      insert!(s.objective, errminus, constraint.strength)
    else
      dummy = KiwiSymbol(s, :DUMMY)
      tag.marker = dummy
      insert!(row, dummy)
    end
  end

  # Ensure the row as a positive constant.
  if row.constant < 0
    reverse_sign!(row)
  end

  return row
end

"""
Choose the subject for solving for the row.

This method will choose the best subject for using as the solve
target for the row. An invalid symbol will be returned if there
is no valid target.

The symbols are chosen according to the following precedence:

1) The first symbol representing an external variable.
2) A negative slack or error tag variable.

If a subject cannot be found, an invalid symbol will be returned.
"""
function choose_subject(row::Row, tag::Tag)
  for s in keys(row.cells)
    if kind(s) == :EXTERNAL
      return s
    end
  end

  if tag.marker in [:SLACK, :ERROR] && coefficient_for(row, tag.marker) < 0
    return tag.marker
  end

  if tag.other in [:SLACK, :ERROR] && coefficient_for(row, tag.other) < 0
    return tag.other
  end

  return KiwiSymbol() # Return invalid symbol
end

"""
Add the row to the tableau using an artificial variable.

This will return false if the constraint cannot be satisfied.
"""
function add_with_artificial_variable(s::Solver, row::Row)
  art = KiwiSymbol(s, :SLACK)
  s.rows[art] = deepcopy(row)

  s.artificial = deepcopy(row)

  optimize!(s, s.artificial)
  success = is_approx_zero(s.artificial.constant)
  s.artificial = Row()

  rowptr = get(s.rows, art, nothing)

  if !isnothing(rowptr)
    for sym in keys(s.rows)
      if s.rows[sym] === rowptr
        delete!(s.rows, sym)
      end
    end

    if isempty(rowptr.cells)
      return success
    end

    entering = any_pivotable_symbol(rowptr)
    if invalid(entering)
      return false
    end

    solve!(rowptr, art, entering)
    substitute!(s, entering, rowptr)
    s.rows[entering] = rowptr
  end

  for v in values(s.rows)
    remove!(v, art)
  end

  remove!(s.objective, art)

  return success
end

"""
Substitute the parametric symbol with the given row.

This method will substitute all instances of the parametric symbol
in the tableau and the objective function with the given row.
"""
function substitute!(s::Solver, symbol::KiwiSymbol, row::Row)
  for sym in keys(s.rows)
    r = s.rows[sym]
    substitute!(r, symbol, row)
    if kind(sym) != :EXTERNAL && r.constant < 0
      push!(s.infeasible_rows, sym)
    end
  end

  substitute!(s.objective, symbol, row)

  if !isnothing(s.artificial)
    substitute!(s.artificial, symbol, row)
  end
end

"""
Compute the row which holds the exit symbol for a pivot.

This method will return an iterator to the row in the row map
which holds the exit symbol. If no appropriate exit symbol is
found, the end() iterator will be returned. This indicates that
the objective function is unbounded.
"""
function get_leaving_row(s::Solver, entering::KiwiSymbol)
  ratio = Inf
  result = Row() # Invalid row
  for (key, candidateRow) in s.rows
    if kind(key) != :EXTERNAL
      temp = coefficient_for(candidateRow, entering)
      if temp < 0
        temp_ratio = (-candidateRow.constant / temp)
        if temp_ratio < ratio
          ratio = temp_ratio
          result = candidateRow
        end
      end
    end
  end
  return result
end

"""
Optimize the system for the given objective function.

This method performs iterations of Phase 2 of the simplex method
until the objective function reaches a minimum.

Errors:

- InternalSolverException: The value of the objective function is unbounded.
"""
function optimize!(s::Solver, objective::Row)
  while true
    entering = get_entering_symbol(objective)
    if invalid(entering)
      return
    end

    entry = get_leaving_row(s, entering)
    @assert !invalid(entry) "The objective is unbounded."

    entryKey = [key for (key, value) in s.rows if value === entry][1]

    delete!(s.rows, entryKey)
    solve!(entry, entryKey, entering)
    substitute!(s, entering, entry)
    s.rows[entering] = entry
  end
end

# Variable multiply, divide, and unary invert
*(variable::Variable, coefficient::Real) = Term(variable, coefficient)
/(variable::Variable, denominator::Real) = variable * (1 / denominator)
-(variable::Variable) = variable * (-1.0)

# Term multiply, divide, and unary invert
*(term::Term, coefficient::Real) = Term(term.variable, term.coefficient * coefficient)
/(term::Term, denominator::Real) = term * (1 / denominator)
-(term::Term) = term * (-1.0)

# Expression multiply, divide, and unary invert
function *(expression::Expression, coefficient::Real)
  terms = [term * coefficient for term in expression.terms]
  Expression(terms, expression.constant * coefficient)
end

/(expression::Expression, denominator::Real) = expression * (1 / denominator)
-(expression::Expression) = expression * (-1.0)

# Expression and Expression multiplication
function *(expression1::Expression, expression2::Expression)
  if is_constant(expression1)
    return expression2 * expression1.constant
  elseif is_constant(expression2)
    return expression1 * expression2.constant
  else
    error("NonlinearExpressionException")
  end
end

# Expression division
function /(expression1::Expression, expression2::Expression)
  if is_constant(expression2)
    return expression1 / expression2.constant
  else
    error("NonlinearExpressionException")
  end
end

# Double multiply
*(lhs::Real, rhs::Expression) = rhs * lhs
*(lhs::Real, rhs::Term) = rhs * lhs
*(lhs::Real, rhs::Variable) = rhs * lhs

# Expression add and subtract
+(first::Expression, second::Expression) = Expression(vcat(first.terms, second.terms), first.constant + second.constant)
+(first::Expression, second::Term) = Expression(push!(first.terms, second), first.constant)
+(expression::Expression, variable::Variable) = expression + Term(variable)
+(expression::Expression, constant::Real) = Expression(expression.terms, expression.constant + constant)
-(lhs::Expression, rhs::Expression) = lhs + (-rhs)
-(lhs::Expression, rhs::Real) = lhs + (-rhs)

# Term add and subtract
+(term::Term, expression::Expression) = expression + term
+(first::Term, second::Term) = Expression([first, second])
+(term::Term, variable::Variable) = term + Term(variable)
+(term::Term, constant::Real) = Expression(term, constant)
-(lhs::Term, rhs::Expression) = lhs + (-rhs)
-(lhs::Term, rhs::Real) = lhs + (-rhs)

# Variable add and subtract
+(variable::Variable, expression::Expression) = expression + variable
+(variable::Variable, term::Term) = term + variable
+(first::Variable, second::Variable) = Term(first) + second
+(variable::Variable, constant::Real) = Term(variable) + constant
-(lhs::Variable, rhs::Expression) = lhs + (-rhs)
-(lhs::Variable, rhs::Variable) = lhs + (-rhs)
-(lhs::Variable, rhs::Real) = lhs + (-rhs)

# Double add and subtract
+(lhs::Real, rhs::Expression) = rhs + lhs
+(lhs::Real, rhs::Term) = rhs + lhs
+(lhs::Real, rhs::Variable) = rhs + lhs
-(lhs::Real, rhs::Expression) = -rhs + lhs
-(lhs::Real, rhs::Term) = -rhs + lhs
-(lhs::Real, rhs::Variable) = -rhs + lhs

# Expression relations
==(first::Expression, second::Expression) = Constraint(first - second, :(==))
==(expression::Expression, term::Term) = expression == Expression(term)
==(expression::Expression, variable::Variable) = expression == Term(variable)
==(expression::Expression, constant::Real) = expression == Expression(constant)

<=(first::Expression, second::Expression) = Constraint(first - second, :(<=))
<=(expression::Expression, term::Term) = expression <= Expression(term)
<=(expression::Expression, variable::Variable) = expression <= Term(variable)
<=(expression::Expression, constant::Real) = expression <= Expression(constant)

>=(first::Expression, second::Expression) = Constraint(first - second, :(>=))
>=(expression::Expression, term::Term) = expression >= Expression(term)
>=(expression::Expression, variable::Variable) = expression >= Term(variable)
>=(expression::Expression, constant::Real) = expression >= Expression(constant)

# Term relations
==(lhs::Term, rhs::Expression) = rhs == lhs
==(lhs::Term, rhs::Union{Term,Variable,Real}) = Expression(lhs) == rhs
<=(lhs::Term, rhs::Union{Expression,Term,Variable,Real}) = Expression(lhs) <= rhs
>=(lhs::Term, rhs::Union{Expression,Term,Variable,Real}) = Expression(lhs) >= rhs

# Variable relations
==(variable::Variable, expression::Expression) = expression == variable
==(variable::Variable, term::Term) = term == variable
==(first::Variable, second::Variable) = Term(first) == second
==(variable::Variable, constant::Real) = Term(variable) == constant
<=(lhs::Variable, rhs::Union{Expression,Term,Variable,Real}) = Term(lhs) <= rhs
>=(variable::Variable, expression::Expression) = Term(variable) >= expression
>=(variable::Variable, term::Term) = term >= variable
>=(first::Variable, second::Variable) = Term(first) >= second
>=(variable::Variable, constant::Real) = Term(variable) >= constant

# Double relations
==(lhs::Real, rhs::Union{Expression,Term,Variable}) = rhs == lhs
<=(constant::Real, expression::Expression) = Expression(constant) <= expression
<=(constant::Real, term::Term) = constant <= Expression(term)
<=(constant::Real, variable::Variable) = constant <= Term(variable)
>=(constant::Real, term::Term) = Expression(constant) >= term
>=(constant::Real, variable::Variable) = constant >= Term(variable)

# Constraint strength modifier
function modify_strength(constraint::Constraint, strength::Real)
  Constraint(constraint, strength)
end

function suggest(s::Solver, v::Variable, value::Real)
  add_edit_variable(s, v, STRONG)
  suggest_value(s, v, value)
end

function edit_variables(func, s::Solver)
  func()
  update_variables(s)
  remove_all_edit_variables(s)
end

@testset "Kiwi" begin
  @testset "Smoke" begin
    s = Solver()
    x = Variable()
    y = Variable()

    add_constraint(s, x == 20)
    add_constraint(s, x + 2 == y + 10)
    update_variables(s)

    @test x.value == 20
    @test y.value == 12
  end

  @testset "Edits" begin
    s = Solver()
    x = Variable()
    y = Variable()

    add_constraint(s, x + 2 == y + 10)

    edit_variables(s) do
      suggest(s, x, 10)
    end

    @test x.value == 10
    @test y.value == 2
  end

  @testset "Underdefined" begin
    ale = Variable("ale")
    beer = Variable("beer")
    profit = Variable("profit")
    corn = Variable("corn")
    hops = Variable("hops")
    malt = Variable("malt")
    s = Solver()

    constraints = [
      13 * ale + 23 * beer == profit,
      5 * ale + 15 * beer <= corn,
      4 * ale + 4 * beer <= hops,
      35 * ale + 20 * beer <= malt,
      ale >= 0.0,
      beer >= 0.0,
      corn <= 480.0,
      hops <= 160.0,
      malt <= 1190.0,
      malt >= 0.0,
      corn >= 0.0,
      hops >= 0.0,
    ]

    foreach(constraints) do c
      add_constraint(s, c)
    end

    edit_variables(s) do
      suggest(s, profit, 1000)
    end

    @test profit.value ≈ 800 atol = ϵ
    @test ale.value ≈ 12 atol = ϵ
    @test beer.value ≈ 28 atol = ϵ
    @test corn.value ≈ 480 atol = ϵ
    @test hops.value ≈ 160 atol = ϵ
    # underdefined constraint
    @test (malt.value ≈ 980 || malt.value ≈ 1190)
  end

  @testset "More edits" begin
    s = Solver()
    x = Variable()
    y = Variable()
    width = Variable()
    height = Variable()
    xOffset = Variable()
    superWidth = Variable()
    superHeight = Variable()

    add_constraint(s, x == xOffset)
    add_constraint(s, y == 5)
    add_constraint(s, width == superWidth - 10)
    add_constraint(s, height == superHeight - 10)

    add_constraint(s, width >= 150)
    add_constraint(s, height >= 150)

    edit_variables(s) do
      suggest(s, xOffset, 5)
      suggest(s, superWidth, 800)
      suggest(s, superHeight, 600)
    end

    @test xOffset.value == 5
    @test x.value == 5
    @test y.value == 5
    @test width.value == 790
    @test height.value == 590
    @test superWidth.value == 800
    @test superHeight.value == 600
  end

  @testset "More more edits" begin
    s = Solver()
    x = Variable()
    y = Variable()
    width = Variable()
    height = Variable()
    xOffset = Variable()
    superWidth = Variable()
    superHeight = Variable()

    add_constraint(s, x == xOffset)
    add_constraint(s, y == 5)
    add_constraint(s, width == superWidth - 10)
    add_constraint(s, height == superHeight - 10)

    add_constraint(s, width >= 150)
    add_constraint(s, height >= 150)
    edit_variables(s) do
      suggest(s, width, 500)
      suggest(s, height, 400)
    end

    @test xOffset.value == 0
    @test x.value == 0
    @test y.value == 5
    @test width.value == 500
    @test height.value == 400
    @test superWidth.value == 510
    @test superHeight.value == 410
  end
end

export Solver, Variable, Constraint, Expression, Term, add_constraint, update_variables, optimize!

end

