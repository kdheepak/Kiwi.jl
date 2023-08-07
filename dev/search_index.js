var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Kiwi","category":"page"},{"location":"#Kiwi","page":"Home","title":"Kiwi","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Kiwi.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Kiwi]","category":"page"},{"location":"#Kiwi.add_constraint-Tuple{Solver, Constraint}","page":"Home","title":"Kiwi.add_constraint","text":"Add a constraint to the solver.\n\nErrors:\n\nDuplicateConstraint: The given constraint has already been added to the solver.\nUnsatisfiableConstraint: The given constraint is required and cannot be satisfied.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.add_edit_variable-Tuple{Solver, Variable, Real}","page":"Home","title":"Kiwi.add_edit_variable","text":"Add an edit variable to the solver.\n\nThis method should be called before the suggestValue method is used to supply a suggested value for the given edit variable.\n\nErrors:\n\nDuplicateEditVariable: The given edit variable has already been added to the solver.\nBadRequiredStrength: The given strength is >= required.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.add_with_artificial_variable-Tuple{Solver, Kiwi.Row}","page":"Home","title":"Kiwi.add_with_artificial_variable","text":"Add the row to the tableau using an artificial variable.\n\nThis will return false if the constraint cannot be satisfied.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.all_dummies-Tuple{Kiwi.Row}","page":"Home","title":"Kiwi.all_dummies","text":"Test whether a row is composed of all dummy variables.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.any_pivotable_symbol-Tuple{Solver, Kiwi.Row}","page":"Home","title":"Kiwi.any_pivotable_symbol","text":"Get the first Slack or Error symbol in the row.\n\nIf no such symbol is present, and Invalid symbol will be returned.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.choose_subject-Tuple{Kiwi.Row, Kiwi.Tag}","page":"Home","title":"Kiwi.choose_subject","text":"Choose the subject for solving for the row.\n\nThis method will choose the best subject for using as the solve target for the row. An invalid symbol will be returned if there is no valid target.\n\nThe symbols are chosen according to the following precedence:\n\nThe first symbol representing an external variable.\nA negative slack or error tag variable.\n\nIf a subject cannot be found, an invalid symbol will be returned.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.coefficient-Tuple{Kiwi.Row, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.coefficient","text":"Get the coefficient for the given symbol.\n\nIf the symbol does not exist in the row, zero will be returned.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.create_row-Tuple{Solver, Constraint, Kiwi.Tag}","page":"Home","title":"Kiwi.create_row","text":"Create a new Row object for the given constraint.\n\nThe terms in the constraint will be converted to cells in the row. Any term in the constraint with a coefficient of zero is ignored. This method uses the getVarSymbol method to get the symbol for the variables added to the row. If the symbol for a given cell variable is basic, the cell variable will be substituted with the basic row.\n\nThe necessary slack and error variables will be added to the row. If the constant for the row is negative, the sign for the row will be inverted so the constant becomes positive.\n\nThe tag will be updated with the marker and error symbols to use for tracking the movement of the constraint in the tableau.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.dual_optimize!-Tuple{Solver}","page":"Home","title":"Kiwi.dual_optimize!","text":"Optimize the system using the dual of the simplex method.\n\nThe current state of the system should be such that the objective function is optimal, but not feasible. This method will perform an iteration of the dual simplex method to make the solution both optimal and feasible.\n\nErrors:\n\nInternalSolverError: The system cannot be dual optimized.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.get_dual_entering_symbol-Tuple{Solver, Kiwi.Row}","page":"Home","title":"Kiwi.get_dual_entering_symbol","text":"Compute the entering symbol for the dual optimize operation.\n\nThis method will return the symbol in the row which has a positive coefficient and yields the minimum ratio for its respective symbol in the objective function. The provided row must be infeasible. If no symbol is found which meats the criteria, an invalid symbol is returned.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.get_entering_symbol-Tuple{Kiwi.Row}","page":"Home","title":"Kiwi.get_entering_symbol","text":"Compute the entering variable for a pivot operation.\n\nThis method will return first symbol in the objective function which is non-dummy and has a coefficient less than zero. If no symbol meets the criteria, it means the objective function is at a minimum, and an invalid symbol is returned.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.get_leaving_row-Tuple{Solver, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.get_leaving_row","text":"Compute the row which holds the exit symbol for a pivot.\n\nThis method will return an iterator to the row in the row map which holds the exit symbol. If no appropriate exit symbol is found, the end() iterator will be returned. This indicates that the objective function is unbounded.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.get_marker_leaving_row-Tuple{Solver, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.get_marker_leaving_row","text":"Compute the leaving row for a marker variable.\n\nThis method will return an iterator to the row in the row map which holds the given marker variable. The row will be chosen according to the following precedence:\n\nThe row with a restricted basic varible and a negative coefficient  for the marker with the smallest ratio of -constant / coefficient.\nThe row with a restricted basic variable and the smallest ratio  of constant / coefficient.\nThe last unrestricted row which contains the marker.\n\nIf the marker does not exist in any row, the row map end() iterator will be returned. This indicates an internal solver error since the marker should exist somewhere in the tableau.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.hasConstraint-Tuple{Solver, Constraint}","page":"Home","title":"Kiwi.hasConstraint","text":"Test whether a constraint has been added to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.has_edit_variable-Tuple{Solver, Variable}","page":"Home","title":"Kiwi.has_edit_variable","text":"Test whether an edit variable has been added to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.insert!","page":"Home","title":"Kiwi.insert!","text":"Insert a row into this row with a given coefficient.\n\nThe constant and the cells of the other row will be multiplied by the coefficient and added to this row. Any cell with a resulting coefficient of zero will be removed from the row.\n\n\n\n\n\n","category":"function"},{"location":"#Kiwi.insert!-2","page":"Home","title":"Kiwi.insert!","text":"Insert a symbol into the row with a given coefficient.\n\nIf the symbol already exists in the row, the coefficient will be added to the existing coefficient. If the resulting coefficient is zero, the symbol will be removed from the row.\n\n\n\n\n\n","category":"function"},{"location":"#Kiwi.optimize!-Tuple{Solver, Kiwi.Row}","page":"Home","title":"Kiwi.optimize!","text":"Optimize the system for the given objective function.\n\nThis method performs iterations of Phase 2 of the simplex method until the objective function reaches a minimum.\n\nErrors:\n\nInternalSolverError: The value of the objective function is unbounded.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.remove!-Tuple{Kiwi.Row, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.remove!","text":"Remove the given symbol from the row.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.remove_constraint-Tuple{Solver, Constraint}","page":"Home","title":"Kiwi.remove_constraint","text":"Remove a constraint from the solver.\n\nErrors:\n\nUnknownConstraint: The given constraint has not been added to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.remove_constraint_effects-Tuple{Solver, Constraint, Kiwi.Tag}","page":"Home","title":"Kiwi.remove_constraint_effects","text":"Remove the effects of a constraint on the objective function.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.remove_edit_variable-Tuple{Solver, Variable}","page":"Home","title":"Kiwi.remove_edit_variable","text":"Remove an edit variable from the solver.\n\nError:\n\nUnknownEditVariable: The given edit variable has not been added to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.remove_marker_effects-Tuple{Solver, Kiwi.KiwiSymbol, Real}","page":"Home","title":"Kiwi.remove_marker_effects","text":"Remove the effects of an error marker on the objective function.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.reverse_sign!-Tuple{Kiwi.Row}","page":"Home","title":"Kiwi.reverse_sign!","text":"Reverse the sign of the constant and all cells in the row.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.runtests-Tuple","page":"Home","title":"Kiwi.runtests","text":"Kiwi.runtests(pattern...; kwargs...)\n\nEquivalent to ReTest.retest(Kiwi, pattern...; kwargs...). This function is defined automatically in any module containing a @testset, possibly nested within submodules.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.solve!-Tuple{Kiwi.Row, Kiwi.KiwiSymbol, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.solve!","text":"Solve the row for the given symbols.\n\nThis method assumes the row is of the form x = b * y + c and will solve the row such that y = x / b - c / b. The rhs symbol will be removed from the row, the lhs added, and the result divided by the negative inverse of the rhs coefficient.\n\nThe lhs symbol must not exist in the row, and the rhs symbol must exist in the row.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.solve!-Tuple{Kiwi.Row, Kiwi.KiwiSymbol}","page":"Home","title":"Kiwi.solve!","text":"Solve the row for the given symbol.\n\nThis method assumes the row is of the form a * x + b * y + c = 0 and (assuming solve for x) will modify the row to represent the right hand side of x = -b/a * y - c / a. The target symbol will be removed from the row, and the constant and other cells will be multiplied by the negative inverse of the target coefficient.\n\nThe given symbol must exist in the row.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.substitute!-Tuple{Kiwi.Row, Kiwi.KiwiSymbol, Kiwi.Row}","page":"Home","title":"Kiwi.substitute!","text":"Substitute a symbol with the data from another row.\n\nGiven a row of the form a * x + b and a substitution of the form x = 3 * y + c the row will be updated to reflect the expression 3 * a * y + a * c + b.\n\nIf the symbol does not exist in the row, this is a no-op.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.substitute!-Tuple{Solver, Kiwi.KiwiSymbol, Kiwi.Row}","page":"Home","title":"Kiwi.substitute!","text":"Substitute the parametric symbol with the given row.\n\nThis method will substitute all instances of the parametric symbol in the tableau and the objective function with the given row.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.suggest_value-Tuple{Solver, Variable, Real}","page":"Home","title":"Kiwi.suggest_value","text":"Suggest a value for the given edit variable.\n\nThis method should be used after an edit variable as been added to the solver in order to suggest the value for that variable.\n\nErrors:\n\nUnknownEditVariable: The given edit variable has not been added to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#Kiwi.update_variables-Tuple{Solver}","page":"Home","title":"Kiwi.update_variables","text":"Update the values of the external solver variables.\n\n\n\n\n\n","category":"method"}]
}
