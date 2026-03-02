# optimization/pyomo_converter.py
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, minimize

class PyomoConverter:
    """Converts ProblemData into a Pyomo model."""
    def __init__(self, problem_data):
        self.problem = problem_data

    def to_pyomo(self):
        G, c = self.problem.G, self.problem.c
        model = ConcreteModel()

        N = self.problem.a0.size
        model.x = Var(range(N), domain=NonNegativeReals)

        def constraint_rule(model, i):
            row = G.getrow(i)
            return sum(row.data[k] * model.x[row.col[k]] for k in range(len(row.data))) == c[i]

        model.constraints = Constraint(range(G.shape[0]), rule=constraint_rule)
        model.obj = Objective(expr=sum(model.x[i] for i in range(N)), sense=minimize)
        return model
