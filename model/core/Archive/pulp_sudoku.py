import pyomo.environ as pyo
import numpy as np

def sets():

    model = pyo.ConcreteModel()

    modelA = pyo.Set()
    # model B is a Set with tuples of two dimension

    model.B = pyo.Set(dimen=2, )

    # sets
    #The initialize option can accept any Python iterable, including a set, list, or tuple. This data may be returned from a function or specified directly as in

    modelA.initialize = [1, 2, 3, 4, 5]
    model.B.initialize = [(1, 2), (3, 4), (5, 6)]

    # model X is initialized with a initilization function initialization functions need to have the "signature as an input"

    def initX(model):
        return [(i, i+1) for i in range(10)]

    model.X = pyo.Set(initialize=initX)

    model.X.pprint()


    # example convert random array with dim 2, 3, 4 into a set
    arr = np.random.rand(2, 3, 4)
    
    
    # Generate the index tuples
    index_tuples = [(i, j, k) for i in range(2) for j in range(3) for k in range(4)]

    # we generate a set of indexes
    model.IDX = pyo.Set(initialize=index_tuples)

    # convert the data into a pyo parameter
    model.data = pyo.Param(model.IDX, initialize=lambda model, i, j, k: arr[i, j, k])

    # print the model
    model.pprint()

    # we can also initialize a set by an other set... the index is the

    model.Z = pyo.Set(model.IDX)
    model.Z.pprint()

    # we can do a bunch of operations with sets
    model.B = pyo.Set(initialize = [(1, 2), (3, 4), (5, 6)])
    model.C = pyo.Set(initialize = [(1, 2), (3, 4), (5, 6), (7, 8)])

    # D is the sum of B and C
    model.D = model.B | model.C

    model.E = model.B & model.C  # Intersection of B and C
    model.E.pprint()

if __name__ == "__main__":
    sets()