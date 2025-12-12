import itertools

import networkx as nx
import pulp
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, LpInteger


def solve_packing_coloring(G):
    """
    Solves the packing coloring problem on graph G.

    Parameters:
      - G: a NetworkX graph.

    Returns:
      - color_assignment: a dictionary mapping each vertex to its assigned color.
      - z_val: the packing chromatic number of the graph.
    """
    k = G.number_of_nodes()

    # Create the optimization model.
    model = LpProblem(name='PackingColoring', sense=LpMinimize)

    # Decision variables: x[(v, i)] equals 1 if vertex v is assigned color i, 0 otherwise.
    x = {(v, i): LpVariable(name=f"x_{v}_{i}", cat=LpBinary) for v in G.nodes for i in range(1, k + 1)}

    # Variable z representing the maximum color used (1 to k).
    z = LpVariable(name="z", lowBound=1, upBound=k, cat=LpInteger)

    # Objective: minimize z (the maximum color number used).
    model += z

    # Constraint (2): Each vertex must receive exactly one color.
    for v in G.nodes:
        model += lpSum(x[(v, i)] for i in range(1, k + 1)) == 1, f"OneColor_{v}"

    # Constraint (3): For every pair of distinct vertices (v,u) and each color i,
    # if the distance between v and u is <= i, they cannot both be assigned color i.
    distances = dict(nx.all_pairs_shortest_path_length(G))
    for i in range(1, k + 1):
        for v, u in itertools.combinations(G.nodes, 2):
            d = distances[v].get(u)
            if d is not None and d <= i:
                model += x[(v, i)] + x[(u, i)] <= 1, f"Pack_{v}_{u}_color_{i}"

    # Constraint (4): If a vertex v is assigned color i then i must be <= z.
    for v in G.nodes:
        for i in range(1, k + 1):
            model += i * x[(v, i)] <= z, f"MaxColor_{v}_{i}"

    # Solve the model
    model.solve(pulp.CPLEX_PY(msg=False))

    if model.status == pulp.LpStatusOptimal:  # Optimal solution found
        z_val = z.varValue

        # Extract color assignment for each vertex.
        color_assignment = {}
        for v in G.nodes:
            for i in range(1, k + 1):
                if x[(v, i)].varValue > 0.5:
                    color_assignment[v] = i
                    break
        return color_assignment, z_val
    else:
        return None, None
