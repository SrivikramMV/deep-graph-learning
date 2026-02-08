import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
from typing import Dict, Tuple


class GraphRicciCurvature:

    def __init__(self, G: nx.Graph, beta: float = 1.0):
        self.G = G.copy()
        self.beta = beta
        self._cache = {}

        self.node_list = list(self.G.nodes())
        self.n = len(self.node_list)
        self.node_index = {node: i for i, node in enumerate(self.node_list)}

        # Precompute structures
        self._compute_laplacian()
        self._compute_heat_kernel()
        self._compute_shortest_path_matrix()


    def _compute_laplacian(self):
        A = nx.adjacency_matrix(self.G, nodelist=self.node_list).toarray()
        D = np.diag(np.sum(A, axis=1))
        self.L = D - A

    def _compute_heat_kernel(self):
        self.H = expm(-self.beta * self.L)

        # normalize rows 
        row_sums = self.H.sum(axis=1)
        self.H = self.H / row_sums[:, np.newaxis]

    def _compute_shortest_path_matrix(self):
        sp = dict(nx.all_pairs_shortest_path_length(self.G))
        self.distances = np.zeros((self.n, self.n))

        for i, u in enumerate(self.node_list):
            for j, v in enumerate(self.node_list):
                self.distances[i, j] = sp[u][v]


    def _wasserstein_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute W1 distance between two probability distributions
        using linear programming.
        """

        n = self.n
        c = self.distances.flatten()

        A_eq = []
        b_eq = []

        for i in range(n):
            row = np.zeros(n * n)
            row[i * n:(i + 1) * n] = 1
            A_eq.append(row)
            b_eq.append(p[i])

        for j in range(n):
            col = np.zeros(n * n)
            col[j::n] =1
            A_eq.append(col)
            b_eq.append(q[j])

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        bounds = [(0, None)] * (n * n)

        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if not result.success:
            raise RuntimeError("error")

        return result.fun


    def compute_ricci_curvature(self, edge: Tuple[int, int]) -> float:
        if edge in self._cache:
            return self._cache[edge]

        u, v = edge

        i = self.node_index[u]
        j = self.node_index[v]

        p = self.H[i]
        q = self.H[j]

        W = self._wasserstein_distance(p, q)

        d = self.distances[i, j]

        if d == 0:
            curvature = float("inf")
        else:
            curvature = 1 - W / d

        self._cache[edge] = curvature
        return curvature

    def compute_all_curvatures(self) -> Dict[Tuple[int, int], float]:
        return {
            edge: self.compute_ricci_curvature(edge)
            for edge in self.G.edges()
        }


# Example usage

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (1, 4), (2, 4)
    ])

    ricci = GraphRicciCurvature(G, beta=1.0)
    curvatures = ricci.compute_all_curvatures()

    for edge, value in curvatures.items():
        print(f"Edge {edge}: Ricci curvature = {value:.4f}")
