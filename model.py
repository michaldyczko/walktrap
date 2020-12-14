import numpy as np
import networkx as nx
import scipy
from time import time

from scipy.linalg import fractional_matrix_power
from networkx.algorithms.community.quality import modularity

class Walktrap:
    def __init__(self, g):
        self.g = g
        self.n = g.number_of_nodes()
        self.communities = [{n} for n in g.nodes()]
        self.D_transformed = np.diag(np.ravel(list(dict(g.degree).values()))**(-0.5))
        self.modularities = {}
        self.Ptij = np.divide(nx.adjacency_matrix(self.g).todense(), list(dict(g.degree).values()))


    def choose_communities(self):
        community_pairs = {(tuple(C1), tuple(C2)) for C1 in self.communities for C2 in self.adjacent_communities(C1)}
        delta_sigma_ks = dict()
        for C1, C2 in community_pairs:
            delta_sigma_ks[C1, C2] = self.delta_sigma_k(C1, C2)
        # for C1 in self.communities:
        #     for C2 in self.adjacent_communities(C1):
        return min(delta_sigma_ks, key=delta_sigma_ks.get)
    
    def adjacent_communities(self, C1):
        adj_communities = []
        for C2 in self.communities:
            if C1 == C2:
                continue
            g = self.g.subgraph(C1 | C2)
            if nx.is_connected(g):
                adj_communities.append(C2)
        return adj_communities

    def delta_sigma_k(self, C1, C2):
        return 1/self.n * (len(C1)*len(C2)) / (len(C1)+len(C2)) * self.r2(C1, C2)

    def r2(self, C1, C2):
        return np.linalg.norm(np.matmul(self.D_transformed, self.Pt(C1)) - np.matmul(self.D_transformed, self.Pt(C2)))

    def Pt(self, C):
        return np.array(
            [self.Ptj(C, j) for j in range(self.n)]
        )
    
    def Ptj(self, C, j):
        return np.mean([self.Ptij[i, j] for i in C])
    

    
    def merge_communities(self, C1, C2):
        C3 = set(C1) | set(C2)
        # print(C1, C2, C3)
        self.communities.append(C3)
        self.communities.remove(set(C1))
        self.communities.remove(set(C2))

    def iterate(self):
        while len(self.communities) > 1:
            print(len(self.communities))
            C1, C2 = self.choose_communities()
            self.merge_communities(C1, C2)
            self.modularities[tuple(tuple(C) for C in self.communities)] = modularity(self.g, self.communities)


if __name__ == "__main__":
    G = nx.barabasi_albert_graph(100, 3, seed = 2137)
    start = time()
    walk = Walktrap(G)
    walk.iterate()
    print(walk.modularities)
    print(time() - start)