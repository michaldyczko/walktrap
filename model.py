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
        self.r2_cache = dict()
        self.delta_sigma_k_cache = dict()
        self.Pt_cache = dict()
        self.Ptj_cache = dict()
        self.Ptj_cache = dict()
        self.modularities = {}
        self.communities = [{n} for n in g.nodes()]
        self.D_transformed = np.diag(np.ravel(list(dict(g.degree).values()))**(-0.5))
        self.Ptij = np.divide(nx.adjacency_matrix(self.g).todense(), list(dict(g.degree).values()))

    def choose_communities(self):
        community_pairs = [(tuple(C1), tuple(C2)) for C1 in self.communities for C2 in self.adjacent_communities(C1)]
        dct = {(C1, C2): self.delta_sigma_k(C1, C2) for C1, C2 in community_pairs}
        return min(dct, key=dct.get)
    
    def adjacent_communities(self, C1):
        adj_communities = []
        for n1 in C1:
            for C2 in self.communities:
                if C2 in adj_communities:
                    continue
                if C1 == C2:
                    continue
                for n2 in C2:
                    if self.g.has_edge(n1, n2) and C2 not in adj_communities:
                        adj_communities.append(C2)
                        break
        return adj_communities

    def delta_sigma_k(self, C1, C2):
        if (C1, C2) in self.delta_sigma_k_cache:
            return self.delta_sigma_k_cache[C1, C2]
        elif (C2, C1) in self.delta_sigma_k_cache:
            return self.delta_sigma_k_cache[C2, C1]
        else:
            val = 1/self.n * (len(C1)*len(C2)) / (len(C1)+len(C2)) * self.r2(C1, C2)
            self.delta_sigma_k_cache[C1, C2] = val
            self.delta_sigma_k_cache[C2, C1] = val
        return val

    def r2(self, C1, C2):
        if (C1, C2) in self.r2_cache:
            return self.r2_cache[C1, C2]
        elif (C2, C1) in self.r2_cache:
            return self.r2_cache[C2, C1]
        else:
            val = np.linalg.norm(np.matmul(self.D_transformed, self.Pt(C1)-self.Pt(C2)))
            self.r2_cache[C1, C2] = val
            self.r2_cache[C2, C1] = val
        return val

    def Pt(self, C):
        if C in self.Pt_cache:
            return self.Pt_cache[C]
        else:
            val = np.array(
                [self.Ptj(C, j) for j in range(self.n)]
            )
            self.Pt_cache[C] = val
        return val
    
    def Ptj(self, C, j):
        if (C, j) in self.Ptj_cache:
            return self.Ptj_cache[C, j]
        else:
            val = np.mean([self.Ptij[i, j] for i in C])
            self.Ptj_cache[C, j] = val
        return np.mean([self.Ptij[i, j] for i in C])
    
    
    def merge_communities(self, C1, C2):
        C3 = set(C1) | set(C2)
        
#         print(C1, C2, C3)
        for C in [C for C in self.communities]:
            C = tuple(C)
            if C != C1 and C != C2:
                val = ((len(C1) + len(C)) * self.delta_sigma_k(C1, C) + (len(C2) + len(C)) * self.delta_sigma_k(C2, C) - len(C) * self.delta_sigma_k(C1, C2)) / (len(C1) + len(C2) + len(C))
                self.delta_sigma_k_cache[tuple(C3), C] = val
                self.delta_sigma_k_cache[C, tuple(C3)] = val
                
        for C in [C for C in self.communities]:
            C = tuple(C)
            try:
                del self.delta_sigma_k_cache[C1, C]
                del self.delta_sigma_k_cache[C, C1]
                del self.delta_sigma_k_cache[C2, C]
                del self.delta_sigma_k_cache[C, C2]
            except KeyError:
                pass

        self.communities.append(C3)
        self.communities.remove(set(C1))
        self.communities.remove(set(C2))

    def iterate(self):
        while len(self.communities) > 1:
            C1, C2 = self.choose_communities()
            self.merge_communities(C1, C2)
            self.modularities[tuple(tuple(C) for C in self.communities)] = modularity(self.g, self.communities) 


if __name__ == "__main__":
    G = nx.barabasi_albert_graph(100, 3, seed = 2137)
    walk = Walktrap(G)
    walk.iterate()