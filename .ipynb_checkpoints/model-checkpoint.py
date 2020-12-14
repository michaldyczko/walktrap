import numpy as np
import networkx as nx


class Walktrap:
    def __init__(self, g):
        self.g = g
        self.n = g.number_of_nodes()
        self.communities = [{n} for g.nodes()]
        
    def choose_communities():
        delta_sigma_ks = {}
        for C1 in self.communities:
            for C2 in self.adjacent_communities(C1):
                delta_sigma_ks[(C1, C2)] = self.delta_sigma_k(C1, C2)
        return min(delta_sigma_ks, key=delta_sigma_ks.get)
    
    def adjacent_communities(C1):
        adj_communities = []
        for C2 in self.communities:
            g = self.g.subgraph(C1 | C2)
            if nx.is_connected(g):
                adj_communities.append(C2)
        return adj_communities

    def delta_sigma_k(self, C1, C2):
        return 1/self.n * (len(C1)*len(C2)) / (len(C1)+len(C2)) * self.r2(C1, C2)

    def r2(self, C1, C2):
        return np.linalg.norm(np.matmul(np.power(self.D, -1/2), self.Pt(C1)) - np.matmul(np.power(self.D, -1/2), self.Pt(C2)))
        
    def Pt(self, C):
        return np.array(
            [self.Ptj(C, j) for j in range(self.n)]
        )
    
    def Ptj(self, C, j):
        return np.mean([self.Ptij(i, j) for i in C])
    
    def Ptij(self, i, j):
        return nx.adjacency_matrix(self.g)[i, j] / self.g.degree[i]
    
    def merge_communities(C1, C2):
        C3 = C1 | C2
        self.communities.append(C3)
        self.communities.remove(C1)
        self.communities.remove(C2)