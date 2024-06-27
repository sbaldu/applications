
import numpy as np
from dataclasses import dataclass


@dataclass()
class Efficiency:
    average: float
    total: float


@dataclass()
class Purity:
    average: float
    total: float


# blacklist = []
def evaluate_efficiency_cl_vs_tr(clusters: list, recos: list) -> Efficiency:
    '''
    Calculate the efficiency of the pre-clustering by comparing the
    cluster-vertices with the truth ones.

    We find for each cluster the reco-vertex that it recostructs better,
    meaning that the highest number of points in the clusters are also
    contained in the reco-vertex.

    Efficiency is calculated by only considering clusters with a partial
    efficiency of 50% or more. We also consider "reconstructible" only the
    reco-vertices with more than 4 tracks.
    '''

    efficiencies = []
    for cluster in clusters:
        eff = [0. for _ in recos]   # efficiencies for a cluster
        vec = [[] for _ in recos]
        for reco_id, reco_vertex in enumerate(recos):
            if len(reco_vertex) != 0:
                for id in cluster:
                    if id in reco_vertex:
                        eff[reco_id] += 1.
                        vec[reco_id].append(id)
                # divide by the expected number of tracks in the vertex
                # to obtain the ratio
                eff[reco_id] /= len(reco_vertex)
                eff[reco_id] *= 100.
        efficiencies.append(max(eff))   # take the highest efficiency
        best_reco = recos[np.argmax(eff)]
        # for j in best_reco:
        #     if j not in vec[np.argmax(eff)]:
        #         blacklist.append(j)
    passed = len([e for e in efficiencies if e > 50.])
    reconstructible_recos = len([r for r in recos if len(r) > 4])
    total_eff = passed / reconstructible_recos * 100.
    print(f"Average efficiency: {np.mean(efficiencies)}%")
    print(f"Total efficiency: {total_eff}%")

    return Efficiency(np.mean(efficiencies), total_eff)


def evaluate_efficiency_tr_vs_cl(clusters: list, recos: list) -> None:
    efficiencies = []
    for reco in recos:
        eff = [0. for _ in clusters]
        for cl_id, cluster in enumerate(clusters):
            for id in reco:
                if id in cluster:
                    eff[cl_id] += 1.
            eff[cl_id] /= len(cluster)
            eff[cl_id] *= 100.
        if max(eff) != 0.:
            efficiencies.append(max(eff))
    print(efficiencies)
    print(f"Average efficiency: {np.mean(efficiencies)}%")

