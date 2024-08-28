
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataclasses import dataclass

@dataclass()
class Result:
    table: np.ndarray
    efficiency: float
    purity: float
    fake_rate: float
    duplicate_rate: float
    merge_rate: float

def compute_efficiency(table: np.ndarray, n_sims: int) -> float:
    reconstructed = 0
    for sim_matches in table:
        matches = len(np.nonzero(sim_matches)[0])
        if matches >= 1:
            reconstructed += 1

    return reconstructed / n_sims

def compute_purity(table: np.ndarray, reco_vertices:list, sim_vertices: list) -> float:
    pure_recos = 0.
    for sim_id, sim_vertex in enumerate(sim_vertices):
        matches = np.nonzero(table[sim_id])[0]
        for reco_id in matches:
            reco_vertex = reco_vertices[reco_id]
            noise_tracks = 0
            for track in reco_vertex:
                if track not in sim_vertex:
                    noise_tracks += 1
            if noise_tracks < 0.2 * len(reco_vertex):
                pure_recos += 1
    assert pure_recos <= len(reco_vertices)
    return pure_recos / len(reco_vertices)

def compute_fake_rate(table: np.ndarray, n_recos: int) -> float:
    fakes = 0
    for reco_vertex_matches in table.T:
        if len(np.nonzero(reco_vertex_matches)[0]) == 0:
            fakes += 1
    assert fakes <= n_recos
    return fakes / n_recos

def compute_duplicate_rate(table: np.ndarray, n_recos: int) -> float:
    duplicates = set([])
    for sim_vertex_matches in table:
        matches = np.nonzero(sim_vertex_matches)[0]
        if len(matches) > 1:
            duplicates.union(set(matches))
    assert len(duplicates) <= n_recos
    return len(duplicates) / n_recos

def compute_merge_rate(table: np.ndarray, n_recos: int) -> float:
    merged = 0
    for reco_vertex_matches in table.T:
        matches = np.nonzero(reco_vertex_matches)[0]
        if len(matches) > 1:
            merged += 1
    assert merged <= n_recos
    return merged / n_recos

def evaluate_reconstruction(reco_vertices: list, sim_vertices: list) -> Result:
    # table of simulated vertices and matching clusters
    table = np.zeros((len(sim_vertices), len(reco_vertices)))
    for sim_id, sim_vertex in enumerate(sim_vertices):
        for reco_id, reco_vertex in enumerate(reco_vertices):
            count = 0
            for track_id in reco_vertex:
                if track_id in sim_vertex:
                    count += 1
            # check if the cluster contains more than half of the sim vertex
            if count > (len(sim_vertex) / 2):
                table[sim_id][reco_id] = count

    efficiency = compute_efficiency(table, len(sim_vertices))
    purity = compute_purity(table, reco_vertices, sim_vertices)
    fake_rate = compute_fake_rate(table, len(reco_vertices))
    duplicate_rate = compute_duplicate_rate(table, len(reco_vertices))
    merge_rate = compute_merge_rate(table, len(reco_vertices))

    return Result(table, efficiency, purity, fake_rate, duplicate_rate, merge_rate)
