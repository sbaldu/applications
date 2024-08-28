
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import davies_bouldin_score as db_score
from metrics import *
from validation import evaluate_reconstruction
import uproot
from glob import glob
import CLUEstering as clue
import matplotlib.pyplot as plt
import numpy as np

def remove_outliers(data: np.ndarray, labels: np.ndarray, truths: np.ndarray) -> tuple:
    """
    Remove outliers from the data
    """
    new_data = []
    new_labels = []
    new_truths = []
    for i in range(len(labels)):
        if labels[i] != -1:
            new_data.append(data[i])
            new_labels.append(labels[i])
            new_truths.append(truths[i])
    return (np.array(new_data), np.array(new_labels), np.array(new_truths))

def pt_cut(z, pt, sim_id, eta, cut) -> tuple:
    new_z = []
    new_pt = []
    new_sim_id = []
    new_eta = []
    for i in range(len(pt)):
        if pt[i] > cut:
            new_z.append(z[i])
            new_pt.append(pt[i])
            new_sim_id.append(sim_id[i])
            new_eta.append(eta[i])
    return new_z, new_pt, new_sim_id, new_eta


def eta_cut(z, pt, sim_id, eta, cut) -> tuple:
    new_z = []
    new_pt = []
    new_sim_id = []
    new_eta = []
    for i in range(len(eta)):
        if eta[i] < cut:
            new_z.append(z[i])
            new_pt.append(pt[i])
            new_sim_id.append(sim_id[i])
            new_eta.append(eta[i])
    return new_z, new_pt, new_sim_id, new_eta

def construct_sim_vertices(sim_ids: list) -> list:
    sims = [[] for _ in range(max(sim_ids)+1)]
    for track_id, vertex_id in enumerate(sim_ids):
        sims[vertex_id].append(track_id)

    return sims

def construct_reco_vertices(c: clue.clusterer) -> list:
    recos = [[] for _ in range(c.n_seeds)]
    for i, c_id in enumerate(c.cluster_ids):
        if c_id != -1:
            recos[c_id].append(i)

    return recos

def select_reconstructibles(sim_vertices: list) -> list:
    """
    Select only the reconstructible vertices
    """
    return [r for r in sim_vertices if len(r) > 4]

def test_odf(c: clue.clusterer,
             dc: float,
             odf_range,
             sim_vertices: list,
             sim_ids: list) -> tuple:

    efficiency = []
    purity = []
    fake_rate = []
    duplicate_rate = []
    merge_rate = []
    for odf in odf_range:
        c.set_params(dc, 10., odf)
        c.run_clue()
        reco_vertices = construct_reco_vertices(c)

        result = evaluate_reconstruction(reco_vertices, sim_vertices)
        efficiency.append(result.efficiency)
        purity.append(result.purity)
        fake_rate.append(result.fake_rate)
        duplicate_rate.append(result.duplicate_rate)
        merge_rate.append(result.merge_rate)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(odf_range * dc, efficiency, 'b-o', label='Efficiency')
    ax[0].plot(odf_range * dc, purity, 'r--^', label='Purity')
    ax[0].set_xlabel('$\delta_o$', fontsize=16)
    ax[0].set_ylabel('Efficiency/Purity', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    ax[0].set_title(f'Efficiency/Purity vs ODF, $\delta_c$={dc}', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[0].grid(lw=0.5, ls='--', alpha=0.8)

    ax[1].plot(odf_range * dc, fake_rate, 'c--*', label='Fake Rate')
    ax[1].plot(odf_range * dc, duplicate_rate, 'm:+', label='Duplicate Rate')
    ax[1].plot(odf_range * dc, merge_rate, 'y-.v', label='Merge Rate')
    ax[1].set_xlabel('$\delta_o$', fontsize=16)
    ax[1].set_ylabel('Rates', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0., 1.)
    ax[1].set_title(f'Rate vs ODF, $\delta_c$={dc}', fontsize=16)
    ax[1].legend(fontsize=14)
    ax[1].grid(lw=0.5, ls='--', alpha=0.8)
    # plt.show()
    plt.savefig(f'plots/vertexing_dc_{dc}_rhoc_{rhoc}.pdf')
    plt.clf()

def test_rhoc(c: clue.clusterer,
             dc: float,
             odf: float,
             rhoc_range,
             sim_vertices: list,
             sim_ids: list) -> tuple:

    efficiency = []
    purity = []
    fake_rate = []
    duplicate_rate = []
    merge_rate = []
    for rhoc in rhoc_range:
        c.set_params(dc, rhoc, odf)
        c.run_clue()
        reco_vertices = construct_reco_vertices(c)

        result = evaluate_reconstruction(reco_vertices, sim_vertices)
        efficiency.append(result.efficiency)
        purity.append(result.purity)
        fake_rate.append(result.fake_rate)
        duplicate_rate.append(result.duplicate_rate)
        merge_rate.append(result.merge_rate)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(rhoc_range, efficiency, 'b-o', label='Efficiency')
    ax[0].plot(rhoc_range, purity, 'r--^', label='Purity')
    ax[0].set_xlabel('$\delta_o$', fontsize=16)
    ax[0].set_ylabel('Efficiency/Purity', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    ax[0].set_title(f'Efficiency/Purity vs ODF, $\delta_c$={dc}', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[0].grid(lw=0.5, ls='--', alpha=0.8)

    ax[1].plot(rhoc_range, fake_rate, 'c--*', label='Fake Rate')
    ax[1].plot(rhoc_range, duplicate_rate, 'm:+', label='Duplicate Rate')
    ax[1].plot(rhoc_range, merge_rate, 'y-.v', label='Merge Rate')
    ax[1].set_xlabel('rho_c', fontsize=16)
    ax[1].set_ylabel('Rates', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax[1].set_title(f'Rate vs ODF, $\delta_c$={dc}', fontsize=16)
    ax[1].legend(fontsize=14)
    ax[1].grid(lw=0.5, ls='--', alpha=0.8)
    # plt.show()
    plt.savefig(f'plots/vertexing_dc_{dc}_odf_{odf}.pdf')


input_folder = "./data/"
files = glob(f"{input_folder}/*.root")

for i_file, file in enumerate(files):
    with uproot.open(file) as f:
        t = f["TTree"]
        # for branch in t.branches:
        #     print(branch)

        track_data = t.arrays(["gnn_pt",
                               "gnn_z_pca",
                               "gnn_sim_vertex_ID",
                               "gnn_eta"])

        number_of_tracks = len(track_data.gnn_pt)
        print(f"Have {number_of_tracks} tracks in the file")

    z = np.array(track_data.gnn_z_pca)
    pt = np.array(track_data.gnn_pt)
    sim_id = np.array(track_data.gnn_sim_vertex_ID)
    eta = np.array(track_data.gnn_eta)

# remove tracks with pt < 0.8
z, pt, sim_id, eta = pt_cut(z, pt, sim_id, eta, cut=0.8)
# remove tracks with eta > 2.4
z, pt, sim_id, eta = eta_cut(z, pt, sim_id, eta, cut=2.4)

# reconstruct the sim vertices from the sim ids
sim_vertices = construct_sim_vertices(sim_id)
# filter out the non-reconstructible vertices
sim_vertices = select_reconstructibles(sim_vertices)

# run clustering
c = clue.clusterer(0.15, 10., .11)
c.read_data([z, pt])
c.run_clue()
print(f"Found {c.n_seeds} clusters")
c.cluster_plotter(pt_size=30., seed_size=40.)

# construct the reco vertices
reco_vertices = construct_reco_vertices(c)

result = evaluate_reconstruction(reco_vertices, sim_vertices)
print(f'Efficiency = {100 * result.efficiency}%')
print(f'Purity = {100 * result.purity}%')
print(f'fake_rate = {100 * result.fake_rate}%')
print(f'duplicate_rate = {100 * result.duplicate_rate}%')
print(f'merge_rate = {100 * result.merge_rate}%')
print(f'homogeneity = {homogeneity_score(sim_id, c.cluster_ids)}')
print(f'completeness = {completeness_score(sim_id, c.cluster_ids)}')
print(f'mutual info. = {normalized_mutual_info_score(sim_id, c.cluster_ids)}')
weighted_measures = my_homogeneity_completeness_v_measure(sim_id, c.cluster_ids, energy=pt)
print(f'weighted homogeneity = {weighted_measures[0]}')
print(f'weighted completeness = {weighted_measures[1]}')
print(f'weighted v-measure = {weighted_measures[2]}')

test_odf(c, 0.05, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.10, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.15, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.20, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.25, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.30, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.35, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.40, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.45, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
test_odf(c, 0.50, np.arange(0.01, 0.9, 0.01), sim_vertices, sim_id)
