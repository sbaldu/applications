
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import davies_bouldin_score as db_score
from metrics import *
import validate as val
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


def select_reconstructibles(sim_vertices: list) -> list:
    """
    Select only the reconstructible vertices
    """
    return [r for r in sim_vertices if len(r) > 4]

#
# Load the data
#
input_folder = "./data/"
files = glob(f"{input_folder}/*.root")
print(f"Number of files: {len(files)}")

outputPath = './dataset_tracks/'
offset = 0
N = 1
for i_file, file in enumerate(files[offset:]):
    i_file += offset
    if i_file >= N:
        break
    try:
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

    except Exception as e:
        print(f"Error: {e}")
        continue

    z = np.array(track_data.gnn_z_pca)
    pt = np.array(track_data.gnn_pt)
    sim_id = np.array(track_data.gnn_sim_vertex_ID)
    eta = np.array(track_data.gnn_eta)

# remove tracks with pt < 0.8
z, pt, sim_id, eta = pt_cut(z, pt, sim_id, eta, cut=0.8)

# remove tracks with eta > 2.4
z, pt, sim_id, eta = eta_cut(z, pt, sim_id, eta, cut=2.4)

#
# Initialize the clusterer
#
c = clue.clusterer(1., 1., 1.)
c.read_data([z, pt])
c.input_plotter(x_label='z [cm]', y_label='')

#
# Cache the sets of parameters
#
parameters = []
parameters.append((.1, 9., 10.))
parameters.append((.1, 9., 1.))
parameters.append((.05, 9., 10.))
parameters.append((.05, 9., 1.))
parameters.append((.1, 9., .5))
parameters.append((.1, 9., .1))
parameters.append((.05, 9., .5))
parameters.append((.05, 9., .1))
# try to get maximum efficiency
c.set_params(.01, 9., 1.)
c.run_clue()
c.cluster_plotter(pt_size=30, seed_size=40)
clusters = [[] for _ in range(c.n_seeds)]
for i, c_id in enumerate(c.clust_prop.cluster_ids):
    if c_id != -1:
        clusters[c_id].append(i)
recos = [[] for _ in range(len(sim_id))]
for i, reco in enumerate(sim_id):
    recos[reco].append(i)
recos = [r for r in recos if len(r) > 4]

# reconstruct the sim vertices from the sim ids
# sim_vertices = construct_sim_vertices(sim_id)
# filter out the non-reconstructible vertices
# sim_vertices = select_reconstructibles(sim_vertices)

result = val.evaluate_efficiency(clusters, recos)
print(f"Efficiency: {result.efficiency}")
print(f"Purity: {result.purity}")
result2 = val.evaluate_efficiency_duplicates(clusters, recos)
print(f"Efficiency dup: {result2.efficiency}")
print(f"Purity dup: {result2.purity}")
# val.evaluate_purity(clusters, recos)
print(f'The homogeneity score is {homogeneity_score(sim_id, c.cluster_ids)}')
print(f'The completeness score is {completeness_score(sim_id, c.cluster_ids)}')
print(f'The mutual info score is {normalized_mutual_info_score(sim_id, c.cluster_ids)}')
print(my_homogeneity_completeness_v_measure(sim_id, c.cluster_ids, energy=pt))


# try to get maximum purity
c.set_params(.1, 10., .1)
c.run_clue()
c.cluster_plotter(pt_size=30, seed_size=40)
clusters = [[] for _ in range(c.n_seeds)]
for i, c_id in enumerate(c.clust_prop.cluster_ids):
    if c_id != -1:
        clusters[c_id].append(i)
recos = [[] for _ in range(len(sim_id))]
for i, reco in enumerate(sim_id):
    recos[reco].append(i)
recos = [r for r in recos if len(r) > 4]

result = val.evaluate_efficiency(clusters, recos)
print(f"Efficiency: {result.efficiency}")
print(f"Purity: {result.purity}")
result2 = val.evaluate_efficiency_duplicates(clusters, recos)
print(f"Efficiency dup: {result2.efficiency}")
print(f"Purity dup: {result2.purity}")
# val.evaluate_purity(clusters, recos)
print(f'The homogeneity score is {homogeneity_score(sim_id, c.cluster_ids)}')
print(f'The completeness score is {completeness_score(sim_id, c.cluster_ids)}')
print(f'The mutual info score is {normalized_mutual_info_score(sim_id, c.cluster_ids)}')

c.set_params(.1, 10., .06)
c.run_clue()
c.cluster_plotter(pt_size=30, seed_size=40)
clusters = [[] for _ in range(c.n_seeds)]
for i, c_id in enumerate(c.clust_prop.cluster_ids):
    if c_id != -1:
        clusters[c_id].append(i)
recos = [[] for _ in range(len(sim_id))]
for i, reco in enumerate(sim_id):
    recos[reco].append(i)
recos = [r for r in recos if len(r) > 4]

result = val.evaluate_efficiency(clusters, recos)
print(f"Efficiency: {result.efficiency}")
print(f"Purity: {result.purity}")
result2 = val.evaluate_efficiency_duplicates(clusters, recos)
print(f"Efficiency dup: {result2.efficiency}")
print(f"Purity dup: {result2.purity}")

# val.evaluate_purity(clusters, recos)
print(f'The homogeneity score is {homogeneity_score(sim_id, c.cluster_ids)}')
print(f'The completeness score is {completeness_score(sim_id, c.cluster_ids)}')
print(f'The mutual info score is {normalized_mutual_info_score(sim_id, c.cluster_ids)}')


# Run the clustering
dc_values = np.arange(0.001, 1., 0.001)
n_clusters = []
efficiency = []
purity = []
homogeneity = []
completeness = []
mutual_info = []
for dc in dc_values:
    c.set_params(dc, 9., 1.)
    c.run_clue()
    n_clusters.append(c.n_clusters)
    clusters = [[] for _ in range(c.n_seeds)]
    for i, c_id in enumerate(c.cluster_ids):
        if c_id != -1:
            clusters[c_id].append(i)
    recos = [[] for _ in range(len(sim_id))]
    for i, reco in enumerate(sim_id):
        recos[reco].append(i)
    recos = [r for r in recos if len(r) > 4]

    efficiency.append(val.evaluate_efficiency(clusters, recos).efficiency)
    purity.append(val.evaluate_efficiency(clusters, recos).purity)
    homogeneity.append(homogeneity_score(sim_id, c.cluster_ids))
    completeness.append(completeness_score(sim_id, c.cluster_ids))
    mutual_info.append(normalized_mutual_info_score(sim_id, c.cluster_ids))

# plot number of clusters
plt.plot(dc_values, n_clusters, marker='o')
plt.title('Number of clusters vs. dc')
plt.xlabel('dc')
plt.ylabel('Number of clusters')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot efficiency
plt.plot(dc_values, efficiency, marker='o')
plt.title('Efficiency vs. dc')
plt.xlabel('dc')
plt.ylabel('Efficiency')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot purity
plt.plot(dc_values, purity, marker='o')
plt.title('Purity vs. dc')
plt.xlabel('dc')
plt.ylabel('Purity')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot homogeneity, completeness and mutual info
plt.plot(dc_values, homogeneity, ':o', label='homogeneity')
plt.plot(dc_values, completeness, '--^', label='completeness')
plt.plot(dc_values, mutual_info, '-.s', label='mutual info')
plt.xlabel('$\delta c$', fontsize=12)
plt.ylabel('Metrics score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(lw=0.5, ls='--', alpha=0.5)
plt.legend(fontsize=12)
plt.show()

# d = DBSCAN(eps=0.1, min_samples=9)
# d.fit(np.array([z, [0. for _ in z]]).T)
# labels = d.labels_
# plt.scatter(z, [0. for _ in z], c=labels, cmap='viridis')
# plt.show()
dc_values = np.arange(0.001, 1., 0.001)
n_clusters = []
efficiency = []
purity = []
homogeneity = []
completeness = []
mutual_info = []
for dc in dc_values:
    c.set_params(0.01, 9., dc)
    c.run_clue()
    n_clusters.append(c.n_clusters)
    clusters = [[] for _ in range(c.n_seeds)]
    for i, c_id in enumerate(c.cluster_ids):
        if c_id != -1:
            clusters[c_id].append(i)
    recos = [[] for _ in range(len(sim_id))]
    for i, reco in enumerate(sim_id):
        recos[reco].append(i)

    efficiency.append(val.evaluate_efficiency(clusters, recos).efficiency)
    purity.append(val.evaluate_efficiency(clusters, recos).purity)
    homogeneity.append(homogeneity_score(sim_id, c.cluster_ids))
    completeness.append(completeness_score(sim_id, c.cluster_ids))
    mutual_info.append(normalized_mutual_info_score(sim_id, c.cluster_ids))

# plot number of clusters
plt.plot(dc_values, n_clusters, marker='o')
plt.title('Number of clusters vs. dc')
plt.xlabel('dc')
plt.ylabel('Number of clusters')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot efficiency
plt.plot(dc_values, efficiency, marker='o')
plt.title('Efficiency vs. dc')
plt.xlabel('dc')
plt.ylabel('Efficiency')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot purity
plt.plot(dc_values, purity, marker='o')
plt.title('Purity vs. dc')
plt.xlabel('dc')
plt.ylabel('Purity')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot homogeneity, completeness and mutual info
plt.plot(dc_values, homogeneity, ':o', label='homogeneity')
plt.plot(dc_values, completeness, '--^', label='completeness')
plt.plot(dc_values, mutual_info, '-.s', label='mutual info')
plt.xlabel('$\delta c$', fontsize=12)
plt.ylabel('Metrics score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(lw=0.5, ls='--', alpha=0.5)
plt.legend(fontsize=12)
plt.show()

dc_values = np.arange(0.001, 1., 0.001)
n_clusters = []
efficiency = []
purity = []
homogeneity = []
completeness = []
mutual_info = []
for dc in dc_values:
    c.set_params(0.1, 9., dc)
    c.run_clue()
    n_clusters.append(c.n_clusters)
    clusters = [[] for _ in range(c.n_seeds)]
    for i, c_id in enumerate(c.cluster_ids):
        if c_id != -1:
            clusters[c_id].append(i)
    recos = [[] for _ in range(len(sim_id))]
    for i, reco in enumerate(sim_id):
        recos[reco].append(i)

    efficiency.append(val.evaluate_efficiency(clusters, recos).efficiency)
    purity.append(val.evaluate_efficiency(clusters, recos).purity)
    homogeneity.append(homogeneity_score(sim_id, c.cluster_ids))
    completeness.append(completeness_score(sim_id, c.cluster_ids))
    mutual_info.append(normalized_mutual_info_score(sim_id, c.cluster_ids))

# plot number of clusters
plt.plot(dc_values, n_clusters, marker='o')
plt.title('Number of clusters vs. dc')
plt.xlabel('dc')
plt.ylabel('Number of clusters')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot efficiency
plt.plot(dc_values, efficiency, marker='o')
plt.title('Efficiency vs. dc')
plt.xlabel('dc')
plt.ylabel('Efficiency')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot purity
plt.plot(dc_values, purity, marker='o')
plt.title('Purity vs. dc')
plt.xlabel('dc')
plt.ylabel('Purity')
plt.grid(lw=0.5, ls='--')
plt.show()

# plot homogeneity, completeness and mutual info
plt.plot(dc_values, homogeneity, ':o', label='homogeneity')
plt.plot(dc_values, completeness, '--^', label='completeness')
plt.plot(dc_values, mutual_info, '-.s', label='mutual info')
plt.xlabel('$\delta c$', fontsize=12)
plt.ylabel('Metrics score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(lw=0.5, ls='--', alpha=0.5)
plt.legend(fontsize=12)
plt.show()

