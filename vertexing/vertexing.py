
import validate as val
import uproot
from glob import glob
import CLUEstering as clue
# import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

save = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'save':
        save = True

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
# print('\n')

#
# Initialize the clusterer
#
c = clue.clusterer(1., 1., 1.)
c.read_data([z, pt])

#
# Cache the sets of parameters
#
parameters = []
parameters.append((.0091, 9., 1000.))
parameters.append((.0091, 1., 1000.))
parameters.append((.010, 1., 51.))
parameters.append((.0061, 9., 1000.))
parameters.append((.0071, 9., 1000.))
parameters.append((.0081, 9., 1000.))
parameters.append((.0091, 9.5, 1000.))

#
# Run the clustering
#
avg_eff = []
total_eff = []
for pars in parameters:
    print('\n')
    print(f"Parameters: dc={pars[0]}, rhoc={pars[1]}, odf={pars[2]}")
    c.set_params(*pars)
    c.run_clue()
    print(f"number of clusters = {c.clust_prop.n_clusters}")
    if not save:
        c.cluster_plotter()

    clusters = [[] for _ in range(c.clust_prop.n_clusters)]
    for i, c_id in enumerate(c.clust_prop.cluster_ids):
        if c_id != -1:
            clusters[c_id].append(i)
    recos = [[] for _ in range(len(sim_id))]
    for i, reco in enumerate(sim_id):
        recos[reco].append(i)

    val.evaluate_efficiency_cl_vs_tr(clusters, recos)

# print(np.unique(blacklist))
# print(np.sort([z[i] for i in np.unique(blacklist)]))
# plt.hist([z[i] for i in np.unique(blacklist)],
#           bins=len(np.unique(blacklist)))
# plt.show()
# evaluate_efficiency_tr_vs_cl(clusters, recos)

###
# check not connected tracks in reco 0
###
reco0 = recos[148]
cluster0 = clusters[0]
b0 = []
for i in reco0:
    if i not in cluster0:
        b0.append(i)
# print(b0)
# print([z[i] for i in recos[148]])
# print([z[i] for i in recos[142]])
