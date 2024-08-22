
import numpy as np
from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
import scipy.sparse as sp

def f_score(precision, recall, beta=1.):
    """
        F-Beta score
            when beta = 1, this is equal to F1 score
    """
    if  ((precision * beta**2) + recall) ==0:
        fscore = 0
    else:
        fscore = ((1 + beta**2) * precision * recall) / ((precision * beta**2) + recall)
        
    return fscore

def my_prec_rec_fscore(labels_true, labels_pred, energy=None, beta=1.):

    if energy is None:
        energy_tmp = np.array([1. for i in labels_true])
    else:
        energy_tmp = np.array(energy) / np.max(np.array(energy))
        
    #(tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred, energy=energy)
    tn, fp, fn, tp, tn_en, fp_en, fn_en, tp_en, num_all = my_pair_confusion_matrix(labels_true, labels_pred, energy=energy_tmp)
    #first,second =pair_confusion_matrix(labels_true,labels_pred)
    #firstm =multilabel_confusion_matrix(labels_true,labels_pred, sample_weight = energy_tmp)
            
    # convert to Python integer types, to avoid overflow or underflow
    
    tn,fp,fn,tp = int(tn), int(fp), int(fn), int(tp)
    
    
    # Special cases: empty data or full agreement
    if tp == 0 and fp == 0:
        return 0., 0., 0.

    if energy is None:
        # precision, recall, f-beta score
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        fbeta = f_score(prec, rec, beta)
        
    else:
        prec = sum(tp_en) / (sum(tp_en) + sum(fp_en))
        rec = sum(tp_en) / (sum(tp_en) + sum(fn_en))
        fbeta = f_score(prec, rec, beta)
    return prec, rec, fbeta



def mutual_info_score(labels_true, labels_pred, *, contingency=None):


    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    #print(f"nzx: {nzx} nzy:{nzy} nz_val: {nz_val}")
    log_contingency_nm = np.log(nz_val)
    #print(f"log_contingency_nm: {log_contingency_nm}")
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.float32, copy=False) * pj.take(nzy).astype(
        np.float32, copy=False
    )
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)

def contingency_matrix(
    labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.float32, energy=None
):
    
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]    
    
    tr_idxs = None
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    if energy is None:
        
        contingency = sp.coo_matrix(
            (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
            shape=(n_classes, n_clusters),
            dtype=dtype,
        )
        contingency_en_weighted = sp.coo_matrix(
            (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
            shape=(n_classes, n_clusters),
            dtype=dtype,)
    else:
        contingency = sp.coo_matrix(
            (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
            shape=(n_classes, n_clusters),
            dtype=dtype,
        )
        contingency_en_weighted = sp.coo_matrix(
            (energy, (class_idx, cluster_idx)),
            shape=(n_classes, n_clusters),
            dtype=dtype,)
    #print(f"C: {contingency}")
    #print(f"C en: {contingency_en_weighted}")
    
    if sparse:
        contingency = contingency.tocsr()
        contingency_en_weighted = contingency_en_weighted.tocsr()
        #print(f"C: {contingency}")
        contingency.sum_duplicates()
        contingency_en_weighted.sum_duplicates()
        #print(f"C: {contingency}")
        #print(f"C en: {contingency_en_weighted}")
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency, contingency_en_weighted


def entropy(labels, energy):
    if len(labels) == 0:
        return 1.0
    
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    
    pi_en_weighted = np.array([0. for i in range(max(label_idx) + 1)])
    for l, e in zip(label_idx, energy):
        pi_en_weighted[l] += e
    pi_en_weighted = pi_en_weighted[pi_en_weighted > 0]
    #print(f"pi: {pi}\tpi_en_weighted: {pi_en_weighted}")

    # single cluster => zero entropy
    if pi.size == 1:
        return 0.0,0.0

    pi_sum = np.sum(pi)
    pi_weighted_sum = np.sum(pi_en_weighted)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    out = -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))
    out_en_weighed = float(-np.sum((pi_en_weighted / pi_weighted_sum) * (np.log(pi_en_weighted) - np.log(pi_weighted_sum))))
    return out, out_en_weighed

def my_homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0, energy=None):

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0
    
    if energy is None:
        energy = [1. for i in labels_true]
    else:
        energy = np.array(energy) / np.max(np.array(energy))

    entropy_C, entropy_C_weighted = entropy(labels_true, energy)
    entropy_K, entropy_K_weighted = entropy(labels_pred, energy)
    
    #print(f"C entr: {entropy_C},\tC entr weighted:{entropy_C_weighted}")
    #print(f"K entr: {entropy_K},\tK entr weighted:{entropy_K_weighted}")

    contingency, contingency_en_weighted = contingency_matrix(labels_true, labels_pred, sparse=True, energy=energy)
    MI = mutual_info_score(None, None, contingency=contingency_en_weighted)

    homogeneity = MI / (entropy_C_weighted) if entropy_C_weighted else 1.0
    completeness = MI / (entropy_K_weighted) if entropy_K_weighted else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness
            / (beta * homogeneity + completeness)
        )

    return homogeneity, completeness, v_measure_score

def pair_confusion_matrix(labels_true, labels_pred, energy):

    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency, contingency_en_weighted = contingency_matrix(
        labels_true, labels_pred, sparse=True, dtype=np.float32, energy=energy
    )
    #print(contingency_en_weighted)
    #print("--")
    #print(contingency)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    
    #print(f"n_c: {n_c}, n_k: {n_k}")
    sum_squares = (contingency.data**2).sum()
    #print(sum_squares)
    #n_samples = energy.sum()
    #print(n_samples)
    C = np.empty((2, 2), dtype=np.float32)
    C[1, 1] = sum_squares - n_samples # tp
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares # fp
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares # fn
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares  # tn
    return C

def create_pairs_from_labels(labels, energy, all_pairs=False):
    pairs, pair_en, explored_labels = [], [], []
    
    for i in range(len(labels)):
        l = labels[i]
        
        if i != len(labels):
            for j in range(i+1, len(labels)):
                if l == labels[j] or all_pairs:
                    pairs.append((i, j))
                    pair_en.append(energy[i] + energy[j])
                    if l not in explored_labels:
                        explored_labels.append(l)
                
#         if l not in explored_labels:
#             pairs.append((i, i))    
#             pair_en.append(energy[i])
            
    return pairs, pair_en

def my_pair_confusion_matrix(labels_true, labels_pred, energy):
    
    true_pairs, true_pair_en = create_pairs_from_labels(labels_true, energy)
    pred_pairs, pred_pair_en = create_pairs_from_labels(labels_pred, energy)
    all_pairs, all_pair_en = create_pairs_from_labels(labels_true, energy, all_pairs=True)
    
    true_pairs = set(true_pairs)
    pred_pairs = set(pred_pairs)
    all_pairs = set(all_pairs)
    
    tp_pairs = true_pairs & pred_pairs
    fn_pairs = true_pairs - pred_pairs
    fp_pairs = pred_pairs - true_pairs
    tn_pairs = all_pairs - (true_pairs | pred_pairs)
    
    tp_en = [energy[p1] + energy[p2] for p1, p2 in tp_pairs]
    fn_en = [energy[p1] + energy[p2] for p1, p2 in fn_pairs]
    fp_en = [energy[p1] + energy[p2] for p1, p2 in fp_pairs]
    tn_en = [energy[p1] + energy[p2] for p1, p2 in tn_pairs]
    
    tn = len(tn_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)
    tp = len(tp_pairs)
    
    num_all = len(all_pairs)
    
    return tn, fp, fn, tp, tn_en, fp_en, fn_en, tp_en, num_all


def my_adjusted_rand_score(labels_true, labels_pred, energy=None, weighted=False):

    if energy is None:
        energy = np.array([1. for i in labels_true])
    else:
        energy = np.array(energy) / np.max(np.array(energy))
        
    #(tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred, energy=energy)
    tn, fp, fn, tp, tn_en, fp_en, fn_en, tp_en, num_all = my_pair_confusion_matrix(labels_true, labels_pred, energy=energy)
    
    #print(f"tn: {tn} fp: {fp} fn: {fn} tp: {tp}")
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    if not weighted:
        return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    else:
        return 2.0 * (sum(tp_en) * sum(tn_en) - sum(fn_en) * sum(fp_en)) / ((sum(tp_en) + sum(fn_en)) * (sum(fn_en) + sum(tn_en)) + (sum(tp_en) + sum(fp_en)) * (sum(fp_en) + sum(tn_en)))


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )

def my_adjusted_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic", energy = None):
   
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    if energy is None:
        energy = np.array([1. for i in labels_true])
    else:
        energy = np.array(energy) / np.max(np.array(energy))
        
    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0
    
    _,contingency = contingency_matrix(labels_true, labels_pred, sparse=True,energy = energy)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true,energy), entropy(labels_pred,energy)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return ami

