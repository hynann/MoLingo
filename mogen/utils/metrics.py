import numpy as np
from scipy import linalg
import torch
import logging

def calculate_R_precision_ms(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_mpjpe_with_bs(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, :, [0]].mean(2)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=2)
    pelvis = pred_joints[:, :, [0]].mean(2)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=2) # [bs,T, 22, 3]

    # print(f'gt joints shape {gt_joints.shape}')
    # print(f'pred joints shape {pred_joints.shape}')

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22

    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]

    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist

def calculate_activation_statistics_normalized(activations):
    """
    with normalization (as TMR embeddings should be used with norm 1)
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations / np.linalg.norm(activations, axis=-1)[:, None]
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)

    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def print_latex_metrics(metrics, ranks=[1, 2, 3, 5, 10], t2m=True, m2t=True, MedR=True):
    vals = [str(x).zfill(2) for x in ranks]
    t2m_keys = [f"t2m/R{i}" for i in vals]
    if MedR:
        t2m_keys += ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals]
    if MedR:
        m2t_keys += ["m2t/MedR"]

    keys = []
    if t2m:
        keys += t2m_keys
    if m2t:
        keys += m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    if "t2m/len" in metrics:
        print("Number of samples: {}".format(int(metrics["t2m/len"])))
    else:
        print("Number of samples: {}".format(int(metrics["m2t/len"])))
    print(str_)

    ### Norm part for action recognition

    norm_keys = []
    for key in keys:
        if f"{key}_norm" in metrics:
            norm_keys.append(f"{key}_norm")

    str_ = "& " + " & ".join([ff(metrics[key]) for key in norm_keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in norm_keys}
    print(dico)
    print(str_)


def all_contrastive_metrics(
        sims, emb=None, threshold=None, rounding=2, return_cols=False, m2t=True, t2m=True
):
    if not t2m and not m2t:
        logging.warning("No metrics asked to be computed")
        return None

    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    if t2m:
        t2m_m, t2m_cols = contrastive_metrics(
            sims, text_selfsim, threshold, return_cols=True, rounding=rounding
        )
    if m2t:
        m2t_m, m2t_cols = contrastive_metrics(
            sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
        )

    all_m = {}
    if t2m:
        keys = t2m_m.keys()
    else:
        keys = m2t_m.keys()
    for key in keys:
        if t2m:
            all_m[f"t2m/{key}"] = t2m_m[key]
        if m2t:
            all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    if m2t:
        all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        if not m2t:
            m2t_cols = None
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
        sims,
        text_selfsim=None,
        threshold=None,
        return_cols=False,
        rounding=2,
        break_ties="optimistically",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim >= real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries=None, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]

    if num_queries is None:
        num_queries = len(cols)
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics


def contrastive_metrics_m2t_action_retrieval(
        sims,
        motion_cat_idx,
        return_cols=False,
        rounding=2,
        break_ties="averaging",
        norm_metrics=True
):
    n, m = sims.shape
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx == i]
            cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))

        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(np.mean([elt[k] for elt in cat_metrics]), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)

    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval(
        sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):
    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m


def contrastive_metrics_m2t_action_retrieval_multi_labels(
        sims,
        motion_cat_idx,
        return_cols=False,
        rounding=2,
        break_ties="averaging",
        norm_metrics=True
):
    n, m = sims.shape
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    motion_cat_idx = [cat_idx[np.argmin([dists[i, elt] for elt in cat_idx])] for i, cat_idx in
                      enumerate(motion_cat_idx)]
    gt_dists = dists[range(n), motion_cat_idx]
    gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if norm_metrics:
        motion_cat_idx = np.array(motion_cat_idx)
        cat_metrics = []
        for i in range(np.max(motion_cat_idx) + 1):
            cols_cat = cols[motion_cat_idx == i]
            if len(cols_cat) > 0:
                cat_metrics.append(cols2metrics(cols_cat, rounding=rounding))

        print("len(cat_metrics) : ", len(cat_metrics))

        metrics_norm = {}
        keys = cat_metrics[0].keys()
        for k in keys:
            metrics_norm[f"{k}_norm"] = round(float(np.mean([elt[k] for elt in cat_metrics])), 2)

    metrics = cols2metrics(cols, num_queries, rounding=rounding)

    if norm_metrics:
        metrics.update(metrics_norm)

    if return_cols:
        return metrics, cols
    return metrics


def all_contrastive_metrics_action_retrieval_multi_labels(
        sims, motion_cat_idx, rounding=2, return_cols=False, norm_metrics=True
):
    m2t_m, m2t_cols = contrastive_metrics_m2t_action_retrieval_multi_labels(
        sims.T, motion_cat_idx, return_cols=True, rounding=rounding, norm_metrics=norm_metrics
    )

    all_m = {}
    keys = m2t_m.keys()
    for key in keys:
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, m2t_cols
    return all_m

