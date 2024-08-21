from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch
from einops import rearrange
from tqdm import tqdm

ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")

# TODO: FID Score
# Compare Distributions of generated and real images


# image similarity metric
def ssim_distance(img1, img2):
    """
    Compute the Structural Similarity Index (SSIM) between grayscale MNIST images.

    :param img1: tensor of shape (28, 28) with values in [0, 1]
    :param img2: tensor of shape (28, 28) with values in [0, 1]

    :return: SSIM distance between img1 and img2 -> torch.Tensor of shape (B?, 1) * 10 ^ 2
    """
    return ((ssim(img1, img2) * -1) + 1) / 2


def pairwise_ssim_distance(imgs1, imgs2):
    """
    Compute the Structural Similarity Index (SSIM) between grayscale MNIST images.

    :param imgs1: tensor of shape (B1, 1, 28, 28) with values in [0, 1]
    :param imgs2: tensor of shape (B2, 1, 28, 28) with values in [0, 1]

    :return: Matrix of SSIM distances between imgs1 and imgs2 -> torch.Tensor of shape (B1, B2)
    """
    B1 = imgs1.size(0)
    B2 = imgs2.size(0)

    # TODO: compare every image in imgs1 with every image in imgs2
    distances = torch.zeros(B1, B2)
    for i in range(B1):
        # give images as single batch to ssim_distance function
        # have tensor with shape (B1, 1, 28, 28) with the i-th image repeated B1 times
        img1repeat = imgs1[i].repeat(B2, 1, 1, 1)
        distances[i] = ssim_distance(img1repeat, imgs2)

    return distances


# metrics measures
# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def lgan_mmd_cov(distances):
    """
    Compute the Minimum Matching Distance (MMD) between two sets of images.
    """
    N_sample, N_ref = distances.size(0), distances.size(1)
    all_dist = distances
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


def compute_all_metrics(generated, reference):
    result = {}

    distances = pairwise_ssim_distance(generated, reference)
    distances_gen = pairwise_ssim_distance(generated, generated)
    distances_ref = pairwise_ssim_distance(reference, reference)

    # compute knn
    knn_result = knn(distances_gen, distances, distances_ref, 1)
    for key, value in knn_result.items():
        result[f"KNN-{key}"] = value

    # compute lgan_mmd_cov
    result.update(lgan_mmd_cov(distances))
    return result


def emd_approx(sample, ref):
    emd_val = torch.zeros([sample.size(0)]).to(sample)
    return emd_val


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(ref_pcs, batch_size, diff_module, n_points, reduced=True):
    N_ref = ref_pcs.shape[0]

    cd_lst = []
    iterator = range(0, N_ref, batch_size)

    for b_start in tqdm(iterator, desc="EMD-CD"):
        b_end = min(N_ref, b_start + batch_size)
        ref_batch = ref_pcs[b_start:b_end]
        sample_x_0s = diff_module.diff.sample(len(ref_batch))
        sample_meshes, _ = diff_module.generate_meshes(sample_x_0s, None)
        sample_batch = []
        for mesh in sample_meshes:
            pc = torch.tensor(mesh.sample(n_points))
            sample_batch.append(pc)
        sample_batch = torch.stack(sample_batch)
        # sample_batch = torch.randn(*ref_batch.shape).double()
        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

    if reduced:
        cd = torch.cat(cd_lst).mean()
    else:
        cd = torch.cat(cd_lst)

    results = {
        "MMD-CD": cd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc="Pairwise EMD-CD")
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def compute_all_metrics_3D(sample_pcs, ref_pcs, batch_size):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## EMD
    res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())

    # We use the below code to visualize some goodly&badly performing shapes
    # you can uncomment if you want to analyze that

    # if len(sample_pcs) > 60:
    #     r = Rotation.from_euler('x', 90, degrees=True)
    #     min_dist, min_dist_sample_idx = torch.min(M_rs_cd.t(), dim=0)
    #     min_dist_sorted_idx = torch.argsort(min_dist)
    #     orig_meshes_dir = f"orig_meshes/run_{wandb.run.name}"
    #
    #     for i, ref_id in enumerate(min_dist_sorted_idx):
    #         ref_id = ref_id.item()
    #         matched_sample_id = min_dist_sample_idx[ref_id].item()
    #         mlp_pc = trimesh.points.PointCloud(sample_pcs[matched_sample_id].cpu())
    #         mlp_pc.export(f"{orig_meshes_dir}/mlp_top{i}.obj")
    #         mlp_pc = trimesh.points.PointCloud(ref_pcs[ref_id].cpu())
    #         mlp_pc.export(f"{orig_meshes_dir}/mlp_top{i}ref.obj")
    #
    #     for i, ref_id in enumerate(min_dist_sorted_idx[:4]):
    #         ref_id = ref_id.item()
    #         matched_sample_id = min_dist_sample_idx[ref_id].item()
    #         logger.experiment.log({f'pc/top_{i}': [
    #             wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
    #             wandb.Object3D(r.apply(ref_pcs[ref_id].cpu()))]})
    #     for i, ref_id in enumerate(reversed(min_dist_sorted_idx[-4:])):
    #         ref_id = ref_id.item()
    #         matched_sample_id = min_dist_sample_idx[ref_id].item()
    #         logger.experiment.log({f'pc/bottom_{i}': [
    #             wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
    #             wandb.Object3D(r.apply(ref_pcs[ref_id].cpu()))]})
    #     print(min_dist, min_dist_sample_idx, min_dist_sorted_idx)
    #     torch.save(min_dist, f"{orig_meshes_dir}/min_dist.pth")
    #     torch.save(min_dist_sample_idx, f"{orig_meshes_dir}/min_dist_sample_idx.pth")
    #     torch.save(min_dist_sorted_idx, f"{orig_meshes_dir}/min_dist_sorted_idx.pth")
    #     torch.save(min_dist[min_dist_sorted_idx], f"{orig_meshes_dir}/min_dist_sorted.pth")
    #     print("Sorted:", min_dist[min_dist_sorted_idx])

    results.update({"%s-CD" % k: v for k, v in res_cd.items()})
    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k}
    )

    ## EMD
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    return results
