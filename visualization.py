import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, wasserstein_distance


def ensure_sequoia_on_path(sequoia_root: Optional[str]) -> None:
    if sequoia_root:
        if sequoia_root not in sys.path:
            sys.path.insert(0, sequoia_root)


def load_h5(path: str):
    with h5py.File(path, "r") as f:
        fk = None
        ck = None
        for k in ["features", "feats", "X"]:
            if k in f:
                fk = k
                break
        for k in ["coords", "coordinates", "xy"]:
            if k in f:
                ck = k
                break
        if fk and ck:
            return f[fk][:], f[ck][:], dict(f.attrs)
        if len(f.keys()) == 1:
            grp = f[list(f.keys())[0]]
            fk = None
            ck = None
            for k in ["features", "feats", "X"]:
                if k in grp:
                    fk = k
                    break
            for k in ["coords", "coordinates", "xy"]:
                if k in grp:
                    ck = k
                    break
            if fk and ck:
                return grp[fk][:], grp[ck][:], dict(f.attrs)
    raise ValueError(f"Could not find features/coords in {path}")


def load_positions_csv(base_dir: str) -> Optional[str]:
    spatial_dir = os.path.join(base_dir, "spatial")
    candidates = [
        os.path.join(spatial_dir, "tissue_positions_list.csv"),
        os.path.join(spatial_dir, "tissue_positions.csv"),
        os.path.join(base_dir, "tissue_positions_list.csv"),
        os.path.join(base_dir, "tissue_positions.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def coords_from_positions_csv(path: str, barcodes: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    if "barcode" not in df.columns:
        df = pd.read_csv(path, header=None)
        df.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row",
            "pxl_col",
        ]

    row_col = None
    if "pxl_row_in_fullres" in df.columns and "pxl_col_in_fullres" in df.columns:
        row_col = ("pxl_row_in_fullres", "pxl_col_in_fullres")
    elif "pxl_row" in df.columns and "pxl_col" in df.columns:
        row_col = ("pxl_row", "pxl_col")
    else:
        raise ValueError(f"No pixel columns found in {path}")

    df = df.set_index("barcode")
    missing = [b for b in barcodes if b not in df.index]
    if missing:
        raise ValueError(f"Missing {len(missing)} barcodes in {path}")
    coords = df.loc[barcodes, [row_col[1], row_col[0]]].astype(float).values
    return coords


def load_gene_list(genes_pkl: Optional[str], gene_list_path: Optional[str]) -> List[str]:
    genes = None
    if genes_pkl:
        with open(genes_pkl, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "genes" in obj:
            genes = obj["genes"]
        elif isinstance(obj, list) and obj:
            if isinstance(obj[0], dict) and "genes" in obj[0]:
                genes = obj[0]["genes"]
            elif isinstance(obj[-1], dict) and "genes" in obj[-1]:
                genes = obj[-1]["genes"]

    if genes is None and gene_list_path:
        if gene_list_path.endswith(".npy"):
            genes = np.load(gene_list_path, allow_pickle=True)
        else:
            df = pd.read_csv(gene_list_path)
            if "gene" in df.columns:
                genes = df["gene"].tolist()
            else:
                genes = df.iloc[:, 0].tolist()

    if genes is None:
        raise ValueError("Could not infer gene list; provide --gene_list or --genes_pkl")
    return list(genes)


def load_genes_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "gene" in df.columns:
        return df["gene"].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_state(model_path: str, device: str) -> Dict[str, torch.Tensor]:
    if model_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required to load .safetensors files") from exc
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    return strip_state_dict_prefix(state)


def infer_model_dims(state_dict: Dict[str, torch.Tensor]):
    input_dim = None
    num_outputs = None

    for key, value in state_dict.items():
        if key.endswith("linear_head.1.weight"):
            num_outputs = value.shape[0]
            input_dim = value.shape[1]
            break

    if input_dim is None:
        for key, value in state_dict.items():
            if key.endswith("linear_head.0.weight"):
                input_dim = value.shape[0]
                break

    if input_dim is None:
        for key, value in state_dict.items():
            if key.endswith("pos_emb1D"):
                input_dim = value.shape[1]
                break

    return input_dim, num_outputs


def load_model(model_path: str, model_type: str, device: str, num_clusters: int):
    state = load_checkpoint_state(model_path, device)
    input_dim, num_outputs = infer_model_dims(state)
    if input_dim is None or num_outputs is None:
        raise ValueError("Could not infer model dimensions from checkpoint")

    if model_type == "vis":
        from src.tformer_lin import ViS

        model = ViS(
            num_outputs=num_outputs,
            input_dim=input_dim,
            depth=6,
            nheads=16,
            dimensions_f=64,
            dimensions_c=64,
            dimensions_s=64,
            num_clusters=num_clusters,
            device=str(device),
        )
    elif model_type == "vit":
        from src.vit import ViT

        model = ViT(
            num_outputs=num_outputs,
            dim=input_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            num_clusters=num_clusters,
            device=str(device),
        )
    else:
        raise ValueError("model_type must be 'vis' or 'vit'")

    model_state = model.state_dict()
    filtered_state = {}
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    return model, input_dim, num_outputs


def smooth_expression_by_neighbors(coords: np.ndarray, values: np.ndarray, k=6, sigma=1.0):
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)
    median_nn = np.median(dists[:, 1])
    weights = np.exp(-0.5 * (dists / (sigma * median_nn)) ** 2)
    return np.sum(values[idxs] * weights, axis=1) / np.sum(weights, axis=1)


def map_patch_preds_to_spots(
    patch_centers: np.ndarray,
    patch_preds: np.ndarray,
    spot_coords_full: np.ndarray,
    row_offset: int,
    col_offset: int,
    k_neigh: int = 3,
    weight_mode: str = "gaussian",
):
    spot_coords_crop = spot_coords_full.copy().astype(float)
    spot_coords_crop[:, 0] -= float(col_offset)
    spot_coords_crop[:, 1] -= float(row_offset)

    tree = cKDTree(patch_centers)
    dists, idxs = tree.query(spot_coords_crop, k=max(1, int(k_neigh)))

    if np.ndim(dists) == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    if weight_mode == "uniform":
        weights = np.ones_like(dists)
    else:
        median_nn = np.median(dists[:, 0])
        sigma = median_nn if median_nn > 0 else 1.0
        weights = np.exp(-0.5 * (dists / sigma) ** 2)

    mapped = np.sum(patch_preds[idxs] * weights[:, :, None], axis=1) / np.sum(
        weights[:, :, None], axis=1
    )
    nearest = dists[:, 0] if dists.ndim == 2 else dists
    return mapped, nearest


def calc_metrics(pred: np.ndarray, true: np.ndarray):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    pred = pred[mask]
    true = true[mask]
    pcc = pearsonr(pred, true)[0]
    mse = float(np.mean((pred - true) ** 2))
    p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    t = (true - true.min()) / (true.max() - true.min() + 1e-8)
    emd = wasserstein_distance(p, t)
    return float(pcc), float(emd), float(mse)


def compute_patch_predictions_sliding(
    feats: np.ndarray,
    coords: np.ndarray,
    gene_indices: List[int],
    model_path: str,
    model_type: str,
    device: str,
    patch_size: int,
    coord_type: str,
    window_size: int,
    stride: int,
    min_coverage: float,
):
    num_tiles = feats.shape[0]
    feat_dim = feats.shape[1]

    model, input_dim, num_outputs = load_model(
        model_path, model_type, device, num_clusters=window_size * window_size
    )

    if feat_dim != input_dim:
        raise ValueError(
            f"Feature dim {feat_dim} does not match model input dim {input_dim}"
        )

    patch_centers = coords.astype(float).copy()
    if coord_type == "top_left":
        half = patch_size / 2.0
        patch_centers[:, 0] = patch_centers[:, 0] + half
        patch_centers[:, 1] = patch_centers[:, 1] + half

    min_x = patch_centers[:, 0].min()
    min_y = patch_centers[:, 1].min()
    x_tf = np.rint((patch_centers[:, 0] - min_x) / patch_size).astype(int)
    y_tf = np.rint((patch_centers[:, 1] - min_y) / patch_size).astype(int)

    max_x = int(x_tf.max())
    max_y = int(y_tf.max())

    pred_sum = np.zeros((num_tiles, len(gene_indices)), dtype=np.float64)
    pred_count = np.zeros(num_tiles, dtype=np.int32)

    coords_tf = np.column_stack([x_tf, y_tf])

    for x in range(0, max_x + 1, stride):
        for y in range(0, max_y + 1, stride):
            in_window = (
                (coords_tf[:, 0] >= x)
                & (coords_tf[:, 0] < x + window_size)
                & (coords_tf[:, 1] >= y)
                & (coords_tf[:, 1] < y + window_size)
            )
            idxs = np.where(in_window)[0]
            if len(idxs) < int(window_size * window_size * min_coverage):
                continue

            xy = coords_tf[idxs]
            order = np.lexsort((xy[:, 0], xy[:, 1]))
            idxs = idxs[order]

            features_all = feats[idxs]
            target_tokens = window_size * window_size
            if features_all.shape[0] > target_tokens:
                features_all = features_all[:target_tokens]
            elif features_all.shape[0] < target_tokens:
                pad = np.zeros((target_tokens - features_all.shape[0], feat_dim))
                features_all = np.concatenate([features_all, pad], axis=0)

            with torch.no_grad():
                x_t = torch.tensor(features_all).float().unsqueeze(0).to(device)
                preds = model(x_t).detach().cpu().numpy()[0]
                pred_vals = preds[gene_indices]

            pred_sum[idxs] += pred_vals
            pred_count[idxs] += 1

    pred_vals = np.full((num_tiles, len(gene_indices)), np.nan, dtype=np.float64)
    mask = pred_count > 0
    pred_vals[mask] = pred_sum[mask] / pred_count[mask, None]
    return pred_vals, patch_centers


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sliding-window SEQUOIA metrics for a list of genes."
    )
    parser.add_argument("--base_dir", required=True, help="Visium base directory")
    parser.add_argument("--count_file", required=True, help="Visium count file name")
    parser.add_argument("--genes_csv", required=True, help="CSV with genes to evaluate")
    parser.add_argument("--h5_path", required=True, help="H5 with patch features and coords")
    parser.add_argument("--model_type", choices=["vis", "vit"], default="vis")
    parser.add_argument("--model_path", required=True, help="Model checkpoint path")
    parser.add_argument("--genes_pkl", default=None, help="Optional model gene list pickle")
    parser.add_argument("--gene_list", default=None, help="Model gene list csv/npy")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--offsets_json", default=None, help="Crop offsets json")
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min_coverage", type=float, default=0.5)
    parser.add_argument("--k_neigh", type=int, default=3)
    parser.add_argument("--weight_mode", choices=["uniform", "gaussian"], default="gaussian")
    parser.add_argument("--smooth_st", action="store_true")
    parser.add_argument("--smooth_pred", action="store_true")
    parser.add_argument("--smooth_k", type=int, default=6)
    parser.add_argument("--smooth_sigma", type=float, default=1.2)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_sequoia_on_path(os.environ.get("SEQUOIA_ROOT"))

    os.makedirs(args.out_dir, exist_ok=True)

    adata = sc.read_visium(
        path=args.base_dir,
        count_file=args.count_file,
        source_image_path=os.path.join(args.base_dir, "spatial"),
        load_images=False,
    )
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if "spatial" in adata.obsm:
        spot_coords_full = adata.obsm["spatial"].astype(float)
    else:
        pos_path = load_positions_csv(args.base_dir)
        if pos_path is None:
            raise ValueError("Cannot find tissue_positions_list.csv or tissue_positions.csv")
        print(f"[WARN] adata.obsm['spatial'] missing, loading coords from {pos_path}")
        spot_coords_full = coords_from_positions_csv(pos_path, adata.obs_names)

    true_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    st_gene_names = list(adata.var_names)

    feats, coords, attrs = load_h5(args.h5_path)

    patch_size = int(attrs.get("patch_size", args.patch_size))
    row_offset = int(attrs.get("row_offset", 0))
    col_offset = int(attrs.get("col_offset", 0))

    if args.offsets_json:
        with open(args.offsets_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        row_offset = int(meta.get("row_offset", meta.get("min_y", row_offset)))
        col_offset = int(meta.get("col_offset", meta.get("min_x", col_offset)))

    coord_type = attrs.get("coord_type", None)
    if isinstance(coord_type, bytes):
        coord_type = coord_type.decode("utf-8", errors="ignore")
    if coord_type is None:
        coord_type = "center" if (row_offset != 0 or col_offset != 0) else "top_left"

    genes_model = load_gene_list(args.genes_pkl, args.gene_list)
    genes_target = load_genes_csv(args.genes_csv)

    st_gene_to_idx = {g: i for i, g in enumerate(st_gene_names)}
    model_gene_to_idx = {g: i for i, g in enumerate(genes_model)}

    common_genes = [g for g in genes_target if g in st_gene_to_idx and g in model_gene_to_idx]
    if not common_genes:
        raise ValueError("No overlap between genes_csv, model genes, and ST genes")

    gene_indices = [model_gene_to_idx[g] for g in common_genes]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_patch_vals, patch_centers = compute_patch_predictions_sliding(
        feats=feats,
        coords=coords,
        gene_indices=gene_indices,
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        patch_size=patch_size,
        coord_type=coord_type,
        window_size=args.window_size,
        stride=args.stride,
        min_coverage=args.min_coverage,
    )

    pred_spot_mat, nn_d0 = map_patch_preds_to_spots(
        patch_centers=patch_centers,
        patch_preds=pred_patch_vals,
        spot_coords_full=spot_coords_full,
        row_offset=row_offset,
        col_offset=col_offset,
        k_neigh=args.k_neigh,
        weight_mode=args.weight_mode,
    )

    true_idx = [st_gene_to_idx[g] for g in common_genes]
    true_eval = true_mat[:, true_idx].copy()
    pred_eval = pred_spot_mat.copy()

    if args.smooth_pred:
        for j in range(pred_eval.shape[1]):
            pred_eval[:, j] = smooth_expression_by_neighbors(
                spot_coords_full, pred_eval[:, j], k=args.smooth_k, sigma=args.smooth_sigma
            )
    if args.smooth_st:
        for j in range(true_eval.shape[1]):
            true_eval[:, j] = smooth_expression_by_neighbors(
                spot_coords_full, true_eval[:, j], k=args.smooth_k, sigma=args.smooth_sigma
            )

    rows = []
    for j, g in enumerate(common_genes):
        pcc, emd, mse = calc_metrics(pred_eval[:, j], true_eval[:, j])
        rows.append({"gene": g, "pcc": pcc, "emd": emd, "mse": mse})

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.out_dir, "spot_level_metrics_all_common_genes.csv")
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "n_common_genes": int(len(common_genes)),
        "pcc_mean": float(np.nanmean(metrics_df["pcc"])),
        "pcc_median": float(np.nanmedian(metrics_df["pcc"])),
        "emd_mean": float(np.nanmean(metrics_df["emd"])),
        "emd_median": float(np.nanmedian(metrics_df["emd"])),
        "mse_mean": float(np.nanmean(metrics_df["mse"])),
        "mse_median": float(np.nanmedian(metrics_df["mse"])),
        "spot_to_nearest_center_dist_px_median": float(np.median(nn_d0)),
        "spot_to_nearest_center_dist_px_p95": float(np.percentile(nn_d0, 95)),
        "spot_to_nearest_center_dist_px_max": float(np.max(nn_d0)),
    }

    summary_path = os.path.join(args.out_dir, "spot_level_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Wrote metrics: {metrics_path}")
    print(f"[DONE] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
