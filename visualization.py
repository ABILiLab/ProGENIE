import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


def load_h5_features(path: str):
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


class FeatureHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        tanh_out = torch.tanh(self.linear1(x))
        sigmoid_out = torch.sigmoid(self.linear2(x))
        return torch.cat([tanh_out, sigmoid_out], dim=-1)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x, return_weights=False):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        if return_weights:
            return pooled, weights
        return pooled


class GeneExpressionModelHead(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        output_dim=16059,
        num_heads=6,
        num_tokens=100,
    ):
        super().__init__()
        self.pos_emb1D = nn.Parameter(torch.randn(num_tokens, input_dim))
        self.num_heads = num_heads
        self.feature_heads = nn.ModuleList(
            [FeatureHead(input_dim, hidden_dim) for _ in range(num_heads)]
        )
        self.attn_pooling_heads = nn.ModuleList(
            [AttentionPooling(input_dim) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * input_dim, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_heads, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        if x.size(1) != self.pos_emb1D.size(0):
            raise ValueError(
                f"Input token length {x.size(1)} != pos_emb tokens {self.pos_emb1D.size(0)}"
            )
        x = x + self.pos_emb1D.to(x.device).unsqueeze(0)

        pooled_list = []
        for head, attn in zip(self.feature_heads, self.attn_pooling_heads):
            features = head(x)
            pooled = attn(features)
            pooled_list.append(pooled)
        pooled_cat = torch.cat(pooled_list, dim=1)
        return self.mlp(pooled_cat)


def load_gene_names_from_pt(path: str) -> List[str]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "gene_names" in obj:
        return list(obj["gene_names"])
    raise ValueError(f"Cannot find 'gene_names' in {path}")


def load_visium_adata(base_dir: str, count_file: str):
    adata = sc.read_visium(
        path=base_dir,
        count_file=count_file,
        source_image_path=os.path.join(base_dir, "spatial"),
        load_images=False,
    )
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


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


def resolve_patch_centers(coords: np.ndarray, attrs: Dict, patch_size_arg: int):
    patch_size = int(attrs.get("patch_size", patch_size_arg))
    coord_type = attrs.get("coord_type", "center")
    if isinstance(coord_type, bytes):
        coord_type = coord_type.decode("utf-8", errors="ignore")
    centers = coords.astype(float).copy()
    if coord_type == "top_left":
        half = patch_size / 2.0
        centers[:, 0] += half
        centers[:, 1] += half
    return centers, patch_size


def infer_grid_step(patch_centers: np.ndarray) -> float:
    tree = cKDTree(patch_centers)
    dists, _ = tree.query(patch_centers, k=2)
    step = float(np.median(dists[:, 1]))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Failed to infer grid step from patch centers.")
    return step


def build_grid_index(patch_centers: np.ndarray, step: float):
    min_x = float(np.min(patch_centers[:, 0]))
    min_y = float(np.min(patch_centers[:, 1]))
    gx = np.rint((patch_centers[:, 0] - min_x) / step).astype(int)
    gy = np.rint((patch_centers[:, 1] - min_y) / step).astype(int)
    grid_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(gx, gy))}
    return gx, gy, grid_to_idx


def make_sliding_windows(
    feats: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    grid_to_idx: Dict[Tuple[int, int], int],
    window_size: int = 10,
):
    half_left = window_size // 2
    half_right = window_size - half_left - 1
    center_indices = []
    window_indices = []

    for center_idx, (cx, cy) in enumerate(zip(gx, gy)):
        idxs = []
        complete = True
        for yy in range(cy - half_left, cy + half_right + 1):
            for xx in range(cx - half_left, cx + half_right + 1):
                j = grid_to_idx.get((xx, yy))
                if j is None:
                    complete = False
                    break
                idxs.append(j)
            if not complete:
                break
        if complete and len(idxs) == window_size * window_size:
            center_indices.append(center_idx)
            window_indices.append(idxs)

    if not window_indices:
        raise ValueError("No complete sliding windows found. Check grid/step/window_size.")

    window_feats = feats[np.array(window_indices, dtype=int)]
    return np.array(center_indices, dtype=int), window_feats


def predict_window_centers(
    model: nn.Module,
    window_feats: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, window_feats.shape[0], batch_size):
            batch = window_feats[start : start + batch_size]
            bt = torch.from_numpy(batch).float().to(device)
            pred = model(bt).cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def smooth_expression_by_neighbors(coords: np.ndarray, values: np.ndarray, k=6, sigma=1.2):
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


def calc_pcc_emd(pred: np.ndarray, true: np.ndarray):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 3:
        return np.nan, np.nan
    pcc = pearsonr(pred[mask], true[mask])[0]
    p = pred[mask]
    t = true[mask]
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    emd = wasserstein_distance(p, t)
    return float(pcc), float(emd)


def normalize_pair(a: np.ndarray, b: np.ndarray, vmin_pct=5.0, vmax_pct=95.0):
    combined = np.concatenate([a, b])
    vmin = np.percentile(combined, vmin_pct)
    vmax = np.percentile(combined, vmax_pct)
    return np.clip(a, vmin, vmax), np.clip(b, vmin, vmax), float(vmin), float(vmax)


def quantile_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return source
    order = np.argsort(source)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=len(source), endpoint=True)
    target_sorted = np.sort(target)
    return np.interp(
        ranks,
        np.linspace(0.0, 1.0, num=len(target_sorted), endpoint=True),
        target_sorted,
    )


def plot_spots(coords, values, vmin, vmax, cmap, spot_size_scale, ax, title):
    x = coords[:, 0]
    y = coords[:, 1]
    tree = cKDTree(np.column_stack([x, y]))
    dists, _ = tree.query(np.column_stack([x, y]), k=2)
    median_nn = np.median(dists[:, 1])
    radius = median_nn * spot_size_scale

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(values))

    patches = [Circle((xi, yi), radius=radius) for xi, yi in zip(x, y)]
    col = PatchCollection(patches, facecolors=colors, edgecolors="none", linewidths=0, zorder=2)
    ax.add_collection(col)
    margin = median_nn
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.max() + margin, y.min() - margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)


def try_compute_svg_top100(adata):
    try:
        import squidpy as sq
    except Exception:
        return None, "squidpy not installed; skip SVG selection"

    ad = adata.copy()
    if "spatial" not in ad.obsm:
        return None, "adata.obsm['spatial'] missing; skip SVG selection"

    sq.gr.spatial_neighbors(ad, coord_type="generic")
    sq.gr.spatial_autocorr(ad, mode="moran", genes=ad.var_names)

    moran = ad.uns["moranI"].copy()
    p_col = None
    for c in ["pval_norm_fdr_bh", "pval_norm", "pval"]:
        if c in moran.columns:
            p_col = c
            break
    if p_col is None:
        return None, "No Moran's I p-value column found; skip SVG selection"

    sig = moran[moran[p_col] < 0.05].sort_values("I", ascending=False)
    top100 = sig.head(100).index.tolist()
    return top100, f"SVG selected by {p_col}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spatial visualization and spot-level metrics with sliding windows."
    )
    parser.add_argument(
        "--base_dir",
        default=None,
    )
    parser.add_argument(
        "--count_file",
        default=None,
    )
    parser.add_argument(
        "--h5_path",
        default=None,
    )
    parser.add_argument(
        "--model_path",
        default=None,
    )
    parser.add_argument(
        "--gene_names_pt",
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        default=None,
    )
    parser.add_argument(
        "--offsets_json",
        default=None,
    )
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--window_batch_size", type=int, default=128)
    parser.add_argument("--k_neigh", type=int, default=3)
    parser.add_argument("--weight_mode", choices=["uniform", "gaussian"], default="gaussian")
    parser.add_argument("--smooth_st", action="store_true", default=True)
    parser.add_argument("--smooth_pred", action="store_true", default=True)
    parser.add_argument("--smooth_k", type=int, default=6)
    parser.add_argument("--smooth_sigma", type=float, default=1.2)
    parser.add_argument("--sequoia_pred_csv", default=None)
    parser.add_argument("--gene", default="ACTA2", help="Optional single gene for map visualization.")
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--spot_size_scale", type=float, default=0.45)
    parser.add_argument("--vmin_pct", type=float, default=5.0)
    parser.add_argument("--vmax_pct", type=float, default=95.0)
    parser.add_argument(
        "--color_mode",
        choices=["shared", "true_range", "quantile_match"],
        default="quantile_match",
        help="Color normalization mode for single-gene visualization.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading Visium and preprocessing with Scanpy")
    adata = load_visium_adata(args.base_dir, args.count_file)
    if "spatial" in adata.obsm:
        spot_coords_full = adata.obsm["spatial"].astype(float)
    else:
        pos_path = load_positions_csv(args.base_dir)
        if pos_path is None:
            raise ValueError(
                "adata.obsm['spatial'] missing and no tissue_positions_list.csv found."
            )
        print(f"[WARN] adata.obsm['spatial'] missing, loading coords from {pos_path}")
        spot_coords_full = coords_from_positions_csv(pos_path, list(adata.obs_names))
    true_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    st_gene_names = list(adata.var_names)

    print("[INFO] Loading patch features")
    feats, coords, attrs = load_h5_features(args.h5_path)
    patch_centers, patch_size = resolve_patch_centers(coords, attrs, args.patch_size)
    print(f"[INFO] patch_size={patch_size}, n_patch={len(patch_centers)}")

    row_offset = int(attrs.get("row_offset", 0))
    col_offset = int(attrs.get("col_offset", 0))
    if args.offsets_json:
        with open(args.offsets_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        row_offset = int(meta.get("row_offset", meta.get("min_y", row_offset)))
        col_offset = int(meta.get("col_offset", meta.get("min_x", col_offset)))
    print(f"[INFO] row_offset={row_offset}, col_offset={col_offset}")

    print("[INFO] Building sliding windows (stride=1)")
    step = infer_grid_step(patch_centers)
    gx, gy, grid_to_idx = build_grid_index(patch_centers, step)
    center_idx, window_feats = make_sliding_windows(
        feats, gx, gy, grid_to_idx, window_size=args.window_size
    )
    print(
        f"[INFO] grid_step~{step:.3f} px, complete_windows={len(center_idx)}, "
        f"window_shape={window_feats.shape}"
    )

    gene_names = load_gene_names_from_pt(args.gene_names_pt)
    model = GeneExpressionModelHead(
        input_dim=feats.shape[1],
        hidden_dim=512,
        output_dim=len(gene_names),
        num_heads=6,
        num_tokens=args.window_size * args.window_size,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    print("[INFO] Predicting center-patch transcriptome from 10x10 windows")
    center_preds = predict_window_centers(
        model, window_feats, device=device, batch_size=args.window_batch_size
    )

    center_coords = patch_centers[center_idx]
    np.save(os.path.join(args.out_dir, "center_patch_coords.npy"), center_coords)
    np.save(os.path.join(args.out_dir, "center_patch_preds.npy"), center_preds)
    pd.Series(gene_names, name="gene").to_csv(
        os.path.join(args.out_dir, "pred_gene_names.csv"), index=False
    )

    print("[INFO] Mapping patch-center predictions to spot level (kNN + Gaussian)")
    pred_spot_mat, nn_d0 = map_patch_preds_to_spots(
        patch_centers=center_coords,
        patch_preds=center_preds,
        spot_coords_full=spot_coords_full,
        row_offset=row_offset,
        col_offset=col_offset,
        k_neigh=args.k_neigh,
        weight_mode=args.weight_mode,
    )
    print(
        "[DEBUG] spot->nearest-center distance (px): median=%.2f, p95=%.2f, max=%.2f"
        % (float(np.median(nn_d0)), float(np.percentile(nn_d0, 95)), float(np.max(nn_d0)))
    )

    st_gene_to_idx = {g: i for i, g in enumerate(st_gene_names)}
    common_genes = [g for g in gene_names if g in st_gene_to_idx]
    pred_idx = [gene_names.index(g) for g in common_genes]
    true_idx = [st_gene_to_idx[g] for g in common_genes]

    pred_eval = pred_spot_mat[:, pred_idx].copy()
    true_eval = true_mat[:, true_idx].copy()

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
        pcc, emd = calc_pcc_emd(pred_eval[:, j], true_eval[:, j])
        rows.append({"gene": g, "pcc": pcc, "emd": emd})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(args.out_dir, "spot_level_metrics_all_common_genes.csv"), index=False)

    top100_svg, svg_note = try_compute_svg_top100(adata)
    with open(os.path.join(args.out_dir, "svg_note.txt"), "w", encoding="utf-8") as f:
        f.write(svg_note + "\n")
    if top100_svg is not None:
        top100_overlap = [g for g in top100_svg if g in set(common_genes)]
        metrics_top100 = metrics_df[metrics_df["gene"].isin(top100_overlap)].copy()
        metrics_top100.to_csv(
            os.path.join(args.out_dir, "spot_level_metrics_svg_top100_overlap.csv"),
            index=False,
        )

    if args.gene is not None:
        if args.gene not in common_genes:
            raise ValueError(
                f"--gene {args.gene} not found in common genes. "
                f"Example available gene: {common_genes[0] if common_genes else 'None'}"
            )
        gj = common_genes.index(args.gene)
        pred_g = pred_eval[:, gj]
        true_g = true_eval[:, gj]
        mask = np.isfinite(pred_g) & np.isfinite(true_g)
        pred_g = pred_g[mask]
        true_g = true_g[mask]
        coords_g = spot_coords_full[mask]
        pcc_g, emd_g = calc_pcc_emd(pred_g, true_g)

        if args.color_mode == "quantile_match":
            pred_plot = quantile_match(pred_g, true_g)
            vmin = float(np.percentile(true_g, args.vmin_pct))
            vmax = float(np.percentile(true_g, args.vmax_pct))
            pred_clip = np.clip(pred_plot, vmin, vmax)
            true_clip = np.clip(true_g, vmin, vmax)
        elif args.color_mode == "true_range":
            vmin = float(np.percentile(true_g, args.vmin_pct))
            vmax = float(np.percentile(true_g, args.vmax_pct))
            pred_clip = np.clip(pred_g, vmin, vmax)
            true_clip = np.clip(true_g, vmin, vmax)
        else:
            pred_clip, true_clip, vmin, vmax = normalize_pair(
                pred_g, true_g, vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct
            )

        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
        fig.subplots_adjust(left=0.04, right=0.84, top=0.88, bottom=0.08, wspace=0.10)

        plot_spots(
            coords_g,
            pred_clip,
            vmin,
            vmax,
            args.cmap,
            args.spot_size_scale,
            axes[0],
            f"{args.gene}_pred",
        )
        plot_spots(
            coords_g,
            true_clip,
            vmin,
            vmax,
            args.cmap,
            args.spot_size_scale,
            axes[1],
            f"{args.gene}_true",
        )

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
        sm.set_array([])
        cax = fig.add_axes([0.89, 0.22, 0.018, 0.52])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=9)

        fig.suptitle(f"{args.gene} | PCC: {pcc_g:.3f}, EMD: {emd_g:.3f}", fontsize=13)
        out_png = os.path.join(args.out_dir, f"{args.gene}_pred_vs_true_spatial.png")
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

        with open(
            os.path.join(args.out_dir, f"{args.gene}_spot_level_metrics.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"gene={args.gene}\n")
            f.write(f"pcc={pcc_g:.6f}\n")
            f.write(f"emd={emd_g:.6f}\n")

    if args.sequoia_pred_csv:
        seq_df = pd.read_csv(args.sequoia_pred_csv, index_col=0)
        common_spots = [s for s in adata.obs_names if s in seq_df.index]
        seq_common_genes = [g for g in common_genes if g in seq_df.columns]
        if common_spots and seq_common_genes:
            pred_seq = seq_df.loc[common_spots, seq_common_genes].to_numpy(float)
            true_seq = pd.DataFrame(
                true_mat, index=adata.obs_names, columns=st_gene_names
            ).loc[common_spots, seq_common_genes].to_numpy(float)
            if args.smooth_pred:
                coords_seq = pd.DataFrame(spot_coords_full, index=adata.obs_names).loc[
                    common_spots
                ].to_numpy(float)
                for j in range(pred_seq.shape[1]):
                    pred_seq[:, j] = smooth_expression_by_neighbors(
                        coords_seq, pred_seq[:, j], k=args.smooth_k, sigma=args.smooth_sigma
                    )
            if args.smooth_st:
                coords_seq = pd.DataFrame(spot_coords_full, index=adata.obs_names).loc[
                    common_spots
                ].to_numpy(float)
                for j in range(true_seq.shape[1]):
                    true_seq[:, j] = smooth_expression_by_neighbors(
                        coords_seq, true_seq[:, j], k=args.smooth_k, sigma=args.smooth_sigma
                    )
            rows = []
            for j, g in enumerate(seq_common_genes):
                pcc, emd = calc_pcc_emd(pred_seq[:, j], true_seq[:, j])
                rows.append({"gene": g, "pcc": pcc, "emd": emd})
            pd.DataFrame(rows).to_csv(
                os.path.join(args.out_dir, "sequoia_spot_level_metrics_all_common_genes.csv"),
                index=False,
            )

    summary = {
        "n_spots": int(spot_coords_full.shape[0]),
        "n_input_patches": int(feats.shape[0]),
        "n_center_patches_predicted": int(center_preds.shape[0]),
        "n_pred_genes": int(len(gene_names)),
        "n_common_genes_with_st": int(len(common_genes)),
        "window_size": int(args.window_size),
        "stride": 1,
        "k_neigh": int(args.k_neigh),
        "weight_mode": args.weight_mode,
        "smooth_st": bool(args.smooth_st),
        "smooth_pred": bool(args.smooth_pred),
        "smooth_k": int(args.smooth_k),
        "smooth_sigma": float(args.smooth_sigma),
        "spot_to_nearest_center_dist_px_median": float(np.median(nn_d0)),
        "spot_to_nearest_center_dist_px_p95": float(np.percentile(nn_d0, 95)),
        "spot_to_nearest_center_dist_px_max": float(np.max(nn_d0)),
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Outputs written to:", args.out_dir)
    print("[DONE] Common genes evaluated:", len(common_genes))


if __name__ == "__main__":
    main()
