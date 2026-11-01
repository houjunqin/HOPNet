import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

import cupy as cp
from cuml.neighbors import kneighbors_graph as cuml_knn
from torch_geometric.utils import from_scipy_sparse_matrix
from global_land_mask import globe
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


# ----------------------------
# Utils: geo + graph
# ----------------------------
EARTH_R = 6371000.0  # meters

def haversine_m(lonlat1, lonlat2):
    lon1, lat1 = lonlat1
    lon2, lat2 = lonlat2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_R * c

def lonlat_to_unit_xyz(lon_deg, lat_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])

def build_adjacency_from_gdf(gdf, sigma=None, symmetric=True):
    N = len(gdf)
    rows, cols, dists = [], [], []
    for i, row in gdf.iterrows():
        nb_str = row["Neighbors"]
        dist_str = row["Distance"]
        if not isinstance(nb_str, str) or not nb_str.strip():
            continue
        nbs = list(map(int, nb_str.split()))
        ds = list(map(float, dist_str.split()))
        for j, d in zip(nbs, ds):
            if j < 0 or j >= N or j == i:
                continue
            rows.append(i)
            cols.append(j)
            dists.append(d)

    if len(dists) == 0:
        raise ValueError("No edges found. Check gdf['Neighbors'] / gdf['Distance'].")

    if sigma is None:
        sigma = float(np.median(dists))

    w = np.exp(-(np.array(dists, dtype=float) ** 2) / (sigma ** 2))
    A = sp.coo_matrix((w, (rows, cols)), shape=(N, N)).tocsr()
    if symmetric:
        A = 0.5 * (A + A.T)
    return A, sigma

def add_self_loops(A):
    return A + sp.eye(A.shape[0], format="csr")

def normalize_gcn(A):
    # D^{-1/2} A D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    inv_sqrt = np.power(deg, -0.5, where=deg>0, out=None)
    inv_sqrt[deg == 0] = 0.0
    Dinv = sp.diags(inv_sqrt)
    return (Dinv @ A @ Dinv).tocsr()

def normalize_random_walk(A):
    # D^{-1} A
    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.clip(deg, a_min=1e-300, a_max=None)
    inv = np.power(deg_safe, -1.0, where=deg>0, out=None)
    inv[deg == 0] = 0.0
    Dinv = sp.diags(inv)
    return (Dinv @ A).tocsr()

def dual_random_walk(A):
    P_fwd = normalize_random_walk(A)
    P_bwd = normalize_random_walk(A.T.tocsr())
    return P_fwd, P_bwd

def sparse_to_pyg(A_csr):
    A_coo = A_csr.tocoo()
    edge_index = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
    edge_attr = torch.tensor(A_coo.data, dtype=torch.float32)
    return edge_index, edge_attr


# ----------------------------
# Sampler: strict equal group size
# ----------------------------
@dataclass
class EvenFixedGroupSampler:
    """
    STRICT: every group has exactly group_size nodes.
    If N % group_size != 0, duplicates across groups are unavoidable (wrap padding).
    """
    N: int
    group_size: int
    #seed: int | None = None
    even_stride: bool = True

    def __post_init__(self):
        if not (1 <= self.group_size <= self.N):
            raise ValueError("group_size must be in [1, N].")
        #self.rng = np.random.default_rng(self.seed)
        self._build_cycle()

    def _build_cycle(self):
        
        """         N, m = self.N, self.group_size
        #perm = self.rng.permutation(N)
        seq = np.arange(N)
        G = int(np.ceil(N / m))

        groups = []
        if self.even_stride:
            order = []
            for offset in range(G):
                #order.extend(perm[offset::G].tolist())
                order.extend(seq[offset::G].tolist())
            order = np.array(order, dtype=np.int64)

            for g in range(G):
                start = g * m
                end = start + m
                if end <= N:
                    idx = order[start:end]
                else:
                    idx = np.concatenate([order[start:N], order[0:(end - N)]], axis=0)
                #groups.append(self.rng.permutation(idx))
                groups.append(idx)
        else:
            for g in range(G):
                start = g * m
                end = start + m
                if end <= N:
                    #idx = perm[start:end]
                    idx = seq[start:end]
                else:
                    #idx = np.concatenate([perm[start:N], perm[0:(end - N)]], axis=0)
                    idx = np.concatenate([seq[start:N], seq[0:(end - N)]], axis=0)
                #groups.append(self.rng.permutation(idx))
                groups.append(idx) """
                
        N, m = self.N, self.group_size
        seq = np.arange(N)
        G = N // m
        groups = []

        if self.even_stride:
            # Option 2: Clean stride intervals per group
            for offset in range(G):
                # Grab every G-th element and cut it off at exactly group_size
                idx = seq[offset::G][:m]
                groups.append(idx)
        else:
            # Option 1: Clean continuous sequential blocks
            for g in range(G):
                start = g * m
                end = start + m
                idx = seq[start:end]
                groups.append(idx)

        self.groups = groups
        self.ptr = 0
        self.G = G

    def rounds_per_cycle(self) -> int:
        return self.G

    def next_indices(self) -> np.ndarray:
        if self.ptr >= self.G:
            self._build_cycle()
        idx = self.groups[self.ptr]
        self.ptr += 1
        return idx

    def apply(self, X: np.ndarray, NF: np.ndarray, node_id: np.ndarray):
        idx = self.next_indices()
        return X[:, idx, :], NF[idx, :], node_id[idx], idx


# ----------------------------
# Split + scaler + loader
# ----------------------------
IndexLike = Union[slice, int, Iterable[int], np.ndarray]

def split_nd_with_feature_select(
    data: np.ndarray,
    input_feature_idx: IndexLike = (0, 2, 4, 5),
    split_ratio: Tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError(f"data must be at least 2D (T,...,F). Got {data.shape}")

    F = data.shape[-1]
    idx_arr = np.asarray(input_feature_idx)
    if idx_arr.dtype.kind in "iu":
        if np.any(idx_arr < -F) or np.any(idx_arr >= F):
            raise IndexError(f"feature_idx {list(input_feature_idx)} out of bounds for last dim F={F}")

    X = np.take(data, indices=input_feature_idx, axis=-1).astype(np.float32, copy=False)

    r = np.array(split_ratio, dtype=np.float64)
    if r.shape != (3,) or np.any(r < 0):
        raise ValueError("split_ratio must be 3 non-negative numbers (train,val,test)")
    if not np.isclose(r.sum(), 1.0):
        r = r / r.sum()

    T = X.shape[0]
    n_train = int(np.floor(T * r[0]))
    n_val   = int(np.floor(T * r[1]))
    n_test  = T - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Bad split sizes: train={n_train}, val={n_val}, test={n_test} for first-dim={T}")

    return X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]



ArrayLike = Union[np.ndarray, torch.Tensor]

class ZScoreScaler:
    """
    Z-score scaler using TRAIN stats only.
    Works with:
      - numpy arrays
      - torch tensors (CPU or CUDA)

    Feature dim = last dim.

    Notes:
      - Stats are stored both as numpy (mean_np/std_np) and torch (mean_t/std_t).
      - When transforming, we keep computation in the same backend/device as input.
    """
    def __init__(self, mean: ArrayLike, std: ArrayLike, eps: float = 1e-8):
        self.eps = float(eps)

        # Store numpy copy (for numpy inputs)
        if isinstance(mean, torch.Tensor):
            mean_np = mean.detach().float().cpu().numpy()
        else:
            mean_np = np.asarray(mean, dtype=np.float32)

        if isinstance(std, torch.Tensor):
            std_np = std.detach().float().cpu().numpy()
        else:
            std_np = np.asarray(std, dtype=np.float32)

        std_np = np.where(std_np < self.eps, 1.0, std_np).astype(np.float32)

        self.mean_np = mean_np
        self.std_np = std_np

        # Store torch copy (for torch inputs) — created lazily per device/dtype
        self._mean_t: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._std_t: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _reduce_dims(x: torch.Tensor):
        # reduce over all dims except last
        return tuple(range(x.ndim - 1))

    def _torch_stats(self, device: torch.device, dtype: torch.dtype):
        key = (device, dtype)
        if key not in self._mean_t:
            mean_t = torch.tensor(self.mean_np, device=device, dtype=dtype)
            std_t  = torch.tensor(self.std_np, device=device, dtype=dtype)
            self._mean_t[key] = mean_t
            self._std_t[key] = std_t
        return self._mean_t[key], self._std_t[key]

    @classmethod
    def fit(cls, x_train: ArrayLike, eps: float = 1e-8):
        # torch path (CPU or CUDA)
        if isinstance(x_train, torch.Tensor):
            if x_train.ndim < 1:
                raise ValueError("x_train must have at least 1 dim")
            rd = cls._reduce_dims(x_train)
            mean = x_train.mean(dim=rd)
            std = x_train.std(dim=rd, unbiased=False)  # match numpy default behavior
            # clamp small std
            std = torch.where(std < eps, torch.ones_like(std), std)
            return cls(mean, std, eps=eps)

        # numpy path
        x_train = np.asarray(x_train, dtype=np.float32)
        if x_train.ndim < 1:
            raise ValueError("x_train must have at least 1 dim")
        K = x_train.shape[-1]
        flat = x_train.reshape(-1, K)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return cls(mean, std, eps=eps)

    def transform(self, x: ArrayLike) -> ArrayLike:
        # torch path
        if isinstance(x, torch.Tensor):
            mean_t, std_t = self._torch_stats(x.device, x.dtype)
            return (x - mean_t) / std_t

        # numpy path
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_np) / self.std_np

    def inverse_transform(self, x_scaled: ArrayLike) -> ArrayLike:
        # torch path
        if isinstance(x_scaled, torch.Tensor):
            mean_t, std_t = self._torch_stats(x_scaled.device, x_scaled.dtype)
            return x_scaled * std_t + mean_t

        # numpy path
        x_scaled = np.asarray(x_scaled, dtype=np.float32)
        return x_scaled * self.std_np + self.mean_np



class TimeSeriesWindowDataset(Dataset):
    """
    Build sliding-window (X, Y) pairs from a time series array.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N, F)
    input_len : int
        Length of input window
    horizon : int
        Length of output window
    return_index : bool
        If True, __getitem__ returns (X, Y, idx)
    permute : Optional[Tuple[int, ...]]
        Optional permutation applied to X and Y
        Example:
          (0,1,2) -> no change
          (1,0,2) -> (T,N,F) → (N,T,F)
    """

    def __init__(
        self,
        data: np.ndarray,
        input_len: int = 12,
        horizon: int = 12,
        return_index: bool = False,
        permute: Optional[Tuple[int, ...]] = None,
    ):
        if data.ndim != 3:
            raise ValueError(f"data must be (T,N,F). Got {data.shape}")

        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.return_index = bool(return_index)
        self.permute = permute

        T = self.data.shape[0]
        self.num_samples = T - self.input_len - self.horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough timesteps: T={T}, input_len={input_len}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def _maybe_permute(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute is None:
            return x
        return x.permute(*self.permute).contiguous()

    def __getitem__(self, idx: int):
        x0 = idx
        x1 = idx + self.input_len
        y0 = x1
        y1 = y0 + self.horizon

        X = self.data[x0:x1]  # (input_len, N, F)
        Y = self.data[y0:y1]  # (horizon, N, F)

        X = self._maybe_permute(X)
        Y = self._maybe_permute(Y)

        if self.return_index:
            return X, Y, idx
        return X, Y

def make_loader(
    data_np: np.ndarray,
    input_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    return_index: bool = False,
    permute: Optional[Tuple[int, ...]] = None,
    drop_last: bool = True,
):
    """
    Create Dataset + DataLoader for time-series windows.
    """

    dataset = TimeSeriesWindowDataset(
        data=data_np,
        input_len=input_len,
        horizon=horizon,
        return_index=return_index,
        permute=permute,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataset, loader

class CoastalGraph:
    def __init__(
        self, 
        pos: torch.Tensor, 
        static_features: torch.Tensor, 
        device: torch.device, 
        k: int = 15, 
        factor_x: int = 2, 
        factor_y: int = 4
    ):
        """
        Hierarchical Coastal Graph for Hydrodynamic GNNs.
        
        Args:
            pos: [N, 2] tensor (Longitude, Latitude)
            static_features: [N, 2] tensor (Bathymetry, Manning's n)
            device: torch.device (CUDA)
            k: Number of neighbors for fine-mesh KNN
            factor_x/y: Quantization factors for hierarchical partitioning
        """
        self.device = device
        self.pos = pos.to(device).float()
        self.static_node = static_features.to(device).float()
        N = self.pos.shape[0]

        # 1. Open Boundary Mask (Global Land Mask check on CPU, then move to GPU)
        # globe.is_land requires (Latitude, Longitude)
        pos_cpu = pos.cpu().numpy()
        lons, lats = pos_cpu[:, 0], pos_cpu[:, 1]
        self.open_boundary_mask = torch.from_numpy(~globe.is_land(lats, lons)).to(device)

        # 2. Fine Level KNN (Using cuML on GPU)
        pos_cp = cp.asarray(self.pos)
        # Metric 'haversine' expects radians; convert deg to rad for accurate distances
        knn_res = cuml_knn(cp.deg2rad(pos_cp), n_neighbors=k, mode='distance', metric='haversine')
        
        # Bridge cuML GPU-CSR to PyTorch Geometric via SciPy
        # (Current PyG compatibility requires this CPU hop for sparse conversion)
        knn_scipy = knn_res.get() 
        self.edge_index, dists = from_scipy_sparse_matrix(knn_scipy)
        self.edge_index = self.edge_index.to(device)
        dists = dists.to(device).float()
        
        src, dst = self.edge_index
        delta = self.pos[dst] - self.pos[src]
        self.edge_attr = torch.cat([delta, dists.unsqueeze(-1)], dim=-1)

        # 3. Hierarchical Assignment (Vectorized Quantization on GPU)
        scale_x, scale_y = 1.0 / factor_x, 1.0 / factor_y
        qx = torch.clamp((self.pos[:, 0] * scale_x * N).long(), min=0)
        qy = torch.clamp((self.pos[:, 1] * scale_y * N).long(), min=0)
        
        coarse_grid_width_x = int(N // factor_x) + 1
        unique_ids = qy * coarse_grid_width_x + qx
        _, self.level_assignment = torch.unique(unique_ids, return_inverse=True)
        
        # 4. Coarse Level Construction (Entirely on GPU)
        num_coarse = int(self.level_assignment.max().item()) + 1
        coarse_pos = torch.zeros(num_coarse, 2, device=device)
        counts = torch.zeros(num_coarse, 2, device=device)
        
        coarse_pos.index_add_(0, self.level_assignment, self.pos)
        counts.index_add_(0, self.level_assignment, torch.ones_like(self.pos))
        coarse_pos = coarse_pos / counts.clamp(min=1)

        # Coarse KNN using cuML
        cp_coarse = cp.asarray(coarse_pos)
        k_c = min(15, max(1, num_coarse - 1))
        c_knn_res = cuml_knn(cp_coarse, n_neighbors=k_c, mode='distance', metric='haversine')
        
        c_knn_scipy = c_knn_res.get()
        self.coarse_edge_index, c_dists = from_scipy_sparse_matrix(c_knn_scipy)
        self.coarse_edge_index = self.coarse_edge_index.to(device)
        c_dists = c_dists.to(device).float()
        
        c_src, c_dst = self.coarse_edge_index
        c_delta = coarse_pos[c_dst] - coarse_pos[c_src]
        self.coarse_edge_attr = torch.cat([c_delta, c_dists.unsqueeze(-1)], dim=-1)
        

# ----------------------------
# Main wrapper class
# ----------------------------
class StormSurgeMiniBatchGraphBuilder:
    """
    Build per-node-group datasets:
      - sample fixed-size node groups until all nodes are covered at least once
      - build per-group neighbors + adjacency (BallTree + haversine refinement)
      - feature-select + split
      - global scaler fitted on concatenated train splits
      - attach DataLoaders for each split

    Output:
      datasets[round_id] dict with X_train/X_val/X_test (+ scaled versions) and graph tensors/matrices.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_names: Iterable[str],
        node_feature_names: Iterable[str],
        seed: int = 42,
    ):
        self.cfg = config
        self.feature_names = list(feature_names)
        self.node_feature_names = list(node_feature_names)
        self.seed = seed

    def _build_gdf_with_nn(self, NF_sub: np.ndarray, k_candidates: int) -> gpd.GeoDataFrame:
        df = pd.DataFrame(NF_sub, columns=self.node_feature_names)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"], crs="EPSG:4326"),
        )

        xyz = lonlat_to_unit_xyz(gdf["Longitude"].to_numpy(), gdf["Latitude"].to_numpy())
        tree = BallTree(xyz, metric="euclidean")

        d_euc, idxs = tree.query(xyz, k=min(k_candidates + 1, len(gdf)))

        lons = gdf["Longitude"].to_numpy()
        lats = gdf["Latitude"].to_numpy()

        neighbors_col = []
        distance_col = []

        for i in range(len(gdf)):
            cand = idxs[i].tolist()
            if i in cand:
                cand.remove(i)
            cand = cand[:k_candidates]

            dists = np.array(
                [haversine_m((lons[i], lats[i]), (lons[j], lats[j])) for j in cand],
                dtype=float
            )

            if dists.size == 0:
                neighbors_col.append("")
                distance_col.append("")
                continue

            avg = float(dists.mean())
            mask = dists < avg

            nn = np.array(cand, dtype=int)[mask]
            nd = dists[mask]

            neighbors_col.append(" ".join(map(str, nn.tolist())))
            distance_col.append(" ".join(str(int(round(x))) for x in nd.tolist()))

        gdf["Neighbors"] = neighbors_col
        gdf["Distance"] = distance_col
        return gdf

    def build(
        self,
        data: np.ndarray,             # (T,N,F)
        node_id: np.ndarray,          # (N,)
        node_features: np.ndarray,    # (N,K)
        time_index: pd.DatetimeIndex, # (T,)
    ) -> Dict[int, Dict[str, Any]]:
        
        rawdata_cfg = self.cfg["raw_data"]
        dl_cfg = self.cfg["data"]
        frac = float(dl_cfg["node_frac"])
        group_size = int(round(frac * len(node_id)))
        group_size = max(1, min(group_size, len(node_id)))
        drypoints_mask = data[..., 0:1] == rawdata_cfg["outlier_value"]
        data[data == rawdata_cfg["outlier_value"]] = 0

        sampler = EvenFixedGroupSampler(
            N=len(node_id),
            group_size=group_size,
            #seed=self.seed,
            even_stride=True,
        )

        k_candidates = int(dl_cfg["neighbors"])
        input_feat_idx = dl_cfg["input_feature_idx"]
        output_feat_idx = dl_cfg["output_feature_idx"]
        split_ratio = tuple(dl_cfg["split_ratio"])
        input_len = int(dl_cfg["input_len"])
        horizon = int(dl_cfg["horizon"])
        batch_size = int(dl_cfg["batch_size"])
        num_workers = int(dl_cfg.get("num_workers", 0))
        
        # time_index: xarray DataArray or coordinate with dtype datetime64[ns]
        # data: shape (T, N, F)

        # 1) infer sampling interval in seconds
        if time_index.size < 2:
            raise ValueError("time_index must contain at least 2 timestamps")

        delta = time_index[1] - time_index[0]
        delta_sec = int(delta.values / np.timedelta64(1, "s"))

        if delta_sec <= 0:
            raise ValueError(f"Invalid sampling interval: {delta_sec} seconds")

        # 2) number of bins per day
        bins_per_day = 86400 // delta_sec
        self.sampling_freq_day = int(bins_per_day)

        # 3) seconds since midnight
        sec_mid = (
            time_index.dt.hour.astype(np.int64) * 3600
            + time_index.dt.minute.astype(np.int64) * 60
            + time_index.dt.second.astype(np.int64)
        )

        # 4) map to nearest time bin, then wrap into one day
        timeofday = np.rint(sec_mid.values / delta_sec).astype(np.int64) % bins_per_day

        # 5) other time features
        dayofweek = time_index.dt.dayofweek.astype(np.int64).values   # Monday=0, Sunday=6
        ts_ints = np.arange(time_index.size, dtype=np.int64)

        # 6) stack per-time features: (T, 3)
        time_features_t = np.stack([ts_ints, dayofweek, timeofday], axis=1)

        # 7) broadcast to all nodes: (T, N, 3)
        time_features = np.repeat(time_features_t[:, np.newaxis, :], data.shape[1], axis=1)
        # concat -> (T, 3 + D)
        data = np.concatenate([np.asarray(data), time_features], axis=2)

        seen = set()
        round_id = 0
        datasets: Dict[int, Dict[str, Any]] = {}

        # run enough rounds to cover all nodes at least once
        max_rounds = sampler.rounds_per_cycle()
        while len(seen) < len(node_id) and round_id < max_rounds:
            round_id += 1

            X_sub, NF_sub, NID_sub, idx = sampler.apply(data, node_features, node_id)
            seen.update(idx.tolist())

            # ---- build graph on this node group ----
            gdf = self._build_gdf_with_nn(NF_sub, k_candidates=k_candidates)

            A, sigma = build_adjacency_from_gdf(gdf, sigma=None, symmetric=True)
            A_gcn = normalize_gcn(add_self_loops(A))
            P_fwd, P_bwd = dual_random_walk(A)

            edge_index, edge_attr = sparse_to_pyg(A)
            edge_index_gcn, edge_attr_gcn = sparse_to_pyg(A_gcn)
            
            X_sub_time = X_sub[:, :, -3:]  # time features only
            X_sub_feat = X_sub[:, :, :-3]  # data features only
            X_sub_dry = drypoints_mask[:,idx,:]  # (N, 1) mask for dry points in this subgraph


            pos_tensor = torch.tensor(gdf[['Longitude', 'Latitude']].values)
            static_tensor = torch.tensor(gdf[['Bathymetry', 'Manning']].values)

            coastal_graph = CoastalGraph(pos_tensor, static_tensor, device='cpu')

            
            X_train_feat, X_val_feat, X_test_feat = split_nd_with_feature_select(
                data=X_sub_feat, input_feature_idx=input_feat_idx, split_ratio=split_ratio
            )
            X_train_time, X_val_time, X_test_time = split_nd_with_feature_select(
                data=X_sub_time, input_feature_idx=[0,1,2], split_ratio=split_ratio
            )

            X_train_dry, X_val_dry, X_test_dry = split_nd_with_feature_select(
                data=X_sub_dry.astype(int), input_feature_idx=[0], split_ratio=split_ratio
            ) 

            datasets[round_id] = {
                "node_id": NID_sub,
                "node_features": NF_sub,
                "bathymetry_manning": gdf[["Bathymetry", "Manning"]].to_numpy(np.float32),

                "gdf": gdf,
                "sigma": sigma,

                "A": A,
                "A_gcn": A_gcn,
                "P_fwd": P_fwd,
                "P_bwd": P_bwd,

                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_index_gcn": edge_index_gcn,
                "edge_attr_gcn": edge_attr_gcn,
                
                "X_sub_feat": X_sub_feat,
                "X_train_feat": X_train_feat,
                "X_val_feat": X_val_feat,
                "X_test_feat": X_test_feat,
                # "Y_train_feat": Y_train_feat,
                # "Y_val_feat": Y_val_feat,
                # "Y_test_feat": Y_test_feat,
                "X_sub_time": X_sub_time,
                "X_train_time": X_train_time,
                "X_val_time": X_val_time,
                "X_test_time": X_test_time,
                
               
                "X_sub_dry": X_sub_dry,
                "X_train_dry": X_train_dry,
                "X_val_dry": X_val_dry,
                "X_test_dry": X_test_dry, 
                "coastal_graph": coastal_graph,
            }

        # ---- global scaler on concatenated train sets ----
        X_train_concat = np.concatenate([datasets[i]["X_train_feat"] for i in datasets], axis=0)
        scaler = ZScoreScaler.fit(X_train_concat)
        self.scaler = scaler



        # ---- attach scaled arrays + loaders ----
        for i in datasets:
            datasets[i]["scaler"] = scaler
            datasets[i]["sub_dataset"] = np.concatenate([scaler.transform(datasets[i]["X_sub_feat"][..., input_feat_idx]), datasets[i]["X_sub_time"],  datasets[i]["X_sub_dry"]], axis=-1)
            for split in ["X_train_feat", "X_val_feat", "X_test_feat"]:
                Xs = datasets[i][split]
                Xt = datasets[i][split.replace("feat", "time")]  # time features (not scaled)
                Xd = datasets[i][split.replace("feat", "dry")]  # dry point mask (not scaled)
                datasets[i][split.replace("feat", "") + "scaled"] = np.concatenate([scaler.transform(Xs), Xt, Xd], axis=-1)
                #datasets[i][split.replace("feat", "") + "scaled_wave"] = np.concatenate([scaler.transform(Xs), Xt], axis=-1)

            # loaders (shuffle only train)
            for split, shuf in [("X_train", True), ("X_val", False), ("X_test", False)]:

                ds_obj, dl_obj = make_loader(
                    data_np = datasets[i][split + "_scaled"],
                    input_len=input_len,
                    horizon=horizon,
                    batch_size=batch_size,
                    shuffle=shuf,
                    num_workers=num_workers,
                    pin_memory=True,
                    return_index= True,
                    permute= None,
                    drop_last= False
                )
                datasets[i][split + "_ds"] = ds_obj
                datasets[i][split + "_loader"] = dl_obj
            ds_sub, dl_sub = make_loader(
                    data_np = datasets[i]["sub_dataset"],
                    input_len=input_len,
                    horizon=horizon,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    return_index= True,
                    permute= None,
                    drop_last= False
                )
            datasets[i]["sub_ds"] = ds_sub
            datasets[i]["sub_loader"] = dl_sub
                
            

        return datasets
