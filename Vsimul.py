import numpy as np
import matplotlib.pyplot as plt

# =========================
# 공통 함수
# =========================

def load_dem_asc(path):
    # ESRI ASCII DEM 읽기
    header = {}
    with open(path, "r") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                raise ValueError("ASC header too short")
            key, val = line.split()
            header[key.lower()] = float(val)
        data = np.loadtxt(f)

    dem = data.astype(float)
    nodata = header.get("nodata_value", None)
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, header


def make_top_bc(NX, NY, cx_frac, cy_frac, sigma_frac, base=1.0, amp=0.5):
    # 구름 전위 상부 경계
    cx = NX * cx_frac
    cy = NY * cy_frac
    sigma = sigma_frac * NX
    top_bc = np.zeros((NX, NY), dtype=float)
    for i in range(NX):
        for j in range(NY):
            r2 = (i - cx)**2 + (j - cy)**2
            top_bc[i, j] = base + amp * np.exp(-r2 / (2.0 * sigma**2))
    return top_bc


def compute_slope(dem, cellsize=1.0):
    # DEM 경사 크기
    dx = cellsize
    dy = cellsize
    NX, NY = dem.shape
    dzdx = np.zeros_like(dem)
    dzdy = np.zeros_like(dem)

    dzdx[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2 * dx)
    dzdx[0, :]    = (dem[1, :] - dem[0, :]) / dx
    dzdx[-1, :]   = (dem[-1, :] - dem[-2, :]) / dx

    dzdy[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / (2 * dy)
    dzdy[:, 0]    = (dem[:, 1] - dem[:, 0]) / dy
    dzdy[:, -1]   = (dem[:, -1] - dem[:, -2]) / dy

    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope


def compute_roughness(dem, size=3):
    # 3x3 거칠기 (표준편차)
    NX, NY = dem.shape
    r = size // 2
    rough = np.zeros_like(dem)
    for i in range(NX):
        for j in range(NY):
            i0 = max(0, i - r)
            i1 = min(NX, i + r + 1)
            j0 = max(0, j - r)
            j1 = min(NY, j + r + 1)
            window = dem[i0:i1, j0:j1]
            rough[i, j] = np.std(window)
    return rough


def bin_stat(X, C, nbins=10):
    # X 구간별 C 평균
    X_flat = X.ravel()
    C_flat = C.ravel()
    xmin, xmax = np.nanmin(X_flat), np.nanmax(X_flat)
    bins = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_C = []

    for a, b in zip(bins[:-1], bins[1:]):
        mask = (X_flat >= a) & (X_flat < b)
        if np.sum(mask) == 0:
            mean_C.append(np.nan)
        else:
            mean_C.append(np.nanmean(C_flat[mask]))
    return centers, np.array(mean_C)


def analyze_geom_relations(dem, slope, rough, counts, title_prefix="baseline"):
    # 고도/경사/거칠기 vs strike 그래프
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    centers_H, mean_C_H = bin_stat(dem, counts, nbins=10)
    axs[0].plot(centers_H, mean_C_H, marker="o")
    axs[0].set_xlabel("Altitude")
    axs[0].set_ylabel("Mean strike")
    axs[0].set_title(f"{title_prefix} H vs strike")

    centers_S, mean_C_S = bin_stat(slope, counts, nbins=10)
    axs[1].plot(centers_S, mean_C_S, marker="o")
    axs[1].set_xlabel("Slope")
    axs[1].set_ylabel("Mean strike")
    axs[1].set_title(f"{title_prefix} slope vs strike")

    centers_R, mean_C_R = bin_stat(rough, counts, nbins=10)
    axs[2].plot(centers_R, mean_C_R, marker="o")
    axs[2].set_xlabel("Roughness")
    axs[2].set_ylabel("Mean strike")
    axs[2].set_title(f"{title_prefix} rough vs strike")

    plt.tight_layout()
    plt.show()


def run_simulation(dem,
                   ground_k,
                   top_bc,
                   K_GROUND_MAX=5,
                   Z_CLEAR=6,
                   omega=1.85,
                   ITERS=500,
                   TRIALS_PER_SEED=50,
                   Eth_mode="uniform",
                   Eth_params=None,
                   seed=0,
                   verbose=True,
                   show_maps=True,
                   title_prefix="case"):
    # 한 케이스 시뮬
    NX, NY = dem.shape
    z_top = K_GROUND_MAX + Z_CLEAR + 1

    # 3D 격자
    mp = [[[0.0] * z_top for _ in range(NY)] for _ in range(NX)]
    mp_bound = [[[True] * z_top for _ in range(NY)] for _ in range(NX)]

    # 경계: 지형 도체 + 구름 평면
    for i in range(NX):
        for j in range(NY):
            gk = int(ground_k[i, j])

            # 지형 도체
            for k in range(0, gk + 1):
                mp_bound[i][j][k] = False
                mp[i][j][k] = 0.0

            # 상부 구름
            mp_bound[i][j][z_top - 1] = False
            mp[i][j][z_top - 1] = float(top_bc[i, j])

    # 공기 초기값: 지형~구름 선형
    for i in range(NX):
        for j in range(NY):
            gk = int(ground_k[i, j])
            top_val = mp[i][j][z_top - 1]
            span = (z_top - 1) - gk
            if span <= 0:
                continue
            for k in range(gk + 1, z_top - 1):
                if mp_bound[i][j][k]:
                    t = (k - gk) / span
                    mp[i][j][k] = top_val * t

    # SOR 해
    for it in range(ITERS):
        if verbose and it % 50 == 0:
            print(f"[{title_prefix}] SOR {it}/{ITERS}", flush=True)

        for parity in (0, 1):
            for i in range(NX):
                for j in range(NY):
                    for k in range(1, z_top - 1):
                        if not mp_bound[i][j][k]:
                            continue
                        if ((i + j + k) & 1) != parity:
                            continue

                        left  = mp[i-1][j][k] if i-1 >= 0 else mp[i+1][j][k]
                        right = mp[i+1][j][k] if i+1 < NX else mp[i-1][j][k]
                        front = mp[i][j-1][k] if j-1 >= 0 else mp[i][j+1][k]
                        back  = mp[i][j+1][k] if j+1 < NY else mp[i][j-1][k]
                        down  = mp[i][j][k-1]
                        up    = mp[i][j][k+1]

                        new_phi = (left + right + front + back + up + down) / 6.0
                        mp[i][j][k] = (1.0 - omega) * mp[i][j][k] + omega * new_phi

    phi = np.array(mp, dtype=float)

    # 표면 k (여기선 지형만)
    surface_k = ground_k.copy()

    # 전기장
    E_mag = np.zeros((NX, NY), dtype=float)
    for i in range(NX):
        for j in range(NY):
            ks = int(surface_k[i, j])
            kf = min(ks + 1, z_top - 2)

            if 0 < i < NX - 1:
                Ex = (phi[i+1, j, kf] - phi[i-1, j, kf]) * 0.5
            else:
                Ex = phi[min(i+1, NX-1), j, kf] - phi[max(i-1, 0), j, kf]

            if 0 < j < NY - 1:
                Ey = (phi[i, j+1, kf] - phi[i, j-1, kf]) * 0.5
            else:
                Ey = phi[i, min(j+1, NY-1), kf] - phi[i, max(j-1, 0), kf]

            if kf + 1 < z_top:
                Ez = phi[i, j, kf+1] - phi[i, j, kf]
            else:
                Ez = 0.0

            E_mag[i, j] = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)

    Ehat = E_mag.copy()

    # Eth_phys 모드
    Eth_phys = np.full((NX, NY), 1.0e5, dtype=float)
    if Eth_params is None:
        Eth_params = {}

    if Eth_mode == "uniform":
        Eth0 = Eth_params.get("Eth0", 1.0e5)
        Eth_phys[:] = Eth0

    elif Eth_mode == "sea":
        Eth_land = Eth_params.get("Eth_land", 1.0e5)
        Eth_sea  = Eth_params.get("Eth_sea", 5.0e4)
        dH = Eth_params.get("sea_band", 5.0)
        thresh = np.nanmin(dem) + dH
        Eth_phys[:] = Eth_land
        Eth_phys[dem < thresh] = Eth_sea

    elif Eth_mode == "alt":
        Eth0 = Eth_params.get("Eth0", 1.0e5)
        H_scale = Eth_params.get("H_scale", 8000.0)
        H0 = np.nanmin(dem)
        rho_rel = np.exp(-(dem - H0) / H_scale)
        Eth_phys = Eth0 * rho_rel

    elif Eth_mode == "sea+alt":
        Eth0 = Eth_params.get("Eth0", 1.0e5)
        H_scale = Eth_params.get("H_scale", 8000.0)
        H0 = np.nanmin(dem)
        rho_rel = np.exp(-(dem - H0) / H_scale)
        Eth_phys = Eth0 * rho_rel
        Eth_sea_factor = Eth_params.get("sea_factor", 0.5)
        dH = Eth_params.get("sea_band", 5.0)
        thresh = np.nanmin(dem) + dH
        Eth_phys[dem < thresh] *= Eth_sea_factor

    # w_att_map 계산
    AUTO_E0  = True
    F_TARGET = 0.05
    eps = 1e-9

    if AUTO_E0:
        R = Eth_phys / np.maximum(Ehat, eps)
        R = R[np.isfinite(R)]
        if R.size:
            E0 = np.quantile(R, F_TARGET)
        else:
            E0 = 1.0e5
    else:
        E0 = 1.0e5

    Eth_hat = Eth_phys / max(E0, eps)

    ETA = 2.0
    TAU = 0.08
    x_soft = (Ehat - Eth_hat) / TAU
    soft = np.maximum(x_soft, 0.0) + np.log1p(np.exp(-np.abs(x_soft)))
    w_att_map = (TAU * soft) ** ETA

    # 리더 이동 규칙
    moves = []
    moves += [(0, 0, -1)]
    moves += [(1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1)]
    moves += [(1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)]
    moves += [(2, 0, -1), (-2, 0, -1), (0, 2, -1), (0, -2, -1)]
    moves += [(2, 1, -1), (2, -1, -1), (-2, 1, -1), (-2, -1, -1),
              (1, 2, -1), (1, -2, -1), (-1, 2, -1), (-1, -2, -1)]
    moves += [(2, 2, -1), (2, -2, -1), (-2, 2, -1), (-2, -2, -1)]

    counts = np.zeros((NX, NY), dtype=float)

    np.random.seed(seed)
    for i0 in range(NX):
        for j0 in range(NY):
            for _ in range(TRIALS_PER_SEED):
                i, j, k = i0, j0, z_top - 2
                while True:
                    pcurr = float(phi[i, j, k])
                    nexts = []
                    weights = []

                    for dx, dy, dk in moves:
                        ni, nj, nk = i + dx, j + dy, k + dk
                        if not (0 <= ni < NX and 0 <= nj < NY and 0 <= nk < z_top):
                            continue
                        dphi = pcurr - float(phi[ni, nj, nk])
                        if dphi <= 0.0:
                            w = 0.0
                        else:
                            step_len = (dx*dx + dy*dy + dk*dk) ** 0.5
                            base_w = dphi / max(step_len, 1e-9)
                            if step_len > 1.5:
                                w = base_w * 0.4
                            else:
                                w = base_w
                        nexts.append((ni, nj, nk))
                        weights.append(w)

                    if not nexts:
                        break

                    s = sum(weights)
                    if s <= 1e-12:
                        diffs = [pcurr - float(phi[ni, nj, nk]) for (ni, nj, nk) in nexts]
                        pick = int(np.argmax(diffs))
                    else:
                        r = np.random.random() * s
                        acc = 0.0
                        pick = 0
                        for idx, w in enumerate(weights):
                            acc += w
                            if r <= acc:
                                pick = idx
                                break

                    i, j, k = nexts[pick]

                    if (not mp_bound[i][j][k]) and (k != z_top - 1):
                        counts[i, j] += w_att_map[i, j]
                        break
                    if k == 0:
                        counts[i, j] += w_att_map[i, j]
                        break

    if verbose:
        print(f"[{title_prefix}] total strike weight = {counts.sum():.3e}")

    if show_maps:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        im0 = axs[0].imshow(dem.T, origin="lower", cmap="terrain")
        axs[0].set_title(f"{title_prefix} DEM")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(ground_k.T, origin="lower", cmap="viridis")
        axs[1].set_title(f"{title_prefix} ground_k")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(counts.T, origin="lower", cmap="inferno")
        axs[2].set_title(f"{title_prefix} strikes")
        plt.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        plt.show()

    return counts, Ehat, Eth_phys


# =========================
# 메인 실행
# =========================

# 1. DEM 읽기
dem, dem_header = load_dem_asc(r"thunder\설곽DEM90.asc")
NX, NY = dem.shape

# NaN 처리
if np.isnan(dem).any():
    mn = np.nanmin(dem)
    dem = np.where(np.isnan(dem), mn, dem)

cellsize = dem_header.get("cellsize", 1.0)

# 2. 지형 k 압축
K_GROUND_MAX = 5
Z_CLEAR      = 6

H = dem.astype(float)
H = H - H.min()
H_norm = H / (H.max() + 1e-12)
ground_k = np.rint(H_norm * K_GROUND_MAX).astype(int)
ground_k = np.clip(ground_k, 0, K_GROUND_MAX)

# 3. 기본 구름 경계 (중앙, sigma=0.35)
top_bc_base = make_top_bc(NX, NY, cx_frac=0.5, cy_frac=0.5, sigma_frac=0.35)

# 4. 기본 케이스 실행
counts_base, Ehat_base, Eth_base = run_simulation(
    dem,
    ground_k,
    top_bc_base,
    K_GROUND_MAX=K_GROUND_MAX,
    Z_CLEAR=Z_CLEAR,
    omega=1.85,
    ITERS=400,
    TRIALS_PER_SEED=50,
    Eth_mode="uniform",
    Eth_params={"Eth0": 1.0e5},
    seed=0,
    verbose=True,
    show_maps=True,
    title_prefix="baseline"
)

# 5. 지형 파라미터 계산 (slope, roughness)
slope = compute_slope(dem, cellsize=cellsize)
rough = compute_roughness(dem, size=3)

# 6. baseline 고도/경사/거칠기 vs strike
analyze_geom_relations(dem, slope, rough, counts_base, title_prefix="baseline")

# =========================
# 추가 실험: 구름 위치
# =========================

# 최고 고도, 최저 고도 위치
peak_idx = np.unravel_index(np.nanargmax(dem), dem.shape)
low_idx  = np.unravel_index(np.nanargmin(dem), dem.shape)

cx_peak = peak_idx[0] / NX
cy_peak = peak_idx[1] / NY
cx_low  = low_idx[0] / NX
cy_low  = low_idx[1] / NY

cases_cloud_pos = [
    ("cloud_center", 0.5, 0.5),
    ("cloud_peak",   cx_peak, cy_peak),
    ("cloud_low",    cx_low,  cy_low),
]

for name, cx_f, cy_f in cases_cloud_pos:
    top_bc = make_top_bc(NX, NY, cx_frac=cx_f, cy_frac=cy_f, sigma_frac=0.35)
    counts, _, _ = run_simulation(
        dem,
        ground_k,
        top_bc,
        K_GROUND_MAX=K_GROUND_MAX,
        Z_CLEAR=Z_CLEAR,
        omega=1.85,
        ITERS=300,
        TRIALS_PER_SEED=40,
        Eth_mode="uniform",
        Eth_params={"Eth0": 1.0e5},
        seed=0,
        verbose=True,
        show_maps=True,
        title_prefix=name
    )
    # 필요하면 아래 주석 풀어서 각 케이스 통계도 볼 수 있음
    # analyze_geom_relations(dem, slope, rough, counts, title_prefix=name)

# =========================
# 추가 실험: 구름 크기
# =========================

cases_cloud_sigma = [
    ("cloud_sigma_small", 0.5, 0.5, 0.2),
    ("cloud_sigma_mid",   0.5, 0.5, 0.35),
    ("cloud_sigma_large", 0.5, 0.5, 0.5),
]

for name, cx_f, cy_f, sig_f in cases_cloud_sigma:
    top_bc = make_top_bc(NX, NY, cx_frac=cx_f, cy_frac=cy_f, sigma_frac=sig_f)
    counts, _, _ = run_simulation(
        dem,
        ground_k,
        top_bc,
        K_GROUND_MAX=K_GROUND_MAX,
        Z_CLEAR=Z_CLEAR,
        omega=1.85,
        ITERS=300,
        TRIALS_PER_SEED=40,
        Eth_mode="uniform",
        Eth_params={"Eth0": 1.0e5},
        seed=0,
        verbose=True,
        show_maps=True,
        title_prefix=name
    )

# =========================
# 추가 실험: Eth_phys (습윤/고도)
# =========================

# 같은 top_bc_base로 Eth 모드만 변경

# 1) sea 모드 (낮은 지형=물)
counts_sea, _, Eth_sea = run_simulation(
    dem,
    ground_k,
    top_bc_base,
    K_GROUND_MAX=K_GROUND_MAX,
    Z_CLEAR=Z_CLEAR,
    omega=1.85,
    ITERS=300,
    TRIALS_PER_SEED=40,
    Eth_mode="sea",
    Eth_params={"Eth_land": 1.0e5, "Eth_sea": 5.0e4, "sea_band": 5.0},
    seed=1,
    verbose=True,
    show_maps=True,
    title_prefix="Eth_sea"
)

# 2) alt 모드 (공기 밀도 효과)
counts_alt, _, Eth_alt = run_simulation(
    dem,
    ground_k,
    top_bc_base,
    K_GROUND_MAX=K_GROUND_MAX,
    Z_CLEAR=Z_CLEAR,
    omega=1.85,
    ITERS=300,
    TRIALS_PER_SEED=40,
    Eth_mode="alt",
    Eth_params={"Eth0": 1.0e5, "H_scale": 8000.0},
    seed=2,
    verbose=True,
    show_maps=True,
    title_prefix="Eth_alt"
)

# 3) sea+alt 모드
counts_sea_alt, _, Eth_sea_alt = run_simulation(
    dem,
    ground_k,
    top_bc_base,
    K_GROUND_MAX=K_GROUND_MAX,
    Z_CLEAR=Z_CLEAR,
    omega=1.85,
    ITERS=300,
    TRIALS_PER_SEED=40,
    Eth_mode="sea+alt",
    Eth_params={"Eth0": 1.0e5, "H_scale": 8000.0, "sea_factor": 0.5, "sea_band": 5.0},
    seed=3,
    verbose=True,
    show_maps=True,
    title_prefix="Eth_sea_alt"
)

print("=== done ===")
