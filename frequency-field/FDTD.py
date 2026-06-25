import numpy as np
import matplotlib.pyplot as plt

import meep as mp
from meep import materials


# 單一頻率 FDTD 模擬
class FDTDSimulator:

    # 參數設定
    def __init__(self, freq=1, resolution=333, Nt=1500, ratio=0.4):
        
        self.fre_unit = 3e14
        self.f0_base = 1e14
        self.fre = self.f0_base * freq
        self.fcen = self.fre / self.fre_unit

        self.resolution = resolution

        self.pml_grid = 5
        self.dpml = self.pml_grid * (1 / self.resolution)

        self.length_unit = 1e-6
        self.side_real = 4.5e-07
        self.side_mp = self.side_real / self.length_unit

        self.rad_real = 1e-07

        self.Nt = Nt

        self.source_pos = -(0.75) * self.side_mp / 2

        self.sim = None

        self.field_frames = []

        self.split_ratio = ratio

    # 參數整理函式
    def params(self):
        
        side_design = self.side_mp
        self.side_design_grid = int(self.side_mp * self.resolution)

        self.side_field = side_design * (32 / 75)
        self.side_field_grid = int(self.side_field * self.resolution)

        self.side_mid = side_design - 2 * self.side_field

        self.side_all = self.side_mp + 2 * self.dpml
        self.side_all_grid = int(self.side_all * self.resolution)

        self.rad_meep = self.rad_real / self.length_unit

    # MEEP 模擬設置
    def setup(self):
        
        self.params()

        pml_layers = [mp.PML(self.dpml)]

        self.cell_size = mp.Vector3(self.side_all, self.side_all)

        geometry = [
            
            mp.Block(
                size=mp.Vector3(self.side_mp, self.side_mp),
                center=mp.Vector3(0, 0, 0),
                material=mp.Medium(epsilon=3.5)
            ),

            mp.Cylinder(
                radius=self.rad_meep,
                center=mp.Vector3(0, 0),
                material=mp.Medium(epsilon=100)
            )
        ]

        sources = [
            mp.Source(
                mp.ContinuousSource(frequency=self.fcen),
                component=mp.Hz,
                center=mp.Vector3(self.source_pos, 0),
            )
        ]

        self.sim = mp.Simulation(
            cell_size=self.cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            dimensions=2
        )

        return self.sim

    # 記錄場圖
    def capture_field(self, sim_obj):

        self.params()

        ez_data = sim_obj.get_array(
            center=mp.Vector3(),
            size=self.cell_size,
            component=mp.Hz
        )

        self.field_frames.append(ez_data)

    # 實際 MEEP 模擬
    def run(self, total_meep_time=100, num_frames=20):

        self.params()

        interval = total_meep_time / num_frames

        self.sim.run(
            mp.at_every(interval, self.capture_field),
            until=total_meep_time
        )

        data_3d = np.array(self.field_frames)

        return data_3d

    # 特徵及標籤的資料分割
    def XY_split(self, data):

        self.params()

        total_frames = data.shape[0]

        if total_frames < 2:
            raise ValueError(f"場圖數量不足，至少需要 2 幀，目前只有 {total_frames} 幀。")

        split_idx = int(total_frames * self.split_ratio)

        split_idx = max(split_idx, 1)

        split_idx = min(split_idx, total_frames - 1)

        X_sample = np.transpose(data[:split_idx], (1, 2, 0))
        Y_sample = np.transpose(data[split_idx:], (1, 2, 0))

        return X_sample, Y_sample


# 多個頻率的資料集建立
def generate_dataset(n_samples: int = 10,
                     
                     freq_min: float = 1.0,
                     freq_max: float = 5.0,
                     
                     total_meep_time: int = 100,
                     num_frames: int = 20,
                     
                     resolution: int = 333,
                     
                     seed: int = 42,
                     
                     ratio: float = 0.4,
                     
                     save_path_X: str = "dataset_X.npy",
                     save_path_Y: str = "dataset_Y.npy",
                     save_path_freq: str = "dataset_freq.npy",
                     
                     verbose: bool = True,
                    ):

    rng = np.random.default_rng(seed)
    freq_list = rng.uniform(freq_min, freq_max, size=n_samples)

    X_list = []
    Y_list = []

    for i, freq in enumerate(freq_list):
        
        if verbose:
            fre_hz = freq * 1e14
            print("=" * 40)
            print(f"[{i+1}/{n_samples}] freq={fre_hz:.3e} Hz")
            print("=" * 40)

        sim = FDTDSimulator(freq=freq, resolution=resolution, ratio=ratio)
        sim.setup()

        data = sim.run(total_meep_time=total_meep_time, num_frames=num_frames)
        
        X, Y = sim.XY_split(data)
        X_list.append(X)
        Y_list.append(Y)

        if verbose:
            print("=" * 40)
            print(f"X : {X.shape} → (Nx, Ny, Tx)")
            print(f"Y : {Y.shape} → (Nx, Ny, Ty)")
            print("=" * 40)

    X_all = np.stack(X_list, axis=0)
    Y_all = np.stack(Y_list, axis=0)

    if verbose:
        print("=" * 40)
        print(f"X_all shape : {X_all.shape} → (Batch, Nx, Ny, Tx)")
        print(f"Y_all shape : {Y_all.shape} → (Batch, Nx, Ny, Ty)")
        print(f"freq range  : [{freq_list.min():.4f}, {freq_list.max():.4f}]")
        print("=" * 40)

    np.save(save_path_X, X_all)
    np.save(save_path_Y, Y_all)
    np.save(save_path_freq, freq_list)

    if verbose:
        print(f"Saved feature data   → {save_path_X}")
        print(f"Saved label data     → {save_path_Y}")
        print(f"Saved frequency data → {save_path_freq}")
        print("=" * 40)

    return X_all, Y_all, freq_list
