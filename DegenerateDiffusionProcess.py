# %%
import sympy as sp
from sympy import symbols, log, det, Matrix
from sympy.tensor.array import derive_by_array
from einsum_sympy import einsum_sympy
from dataclasses import dataclass
from sympy.utilities.lambdify import lambdify
import numpy as np
import sympy as sp
from sympy import Array, symbols, factorial, tensorproduct, log, det, Matrix
from sympy.tensor.array import derive_by_array
# from einsum_sympy import einsum_sympy # 元のコードにはあったが、このコードでは未使用のためコメントアウトしても良い
from dataclasses import dataclass
from sympy.utilities.lambdify import lambdify
import numpy as np
from typing import Tuple, Optional, Sequence # 型ヒント用に追加
# %%
@dataclass(frozen=True)
class DegenerateDiffusionProcess:
    """
    多次元の拡散過程と観測過程を扱うためのクラス。

    記号計算 (sympy) を用いてモデルを定義し、
    数値計算 (numpy) のための関数を lambdify で生成する。

    Attributes:
        x (sp.Array): 状態変数 (sympyシンボル配列, shape=(d_x,))
        y (sp.Array): 観測変数 (sympyシンボル配列, shape=(d_y,))
        theta_1 (sp.Array): パラメータ1 (sympyシンボル配列)
        theta_2 (sp.Array): パラメータ2 (sympyシンボル配列)
        theta_3 (sp.Array): パラメータ3 (sympyシンボル配列)
        A (sp.Array): ドリフト項の x 成分 (sympy式配列, shape=(d_x,))
        B (sp.Array): 拡散項 (sympy式配列, shape=(d_x, r))
        H (sp.Array): 観測過程のドリフト項 (sympy式配列, shape=(d_y,))
        # --- 以下は __post_init__ で自動生成 ---
        A_func : ドリフト A を計算する numpy 関数
        B_func : 拡散項 B を計算する numpy 関数
        H_func : 観測ドリフト H を計算する numpy 関数
    """

    x: sp.Array
    y: sp.Array
    theta_1: sp.Array
    theta_2: sp.Array
    theta_3: sp.Array
    A: sp.Array
    B: sp.Array
    H: sp.Array

    def __post_init__(self):
        """
        初期化後の設定処理。
        派生的な属性の計算、lambdifyによる関数生成を行う。
        frozen=True のため、属性設定には object.__setattr__ を使用する。
        """

        # --- lambdify による数値計算用関数の生成 ---
        common_args = (self.x, self.y)
        try:
            object.__setattr__(self, "A_func",
                               lambdify((*common_args, self.theta_2), self.A, modules="numpy"))
            object.__setattr__(self, "B_func",
                               lambdify((*common_args, self.theta_1), self.B, modules="numpy"))
            object.__setattr__(self, "H_func",
                               lambdify((*common_args, self.theta_3), self.H, modules="numpy"))

        except Exception as e:
            print(f"Error during lambdification in __post_init__: {e}")
            raise

    # --- シミュレーション関数 ---
    def simulate(
        self,
        true_theta: Tuple[np.ndarray, np.ndarray, np.ndarray], # 型ヒント追加
        t_max: float,
        h: float,
        burn_out: float,
        seed: int = 42,
        x0: Optional[np.ndarray] = None, # 型ヒント追加
        y0: Optional[np.ndarray] = None, # 型ヒント追加
        dt: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]: # 返り値の型ヒント追加
        """
        Euler-Maruyama法を用いて状態変数 x と観測変数 y の時系列データを生成する。

        Parameters:
            true_theta (tuple): (theta_1_val, theta_2_val, theta_3_val) の真値。
            t_max (float): 生成する時系列データの期間長（burn_out期間を除く）。
            h (float): 生成されるデータの時間ステップ幅（間引き後）。
            burn_out (float): 定常状態に達するまで捨てる初期期間。
            seed (int): 乱数生成器のシード値。
            x0 (np.ndarray, optional): x の初期値ベクトル。None の場合はゼロベクトル。
            y0 (np.ndarray, optional): y の初期値ベクトル。None の場合はゼロベクトル。
            dt (float): シミュレーション内部の微小時間ステップ幅。

        Returns:
            tuple[np.ndarray, np.ndarray]: (x_series, y_series)
                x_series: x の時系列データ (shape=(T, d_x))
                y_series: y の時系列データ (shape=(T, d_y))
                ここで T は t_max / h に近い整数
        """
        np.random.seed(seed)
        theta_1_val, theta_2_val, theta_3_val = true_theta
        total_steps = int(np.round((t_max + burn_out) / dt)) # 丸め処理追加
        d_x = self.x.shape[0]
        d_y = self.y.shape[0]
        r = self.B.shape[1] # Number of Wiener processes

        # dt と h の関係チェックと step_stride の計算
        if not np.isclose(h % dt, 0, atol=1e-8) and not np.isclose(h % dt, dt, atol=1e-8):
             print(f"Warning: h ({h}) may not be an integer multiple of dt ({dt}). Thinning interval might be slightly inaccurate due to rounding.")
        step_stride = max(1, int(np.round(h / dt))) # 丸めて最低1にする

        x_all = np.zeros((total_steps + 1, d_x))
        y_all = np.zeros((total_steps + 1, d_y))
        x_all[0] = x0 if x0 is not None else np.zeros(d_x)
        y_all[0] = y0 if y0 is not None else np.zeros(d_y)

        for t in range(total_steps):
            xt, yt = x_all[t], y_all[t]
            try:
                # self.*_func を直接呼び出す
                A_val = self.A_func(xt, yt, theta_2_val)
                B_val = self.B_func(xt, yt, theta_1_val)
                H_val = self.H_func(xt, yt, theta_3_val)
            except Exception as e:
                print(f"Error evaluating A, B, or H function at step {t}: {e}")
                print(f"xt={xt}, yt={yt}, theta_1={theta_1_val}, theta_2={theta_2_val}, theta_3={theta_3_val}")
                raise

            dW = np.random.randn(r) * np.sqrt(dt)
            diffusion_term = np.dot(B_val, dW) # np.dotを使用

            # 形状チェックをタプル (d_x,) と比較
            if diffusion_term.shape != (d_x,):
                raise ValueError(f"Shape mismatch in diffusion term: expected ({d_x},), got {diffusion_term.shape}")

            x_all[t + 1] = xt + A_val * dt + diffusion_term
            y_all[t + 1] = yt + H_val * dt
            # もし観測ノイズが必要な場合:
            # 例えば dy = H dt + G dV の G があるなら G_func を __post_init__ で作成
            # G_val = self.G_func(xt, yt, theta_g_val) # Gを計算
            # dV = np.random.randn(d_y) * np.sqrt(dt) # 観測ノイズの増分
            # y_all[t + 1] = yt + H_val * dt + np.dot(G_val, dV)

        # Apply burn-out and thinning
        start_index = int(np.round(burn_out / dt)) # 丸め処理追加

        x_series = x_all[start_index::step_stride]
        y_series = y_all[start_index::step_stride]

        return x_series, y_series

# --- 以下、使用例（元のコードにはなかったので参考として） ---
if __name__ == '__main__':
    # 例: 1次元 Ornstein-Uhlenbeck process (dx = -theta2*x dt + theta1 dW)
    #      観測 y は x に依存しない定数ドリフト (dy = theta3 dt)

    # シンボル定義
    x_sym = sp.Array([symbols('x_0')])
    y_sym = sp.Array([symbols('y_0')])
    theta1_sym = sp.Array([symbols('sigma')]) # diffusion coefficient
    theta2_sym = sp.Array([symbols('kappa')]) # reversion rate
    theta3_sym = sp.Array([symbols('mu')])    # observation drift

    # モデル定義
    A_expr = sp.Array([-theta2_sym[0] * x_sym[0]])
    B_expr = sp.Array([[theta1_sym[0]]]) # shape (d_x, r) = (1, 1)
    H_expr = sp.Array([theta3_sym[0] * x_sym[0]]) # shape (d_y,) = (1,)

    # インスタンス生成
    process = DegenerateDiffusionProcess(
        x=x_sym, y=y_sym,
        theta_1=theta1_sym, theta_2=theta2_sym, theta_3=theta3_sym,
        A=A_expr, B=B_expr, H=H_expr
    )

    # パラメータ真値
    true_sigma = np.array([0.5])
    true_kappa = np.array([1.0])
    true_mu = np.array([0.1])
    true_thetas = (true_sigma, true_kappa, true_mu)

    # シミュレーション実行
    T_MAX = 1000.0
    H_STEP = 0.1 # 間引き後のステップ幅
    BURN_OUT = 100.0
    DT_SIM = 0.01 # シミュレーション内部のステップ幅

    print("Simulating...")
    x_data, y_data = process.simulate(
        true_theta=true_thetas,
        t_max=T_MAX,
        h=H_STEP,
        burn_out=BURN_OUT,
        dt=DT_SIM,
        x0=np.array([0.0]),
        y0=np.array([0.0]),
        seed=123
    )

    print(f"Simulation finished. Generated data shapes:")
    print(f"x_series shape: {x_data.shape}") # (T, d_x)
    print(f"y_series shape: {y_data.shape}") # (T, d_y)

    # 簡単なプロット例 (matplotlibが必要)
    try:
        import matplotlib.pyplot as plt
        time = np.arange(x_data.shape[0]) * H_STEP
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(time, x_data[:, 0], label='x_0 (Simulated)')
        ax[0].set_ylabel('State x')
        ax[0].grid(True)
        ax[0].legend()
        ax[1].plot(time, y_data[:, 0], label='y_0 (Simulated)')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Observation y')
        ax[1].grid(True)
        ax[1].legend()
        plt.suptitle('Simulated Degenerate Diffusion Process')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitleとの重なり調整
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")
# %%
