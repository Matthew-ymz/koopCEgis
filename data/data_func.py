import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns
from numpy.polynomial.legendre import Legendre
from scipy.special import legendre # legendre(n)用于生成n阶勒让德多项式。
from scipy.integrate import fixed_quad 
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import warnings
import matplotlib as mpl
import scipy
import sklearn
import pysindy as ps 
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, fixed
import ipywidgets as widgets

# Lorenz model
def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    """
    Lorenz系统的微分方程
    
    参数:
        t: 时间
        x: 状态向量 [x, y, z]
        sigma: 参数 (默认 10)
        beta: 参数 (默认 2.66667)
        rho: 参数 (默认 28)
    
    返回:
        dx/dt: 状态导数
    """
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

# Nonlinear pendulum
def npendulum(t, x):
    """
    nonlinear pendulum 的 Docstring
    
    参数
    :param t: 时间
    :param x: 状态向量 [x, y]

    返回
    :return dx/dt: 状态导数
    """
    return [
        x[1],
        -np.sin(x[0])
    ]

def double_osc(t, x):
    """
    a simple nonlinear system
    """
    w1 = 1.
    w2 = 1.618
    return [
        - x[1] * w1,
        x[0] * w1,
        x[0]**2 - x[3] * w2,
        x[2] * w2
    ]

def kuramoto_ode_cluster(theta, omega, K_matrix):
    """Kuramoto ODE with custom coupling matrix K_ij."""
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omega[i] + np.sum(
            K_matrix[i, :] * np.sin(theta - theta[i])
        ) / N
    return dtheta

def generate_kuramoto_cluster_data_sin_cos(
    N=12, n_clusters=3, K_intra=2.0, K_inter=0.2,
    dt=0.01, T=30, noise=0.0, random_seed1=0, random_seed2=0
):
    """
    生成带‘团结构’的Kuramoto振子数据。
    团内耦合K_intra > 团间耦合K_inter。
    """
    np.random.seed(random_seed1)
    t_steps = int(T / dt)
    t = np.arange(0, T, dt)
    omega = 2 * np.pi * (0.2 + 0.05 * np.random.randn(N))
    np.random.seed(random_seed2)
    theta = np.random.uniform(0, 2 * np.pi, N)

    # --- 构造耦合矩阵 ---
    cluster_size = N // n_clusters
    K_matrix = np.full((N, N), K_inter)
    for c in range(n_clusters):
        start = c * cluster_size
        end = N if c == n_clusters - 1 else (c + 1) * cluster_size
        K_matrix[start:end, start:end] = K_intra

    # --- 时间积分 ---
    theta_hist = np.zeros((t_steps, N))
    theta_hist[0] = theta
    for i in range(1, t_steps):
        dtheta = kuramoto_ode_cluster(theta, omega, K_matrix)
        theta = np.mod(theta + dtheta * dt, 2 * np.pi)
        if noise > 0:
            theta += noise * np.random.randn(N)
        theta_hist[i] = theta

    X = np.hstack([np.cos(theta_hist), np.sin(theta_hist)])  # sin-cos embedding
    return X, theta_hist, t, K_matrix


def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=1))

def compute_cluster_order_parameters(theta, n_clusters):
    """计算每个团的序参量"""
    N = theta.shape[1]
    cluster_size = N // n_clusters
    group_r = []
    for c in range(n_clusters):
        start = c * cluster_size
        end = N if c == n_clusters - 1 else (c + 1) * cluster_size
        r_c = compute_order_parameter(theta[:, start:end])
        group_r.append(r_c)
    return group_r

def plot_clustered_kuramoto(N=12, n_clusters=3, K_intra=2.0, K_inter=0.2, noise=0.0, T=30, dt=0.01, random_seed1=0, random_seed2=0):
    X_embed, theta_hist, t, K_matrix = generate_kuramoto_cluster_data_sin_cos(
        N=N, n_clusters=n_clusters, K_intra=K_intra, K_inter=K_inter, dt=dt, T=T, noise=noise, random_seed1=random_seed1, random_seed2=random_seed2
    )

    r_total = compute_order_parameter(theta_hist)
    r_groups = compute_cluster_order_parameters(theta_hist, n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    colors = plt.cm.tab10(np.arange(n_clusters))
    N_cluster = N // n_clusters

    # (2) 每个振子的相位随时间
    ax2 = axes[0]
    for g in range(n_clusters):
        for i in range(g * N_cluster, (g + 1) * N_cluster):
            ax2.plot(t, X_embed[:, i], lw=0.8, color=colors[g])
    ax2.set_title("Phase Evolution θ_i(t)")
    ax2.set_xlabel("Time")

    # (3) 各团与总体序参量
    ax3 = axes[1]
    ax3.plot(t, r_total, "k", lw=2, label="Overall r(t)")
    for g, r_c in enumerate(r_groups):
        ax3.plot(t, r_c, lw=2, color=colors[g], label=f"Group {g+1}")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Order parameters")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return X_embed, theta_hist, t, K_matrix

# SIR model
def sir_model_normalized(y, t, beta, gamma):
    """
    归一化SIR模型的微分方程。
    y: 一个包含s, i, r比例的列表或数组
    t: 时间点
    beta: 传染率
    gamma: 康复率
    """
    s, i, r = y
    ds_dt = -beta * s * i
    di_dt = beta * s * i - gamma * i
    dr_dt = gamma * i
    return [ds_dt, di_dt, dr_dt]

def gen_sir_data(initial_infected_ratio=0.5,
                 initial_recovered_ratio=0.0,beta=0.3,gamma=0.05,
                 total_days=200,dt=0.01,noise_mean=0.0,
                 noise_std=0.001,random_seed=None):
    """
    生成带噪声的SIR模型数据
    
    参数:
    --------
    initial_infected_ratio : float初始感染比例
    initial_recovered_ratio : float初始康复比例
    beta : float传染率
    gamma : float康复率
    total_days : int模拟总天数
    dt : int间距
    noise_mean : float高斯噪声均值
    noise_std : float高斯噪声标准差
    random_seed : int or None随机种子，用于复现结果
    
    返回:
    --------
    data : np.ndarray包含噪声的SIR数据，列顺序：s, i,r
    t : np.ndarray时间数组
    """
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 计算初始易感比例
    initial_susceptible_ratio = 1.0 - initial_infected_ratio - initial_recovered_ratio
    y0 = [initial_susceptible_ratio, initial_infected_ratio, initial_recovered_ratio]
    
    # 生成时间点
    t = np.arange(0, total_days, dt)
    
    # 求解SIR模型
    solution = odeint(sir_model_normalized, y0, t, args=(beta, gamma))
    
    # 构造无噪声数据（重复s和i列）
    s_col = solution[:, 0, np.newaxis]
    i_col = solution[:, 1, np.newaxis]
    r_col = solution[:, 2, np.newaxis]
    data_noiseless = np.hstack([s_col, i_col, r_col])
    
    # 添加高斯噪声
    noise = noise_mean + np.random.randn(*data_noiseless.shape) * noise_std
    x = data_noiseless + noise
    
    return x, t

def plot_sir_results(x, t, figsize=(12, 4), title_prefix="SIR Model"):
    if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
        x = [x]
        t = [t]
    # ========== 初始化画布 ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 多序列样式配置（自动渐变颜色+透明度） ==========
    n_sequences = len(x)
    # 生成渐变颜色（基于序列数量自动分配）
    cmap = plt.cm.viridis  # 美观的渐变色谱（蓝→绿→黄→红）
    colors = cmap(np.linspace(0, 1, n_sequences))
    # 透明度梯度（避免多序列重叠过密）
    alphas = np.linspace(0.2, 0.8, n_sequences)

    # ========== 遍历绘图 ==========
    for idx in range(n_sequences):
        data = x[idx]
        t_seq = t[idx]
        color = colors[idx]
        alpha = alphas[idx]
        
        # 时间序列图：单序列展示s/i/r，多序列只展示i（带渐变颜色+透明度）
        if n_sequences == 1:  # 单序列：保留原始样式
            ax1.plot(t_seq, data[:,0], color='blue', label='Susceptible (s)', linewidth=1.5)
            ax1.plot(t_seq, data[:,1], color='red', label='Infected (i)', linewidth=1.5)
            ax1.plot(t_seq, 1-data[:,0]-data[:,1], color='green', label='Recovered (r)', linewidth=1.5)
            ax1.legend()  # 单序列显示图例
        else:  # 多序列：渐变颜色+透明度，仅展示i列
            ax1.plot(t_seq, data[:,1], color=color, alpha=alpha, linewidth=0.5)
        
        # 相平面图：单/多序列差异化样式
        ax2.plot(
            data[:,0], data[:,1], 
            color=color, alpha=alpha,
            linewidth=1.5 if n_sequences == 1 else 0.5
        )

    # ========== 统一图表配置 ==========
    # 时间序列图
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Proportion')
    ax1.set_title(f'{title_prefix} - Time Series')
    ax1.grid(alpha=0.3)
    
    # 相平面图（多序列添加颜色条，直观区分轨迹）
    ax2.set_xlabel('Susceptible (s)')
    ax2.set_ylabel('Infected (i)')
    ax2.set_title(f'{title_prefix} - Phase Plane')
    ax2.grid(alpha=0.3)
    
    # 多序列添加颜色条（无需额外参数，自动生成）
    if n_sequences > 1:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, n_sequences-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, shrink=0.8)
        cbar.set_label('Sequence Index', fontsize=8)

    plt.tight_layout()
    plt.show()

# rulkov map
def generate_two_population_neuron_data(
    n_a=400, n_b=400,              # 两个群体的神经元数量
    alpha_a=4.6,                 # α群体的α参数（可激发性）
    alpha_b=4.6,                 # β群体的α参数
    sigma_a=0.225,                 # α群体的σ参数（异质性）
    sigma_b=0.225,                 # β群体的σ参数
    mu=0.001,                    # 慢变参数（局部动力学）
    gamma=0.08,                   # 群体内耦合强度（μ）
    epsilon=0.04,                # 群体间耦合强度（ε）
    T=10000,                     # 总模拟步数
    transients=1000,             # 舍弃的暂态步数
    x0_a=-1, x0_b=-0.5,          # a/b群体x初始值（None则该群体独立随机）
    y0_a=-3.5, y0_b=-3, 
    seed=42                      # 随机种子
):
    """
    生成两个群体神经元数据，基于平均场耦合动力学。
    
    参数:
        n_a, n_b: 两个群体的神经元数量
        alpha_a, alpha_b: 两个群体的可激发性参数
        sigma_a, sigma_b: 两个群体的异质性参数
        mu: 慢变参数（局部动力学）
        gamma: 群体内耦合强度（对应图中的μ）
        epsilon: 群体间耦合强度（对应图中的ε）
        T: 总模拟步数
        transients: 舍弃的暂态步数
        seed: 随机种子
        
    返回:
        dict: 包含所有数据的字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 总神经元数
    N = n_a + n_b
    
    # 处理参数：如果是标量，扩展为列表
    if isinstance(alpha_a, (int, float)):
        alpha_a = np.full(n_a, alpha_a)
    else:
        alpha_a = np.array(alpha_a)
        
    if isinstance(alpha_b, (int, float)):
        alpha_b = np.full(n_b, alpha_b)
    else:
        alpha_b = np.array(alpha_b)
    
    if isinstance(sigma_a, (int, float)):
        sigma_a = np.full(n_a, sigma_a)
    else:
        sigma_a = np.array(sigma_a)
        
    if isinstance(sigma_b, (int, float)):
        sigma_b = np.full(n_b, sigma_b)
    else:
        sigma_b = np.array(sigma_b)
    # 合并所有参数
    alpha_all = np.concatenate([alpha_a, alpha_b])
    sigma_all = np.concatenate([sigma_a, sigma_b])
            
    # 初始化数组
    x_series = np.zeros((N, T))
    y_series = np.zeros((N, T))
     
    if x0_a is None:
        # None → a群体每个神经元独立随机
        x_series[:n_a, 0] = np.random.uniform(-1.5, -0.5, n_a)
        x0_a_record = "随机值（每个神经元独立）"
    else:
        # 传值 → a群体所有神经元共用该值
        x_series[:n_a, 0] = x0_a
        x0_a_record = x0_a
    
    if x0_b is None:
        # None → b群体每个神经元独立随机
        x_series[n_a:, 0] = np.random.uniform(-1.5, -0.5, n_b)
        x0_b_record = "随机值（每个神经元独立）"
    else:
        # 传值 → b群体所有神经元共用该值
        x_series[n_a:, 0] = x0_b
        x0_b_record = x0_b
    
    # 处理a群体y初始值
    if y0_a is None:
        y_series[:n_a, 0] = np.random.uniform(-4, -3, n_a)
        y0_a_record = "随机值（每个神经元独立）"
    else:
        y_series[:n_a, 0] = y0_a
        y0_a_record = y0_a
    
    # 处理b群体y初始值
    if y0_b is None:
        y_series[n_a:, 0] = np.random.uniform(-4, -3, n_b)
        y0_b_record = "随机值（每个神经元独立）"
    else:
        y_series[n_a:, 0] = y0_b
        y0_b_record = y0_b
    
    # 定义分段函数 f (局部动力学)
    def f(x_val, y_val, alpha):
        """Chialvo神经元模型的非线性函数"""
        if x_val <= 0:
            return alpha / (1 - x_val) + y_val
        elif 0 < x_val < alpha + y_val:
            return alpha + y_val
        else:
            return -1
    
    # 定义函数 g (慢变量动力学)
    def g(x_val, y_val, mu, sigma):
        """慢变量的更新函数"""
        return y_val - mu * (x_val + 1) + mu * sigma
    
    # 模拟迭代
    for t in range(T-1):
        # 计算两个群体的平均场
        # a群体的平均场
        Xbar_a = np.mean(x_series[:n_a, t])
        # b群体的平均场
        Xbar_b = np.mean(x_series[n_a:, t])
        
        # 更新a群体的神经元
        for i in range(n_a):
            # 局部动力学部分
            local_part = f(x_series[i, t], y_series[i, t], alpha_all[i])
            
            # 根据公式: x_{t+1} = (1-γ)f(...) + γXbar_a + εXbar_b
            x_series[i, t+1] = (1 - gamma) * local_part + gamma * Xbar_a + epsilon * Xbar_b
            
            # 根据公式: y_{t+1} = g(x_t, y_t)
            y_series[i, t+1] = g(x_series[i, t], y_series[i, t], mu, sigma_all[i])
        
        # 更新b群体的神经元
        for j in range(n_b):
            idx = n_a + j  # 在总数组中的索引
            
            # 局部动力学部分
            local_part = f(x_series[idx, t], y_series[idx, t], alpha_all[idx])
            
            # 根据公式: x_{t+1} = (1-γ)f(...) + γXbar_b + εXbar_a
            x_series[idx, t+1] = (1 - gamma) * local_part + gamma * Xbar_b + epsilon * Xbar_a
            
            # 根据公式: y_{t+1} = g(x_t, y_t)
            y_series[idx, t+1] = g(x_series[idx, t], y_series[idx, t], mu, sigma_all[idx])
        
    # 计算同步指标
    def compute_instantaneous_std(x_data, mean_field):
        N_neurons = x_data.shape[0]
        T_steps = x_data.shape[1]
        r_t = np.zeros(T_steps)
        
        for t in range(T_steps):
            # 计算每个时刻的标准差
            r_t[t] = np.sqrt(np.mean((x_data[:, t] - mean_field[t])**2))
        
        return r_t
    
    # 计算平均场时间序列
    Xbar_a = np.zeros(T)
    Xbar_b = np.zeros(T)
    
    for t in range(T):
        Xbar_a[t] = np.mean(x_series[:n_a, t])
        Xbar_b[t] = np.mean(x_series[n_a:, t])
    Xbar_a_transient = Xbar_a[transients:]
    Xbar_b_transient = Xbar_b[transients:]
    # 计算瞬时标准差
    r_a_t = compute_instantaneous_std(x_series[:n_a, :], Xbar_a)
    r_b_t = compute_instantaneous_std(x_series[n_a:, :], Xbar_b)
    r_a_t_transient = r_a_t[transients:]
    r_b_t_transient = r_b_t[transients:]
    
    Xbar_all = np.zeros(T)
    for t in range(T):
        Xbar_all[t] = np.mean(x_series[:, t])
        
    r_t = compute_instantaneous_std(x_series, Xbar_all)
    r_t_transient = r_t[transients:]
    
    # 计算时间平均
    T_effective = T - transients
    R_a = np.mean(r_a_t_transient) 
    R_b = np.mean(r_b_t_transient) 
    R_t = np.mean(r_t_transient) 
    R_delta = np.mean(np.abs(Xbar_a[transients:] - Xbar_b[transients:]))  
    
    # 判断同步状态 (基于图片中的定义)
    def determine_sync_state(R_a, R_b, R_delta, threshold=1e-7):
        sync_a = R_a < threshold
        sync_b = R_b < threshold
        sync_ab = R_delta < threshold
        
        if sync_a and sync_b and sync_ab:
            return "Complete Synchronization 完全同步(CS)"
        elif sync_a and sync_b and not sync_ab:
            return "Generalized Synchronization 广义同步 (GS)"
        elif (sync_a and not sync_b) or (not sync_a and sync_b):
            return "Chimera State 奇美拉态(Q)"
        elif not sync_a and not sync_b:
            return "Desynchronization 去同步化(D)"
        else:
            return "Unknown State 未知"
    
    sync_state = determine_sync_state(R_a, R_b, R_delta)
    
    # 准备参数信息
    params = {
        '神经元总数': N,
        'a群体神经元数': n_a,
        'b群体神经元数': n_b,
        'a群体α参数': alpha_a[0] if isinstance(alpha_a, np.ndarray) and len(np.unique(alpha_a)) == 1 else f"列表({len(alpha_a)}个)",
        'b群体α参数': alpha_b[0] if isinstance(alpha_b, np.ndarray) and len(np.unique(alpha_b)) == 1 else f"列表({len(alpha_b)}个)",
        'a群体σ参数': sigma_a[0] if isinstance(sigma_a, np.ndarray) and len(np.unique(sigma_a)) == 1 else f"列表({len(sigma_a)}个)",
        'b群体σ参数': sigma_b[0] if isinstance(sigma_b, np.ndarray) and len(np.unique(sigma_b)) == 1 else f"列表({len(sigma_b)}个)",
        '慢变参数μ': mu,
        '群体内耦合强度γ': gamma,
        '群体间耦合强度ε': epsilon,
        '总步数': T,
        '舍弃暂态': transients,
        '有效数据长度': T_effective,
        '随机种子': seed
    }
    
    # 动力学特征
    dynamics_stats = {
        '⟨σa⟩ (R_a)': R_a,
        '⟨σb⟩ (R_b)': R_b,
        '⟨σt⟩ (R_t)': R_t,
        '⟨δ⟩ (R_delta)': R_delta,
        '同步状态': sync_state,
        'a群体同步': R_a < 1e-7,
        'b群体同步': R_b < 1e-7,
        'Rt群体同步': R_t < 1e-7,
        '群体间同步': R_delta < 1e-7,
        'r_a_t': r_a_t,
        'r_b_t': r_b_t,
        'r_t': r_t
    }
    # 准备返回数据
    data = {
        'params': {**params, **dynamics_stats},
        '群体信息': {
            'a群体神经元数': n_a,
            'b群体神经元数': n_b,
            'a群体索引': list(range(n_a)),
            'b群体索引': list(range(n_a, n_a + n_b)),
            'a群体α参数': alpha_a,
            'b群体α参数': alpha_b,
            'a群体σ参数': sigma_a,
            'b群体σ参数': sigma_b
        },
        '时间序列': {
            't': np.arange(T_effective),
            'Xbar_a': Xbar_a,  # a群体平均场
            'Xbar_b': Xbar_b,  # b群体平均场
            'Xbar_a_transient': Xbar_a_transient,  # a群体平均场
            'Xbar_b_transient': Xbar_b_transient,  # b群体平均场
            'r_a_t': r_a_t,   # a群体瞬时标准差
            'r_b_t': r_b_t,   # b群体瞬时标准差
            'r_t': r_t,   # 全部瞬时标准差
            'r_a_t_transient': r_a_t_transient,   # a群体瞬时标准差
            'r_b_t_transient': r_b_t_transient,   # b群体瞬时标准差
            'r_t_transient': r_t_transient   # 全部瞬时标准差
        },
        '同步指标': {
            'R_a': R_a,      # ⟨σα⟩的时间平均
            'R_b': R_b,      # ⟨σβ⟩的时间平均
            'R_t': R_t,      
            'R_delta': R_delta,  # ⟨δ⟩的时间平均
            'sync_state': sync_state
        }
    }
    
    # 添加每个神经元的完整时间序列
    for i in range(n_a):
        data[f'神经元_a_{i+1:03d}'] = {
            'x_transient': x_series[i, transients:],
            'y_transient': y_series[i, transients:],
            'x': x_series[i],
            'y': y_series[i],
            '群体': 'a',
            'α参数': alpha_a[i] if isinstance(alpha_a, np.ndarray) else alpha_a,
            'σ参数': sigma_a[i] if isinstance(sigma_a, np.ndarray) else sigma_a
        }
    
    for j in range(n_b):
        idx = n_a + j
        data[f'神经元_b_{j+1:03d}'] = {
            'x_transient': x_series[idx, transients:],
            'y_transient': y_series[idx, transients:],
            'x': x_series[idx],
            'y': y_series[idx],
            '群体': 'b',
            'α参数': alpha_b[j] if isinstance(alpha_b, np.ndarray) else alpha_b,
            'σ参数': sigma_b[j] if isinstance(sigma_b, np.ndarray) else sigma_b
        }
    
    return data

def plot_neuron_data(data, title="两群体神经元动态行为", figsize=(18, 10), 
                    show_last_n=None, neuron_sample=None):
    """
    绘制两群体神经元数据。
    
    参数:
        data: 生成的数据字典
        title: 图标题
        figsize: 图形大小
        show_last_n: 只显示最后n个时间点（避免过于拥挤），如果为None则显示全部
        neuron_sample: 每个群体显示的神经元数量，如果为None则显示全部
    """
    # 从数据中提取参数
    params = data['params']
    group_info = data['群体信息']
    time_series = data['时间序列']
    sync_info = data['同步指标']
    
    n_a = group_info['a群体神经元数']
    n_b = group_info['b群体神经元数']
    n_neurons = n_a + n_b
    
    # 时间点
    t = time_series['t']
    
    # 如果指定了show_last_n，只显示最后的部分
    if show_last_n is not None and show_last_n < len(t):
        start_idx = len(t) - show_last_n
        t_display = t[start_idx:]
    else:
        start_idx = 0
        t_display = t
    
    # 确定要绘制的神经元
    if neuron_sample is not None:
        # 从每个群体中随机选择指定数量的神经元
        np.random.seed(42)  # 固定随机种子以保证可重复性
        a_indices = np.random.choice(n_a, min(neuron_sample, n_a), replace=False)
        b_indices = np.random.choice(n_b, min(neuron_sample, n_b), replace=False)
    else:
        a_indices = list(range(n_a))
        b_indices = list(range(n_b))
    
    # 定义颜色列表 - 使用美观的调色板
    a_colors = ['#FF6B6B', '#FF8E8E', '#FFB2B2', '#FFD6D6']  # 暖色调
    b_colors = ['#4C72B0', '#6A8CC7', '#8CA6DE', '#AEC0F5']  # 冷色调
    avg_colors = ['#C44E52', '#2E5A8C']  # 平均场颜色
    
    # 创建图形
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], hspace=0.25, wspace=0.3)
    
    # 第一行：所有神经元的x随时间变化（长子图，跨3列）
    ax1 = fig.add_subplot(gs[0, :])
    
    # 绘制a群体神经元的x-t曲线
    for idx, i in enumerate(a_indices):
        neuron_key = f'神经元_a_{i+1:03d}'
        x = data[neuron_key]['x_transient']
        if start_idx > 0:
            x = x[start_idx:]
        color_idx = idx % len(a_colors)
        ax1.plot(t_display, x, color=a_colors[color_idx], linewidth=0.8, 
                alpha=0.9, label=f'神经元_a_{i+1}' if idx < 5 else None)
    
    # 绘制b群体神经元的x-t曲线
    for idx, j in enumerate(b_indices):
        neuron_key = f'神经元_b_{j+1:03d}'
        x = data[neuron_key]['x_transient']
        if start_idx > 0:
            x = x[start_idx:]
        color_idx = idx % len(b_colors)
        ax1.plot(t_display, x, color=b_colors[color_idx], linewidth=0.8, 
                alpha=0.9, label=f'神经元_b_{j+1}' if idx < 5 else None)
    
    # 绘制平均场
    Xbar_a = time_series['Xbar_a_transient']
    Xbar_b = time_series['Xbar_b_transient']
    if start_idx > 0:
        Xbar_a = Xbar_a[start_idx:]
        Xbar_b = Xbar_b[start_idx:]
    
    ax1.plot(t_display, Xbar_a, color=avg_colors[0], linewidth=2.5, alpha=0.9, label='Xbar_a (α群体平均场)')
    ax1.plot(t_display, Xbar_b, color=avg_colors[1], linewidth=2.5, alpha=0.9, label='Xbar_b (β群体平均场)')
    
    ax1.set_xlabel('时间（迭代步）', fontsize=12)
    ax1.set_ylabel('快变量 x', fontsize=12)
    ax1.set_title('快变量x随时间变化', fontsize=14, fontweight='bold')
    
    # 智能图例：显示有限数量的神经元标签
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # 只显示前6个神经元标签和平均场标签
    legend_items = []
    legend_labels = []
    
    # 添加前几个神经元标签
    for i, label in enumerate(unique_labels):
        if '神经元_a' in label and len([l for l in legend_labels if '神经元_a' in l]) < 2:
            legend_items.append(unique_handles[i])
            legend_labels.append(label)
        elif '神经元_b' in label and len([l for l in legend_labels if '神经元_b' in l]) < 2:
            legend_items.append(unique_handles[i])
            legend_labels.append(label)
        elif '平均场' in label:
            legend_items.append(unique_handles[i])
            legend_labels.append(label)
    
    ax1.legend(legend_items, legend_labels, loc='upper right', fontsize=10, ncol=2)
    ax1.set_facecolor('white')  
    ax1.grid(False) 
    
    # 添加同步状态信息
    sync_state = sync_info.get('sync_state', 'Unknown')
    ax1.text(0.02, 0.98, f'同步状态: {sync_state}', transform=ax1.transAxes, 
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='gray'))
    
    # 第二行第一列：所有神经元的y随时间变化
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 绘制a群体神经元的y-t曲线
    for idx, i in enumerate(a_indices):
        neuron_key = f'神经元_a_{i+1:03d}'
        y = data[neuron_key]['y_transient']
        if start_idx > 0:
            y = y[start_idx:]
        color_idx = idx % len(a_colors)
        ax2.plot(t_display, y, color=a_colors[color_idx], linewidth=0.8, 
                alpha=0.7, label=f'神经元_a_{i+1}' if idx < 3 else None)
    
    # 绘制b群体神经元的y-t曲线
    for idx, j in enumerate(b_indices):
        neuron_key = f'神经元_b_{j+1:03d}'
        y = data[neuron_key]['y_transient']
        if start_idx > 0:
            y = y[start_idx:]
        color_idx = idx % len(b_colors)
        ax2.plot(t_display, y, color=b_colors[color_idx], linewidth=0.8, 
                alpha=0.7, label=f'神经元_b_{j+1}' if idx < 3 else None)
    
    ax2.set_xlabel('时间（迭代步）', fontsize=12)
    ax2.set_ylabel('慢变量 y', fontsize=12)
    ax2.set_title('慢变量y随时间变化', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.set_facecolor('white')  
    ax2.grid(False) 
    
    # 第二行第二列：所有神经元的相平面图
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 绘制a群体神经元的相平面图
    for idx, i in enumerate(a_indices):
        neuron_key = f'神经元_a_{i+1:03d}'
        x = data[neuron_key]['x_transient']
        y = data[neuron_key]['y_transient']
        if start_idx > 0:
            x = x[start_idx:]
            y = y[start_idx:]
        color_idx = idx % len(a_colors)
        ax3.plot(x, y, color=a_colors[color_idx], linewidth=0.8, alpha=0.7, 
                label=f'神经元_a_{i+1}' if idx < 3 else None)
        # 标记起点和终点
        ax3.scatter(x[0], y[0], color=a_colors[color_idx], s=20, zorder=5, 
                   marker='o', alpha=0.8)
        ax3.scatter(x[-1], y[-1], color=a_colors[color_idx], s=20, zorder=5, 
                   marker='s', alpha=0.8)
    
    # 绘制b群体神经元的相平面图
    for idx, j in enumerate(b_indices):
        neuron_key = f'神经元_b_{j+1:03d}'
        x = data[neuron_key]['x_transient']
        y = data[neuron_key]['y_transient']
        if start_idx > 0:
            x = x[start_idx:]
            y = y[start_idx:]
        color_idx = idx % len(b_colors)
        ax3.plot(x, y, color=b_colors[color_idx], linewidth=0.8, alpha=0.7, 
                label=f'神经元_b_{j+1}' if idx < 3 else None)
        # 标记起点和终点
        ax3.scatter(x[0], y[0], color=b_colors[color_idx], s=20, zorder=5, 
                   marker='o', alpha=0.8)
        ax3.scatter(x[-1], y[-1], color=b_colors[color_idx], s=20, zorder=5, 
                   marker='s', alpha=0.8)
    
    ax3.set_xlabel('快变量 x', fontsize=12)
    ax3.set_ylabel('慢变量 y', fontsize=12)
    ax3.set_title('相平面图 (x-y)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.set_facecolor('white')  
    ax3.grid(False) 
    
    # 第二行第三列：参数信息
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # 构建参数文本
    param_text = "系统参数\n"
    param_text += "="*30 + "\n"
    
    # 同步状态信息
    param_text += f"同步状态: {sync_state}\n"
    param_text += f"σa (R_a) = {sync_info.get('R_a', 0):.4e}\n"
    param_text += f"σb (R_b) = {sync_info.get('R_b', 0):.4e}\n"
    param_text += f"δ (R_delta) = {sync_info.get('R_delta', 0):.4e}\n"
    param_text += "="*30 + "\n"
    
    # 神经元数量
    param_text += f"神经元总数: {params.get('神经元总数', 'N/A')}\n"
    param_text += f"α群体神经元数: {n_a}\n"
    param_text += f"β群体神经元数: {n_b}\n"
    param_text += "-"*30 + "\n"
    
    # 可激发性参数
    param_text += "可激发性参数:\n"
    if isinstance(group_info['a群体α参数'], (int, float, np.number)):
        param_text += f"  α_a = {group_info['a群体α参数']}\n"
    else:
        param_text += f"  α_a = 列表({len(group_info['a群体α参数'])}个)\n"
    
    if isinstance(group_info['b群体α参数'], (int, float, np.number)):
        param_text += f"  α_b = {group_info['b群体α参数']}\n"
    else:
        param_text += f"  α_b = 列表({len(group_info['b群体α参数'])}个)\n"
    
    # 异质性参数
    param_text += "异质性参数:\n"
    if isinstance(group_info['a群体σ参数'], (int, float, np.number)):
        param_text += f"  σ_a = {group_info['a群体σ参数']}\n"
    else:
        param_text += f"  σ_a = 列表({len(group_info['a群体σ参数'])}个)\n"
    
    if isinstance(group_info['b群体σ参数'], (int, float, np.number)):
        param_text += f"  σ_b = {group_info['b群体σ参数']}\n"
    else:
        param_text += f"  σ_b = 列表({len(group_info['b群体σ参数'])}个)\n"
    
    param_text += "-"*30 + "\n"
    
    # 动力学参数
    param_text += "动力学参数:\n"
    param_text += f"  慢变参数 μ = {params.get('慢变参数μ', 'N/A')}\n"
    param_text += f"  群体内耦合 γ = {params.get('群体内耦合强度γ', 'N/A')}\n"
    param_text += f"  群体间耦合 ε = {params.get('群体间耦合强度ε', 'N/A')}\n"
    
    param_text += "-"*30 + "\n"
    
    # 模拟参数
    param_text += "模拟参数:\n"
    param_text += f"  总步数: {params.get('总步数', 'N/A')}\n"
    param_text += f"  舍弃暂态: {params.get('舍弃暂态', 'N/A')}步\n"
    param_text += f"  有效数据: {params.get('有效数据长度', 'N/A')}步\n"
    param_text += f"  随机种子: {params.get('随机种子', 'N/A')}\n"
    
    # 计算合适的字体大小
    font_size = 9 if n_neurons > 20 else 10
    
    # 添加文本背景
    ax4.text(0.05, 0.98, param_text, fontsize=font_size, 
            verticalalignment='top', linespacing=1.5,
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', 
                     alpha=0.9, edgecolor='#D3D3D3'))
    
    plt.suptitle(f'{title}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

def plot_neuron_state_heatmap(data, figsize=(15, 8), time_window=None, add_neuron_divider=False,
                             vmin=None, vmax=None, cmap='viridis'):
    """
    绘制神经元状态热力图
    
    参数:
        data: generate_two_population_neuron_data函数返回的数据字典
        figsize: 图形大小
        time_window: 时间窗口，如[0, 1000]表示只显示0-1000时间步的数据
        vmin, vmax: 颜色映射的最小值和最大值
        cmap: 颜色映射
    """
    # 从数据中提取信息
    params = data['params']
    group_info = data['群体信息']
    sync_info = data['同步指标']
    
    n_a = group_info['a群体神经元数']
    n_b = group_info['b群体神经元数']
    N = n_a + n_b
    T_effective = params.get('有效数据长度', 1000)
    
    # 提取所有神经元的x值
    x_data = np.zeros((N, T_effective))
    
    # 提取a群体神经元的x值
    for i in range(n_a):
        neuron_key = f'神经元_a_{i+1:03d}'
        x_data[i, :] = data[neuron_key]['x_transient']
    
    # 提取b群体神经元的x值
    for j in range(n_b):
        neuron_key = f'神经元_b_{j+1:03d}'
        idx = n_a + j
        x_data[idx, :] = data[neuron_key]['x_transient']
    
    # 如果指定了时间窗口
    if time_window is not None:
        start_t, end_t = time_window
        if end_t > T_effective:
            end_t = T_effective
        x_data = x_data[:, start_t:end_t]
        t_display = np.arange(end_t - start_t)
    else:
        t_display = np.arange(T_effective)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确定颜色映射范围
    if vmin is None:
        vmin = np.min(x_data)
    if vmax is None:
        vmax = np.max(x_data)
    
    # 绘制热力图
    im = ax.imshow(x_data, aspect='auto', cmap=cmap, 
                   extent=[0, len(t_display), 0, N],
                   vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
    cbar.set_label('神经元状态 x 值', fontsize=12)
    
    # 设置坐标轴
    ax.set_xlabel('时间步 t', fontsize=12)
    ax.set_ylabel('神经元索引', fontsize=12)
    
    # 设置y轴刻度，只显示群体标签
    ax.set_yticks([n_a/2, n_a + n_b/2])
    ax.set_yticklabels(['群体a', '群体b'], fontsize=12)
    
    # 在两个群体之间添加分隔线
    ax.axhline(y=n_a, color='white', linestyle='--', linewidth=2, alpha=0.8)
    if add_neuron_divider:
        # 在a群体内添加神经元分隔线
        for i in range(1, n_a):
            ax.axhline(y=i, color='white', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # 在b群体内添加神经元分隔线
        for i in range(1, n_b):
            ax.axhline(y=n_a + i, color='white', linestyle=':', linewidth=0.5, alpha=0.3)
            
    # 设置标题
    sync_state = sync_info.get('sync_state', 'Unknown')
    title = f'神经元群体状态热力图 (同步状态: {sync_state})'
    if time_window is not None:
        title += f' (时间窗口: {time_window[0]}-{time_window[1]})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_neuron_analysis_combo(data, figsize=(20, 8), time_window=None, 
                               vmin=None, vmax=None, cmap='viridis'):
    """
    绘制神经元分析组合图：左边热力图，右边瞬时标准差时间序列
    
    参数:
        data: generate_two_population_neuron_data函数返回的数据字典
        figsize: 图形大小
        time_window: 时间窗口
        vmin, vmax: 颜色映射的最小值和最大值
        cmap: 颜色映射
        show_transient: 是否显示暂态部分
    """
    # 从数据中提取信息
    params = data['params']
    group_info = data['群体信息']
    sync_info = data['同步指标']
    time_series = data['时间序列']
    
    n_a = group_info['a群体神经元数']
    n_b = group_info['b群体神经元数']
    N = n_a + n_b
    T_total = params.get('总步数', 10000)
    transients = params.get('舍弃暂态', 1000)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 左图：热力图 ==========
    T_effective = T_total - transients
    # 提取所有神经元的x值
    x_data = np.zeros((N, T_effective))
    
    # 提取a群体神经元的x值
    for i in range(n_a):
        neuron_key = f'神经元_a_{i+1:03d}'
        x_data[i, :] = data[neuron_key]['x_transient']
    
    # 提取b群体神经元的x值
    for j in range(n_b):
        neuron_key = f'神经元_b_{j+1:03d}'
        idx = n_a + j
        x_data[idx, :] = data[neuron_key]['x_transient']
    
    # 如果指定了时间窗口
    if time_window is not None:
        start_t, end_t = time_window
        if end_t > T_effective:
            end_t = T_effective
        x_data = x_data[:, start_t:end_t]
        t_display = np.arange(end_t - start_t)
    else:
        t_display = np.arange(T_effective)
        start_t, end_t = 0, T_effective
    
    # 确定颜色映射范围
    if vmin is None:
        vmin = np.min(x_data)
    if vmax is None:
        vmax = np.max(x_data)
    
    # 绘制热力图
    im = ax1.imshow(x_data, aspect='auto', cmap=cmap, 
                   extent=[0, len(t_display), 0, N],
                   vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1, pad=0.01, shrink=0.8)
    cbar.set_label('神经元状态 x 值', fontsize=12)
    
    # 设置坐标轴
    ax1.set_xlabel('时间步 t', fontsize=12)
    ax1.set_ylabel('神经元索引', fontsize=12)
    
    # 设置y轴刻度，只显示群体标签
    ax1.set_yticks([n_a/2, n_a + n_b/2])
    ax1.set_yticklabels(['群体a', '群体b'], fontsize=12)
    
    # 在两个群体之间添加分隔线
    ax1.axhline(y=n_a, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # 设置标题
    sync_state = sync_info.get('sync_state', 'Unknown')
    title1 = f'神经元群体状态热力图 (同步状态: {sync_state})'
    if time_window is not None:
        title1 += f' (时间窗口: {time_window[0]}-{time_window[1]})'
    ax1.set_title(title1, fontsize=14, fontweight='bold', pad=20)
    
    # ========== 右图：瞬时标准差时间序列 ==========
    # 获取瞬时标准差数据
    r_a_t = time_series['r_a_t']
    r_b_t = time_series['r_b_t']
    r_t = time_series['r_t']
    
    t_plot = np.arange(T_total)
    transient_line = transients
    
    # 绘制三条曲线
    ax2.plot(t_plot, r_a_t, label='r_a(t)', 
         color='blue', alpha=0.8, linewidth=1.5, linestyle=':', marker='.', markersize=5,markevery=50)  # 点线
    ax2.plot(t_plot, r_b_t, label='r_b(t)', 
         color='red', alpha=0.8, linewidth=1.5, linestyle=':', marker='s', markersize=2,markevery=50)  # 方框线
    ax2.plot(t_plot, r_t, label='r(t)', 
         color='green', alpha=0.8, linewidth=1.5, linestyle='--')  # 虚线
    
    ax2.axvline(x=transient_line, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # 设置坐标轴
    ax2.set_xlabel('时间步 t', fontsize=12)
    ax2.set_ylabel('瞬时标准差', fontsize=12)
    ax2.set_title('瞬时标准差时间序列', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11)
    ax2.set_facecolor('white')  
    ax2.grid(False)  
    
    # 调整布局
    ax1.grid(False)
    plt.tight_layout()
    plt.show()

def plot_sync_statistics_histogram(statistics_df, figsize=(12,6)):
    """
    绘制同步统计指标的柱状图
    
    参数:
        statistics_df: 包含统计数据的DataFrame
        figsize: 图形大小
    """
    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 提取数据
    R_a_values = statistics_df['R_a'].values
    R_b_values = statistics_df['R_b'].values
    R_delta_values = statistics_df['R_delta'].values
    
    # 确定合适的bins数量
    bins = 20
    
    # 绘制R_a柱状图
    axes[0].hist(R_a_values, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(R_a_values), color='red', linestyle='--', linewidth=2, 
                    label=f'均值: {np.mean(R_a_values):.2e}')
    axes[0].set_xlabel('R_a值', fontsize=12)
    axes[0].set_ylabel('频次', fontsize=12)
    axes[0].set_title('群体a同步指标R_a分布', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制R_b柱状图
    axes[1].hist(R_b_values, bins=bins, color='red', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(R_b_values), color='blue', linestyle='--', linewidth=2, 
                    label=f'均值: {np.mean(R_b_values):.2e}')
    axes[1].set_xlabel('R_b值', fontsize=12)
    axes[1].set_ylabel('频次', fontsize=12)
    axes[1].set_title('群体b同步指标R_b分布', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].set_facecolor('white')  
    axes[1].grid(False) 
    
    # 绘制R_delta柱状图
    axes[2].hist(R_delta_values, bins=bins, color='green', alpha=0.7, edgecolor='black')
    axes[2].axvline(x=np.mean(R_delta_values), color='purple', linestyle='--', linewidth=2, 
                    label=f'均值: {np.mean(R_delta_values):.2e}')
    axes[2].set_xlabel('R_delta值', fontsize=12)
    axes[2].set_ylabel('频次', fontsize=12)
    axes[2].set_title('群体间差异指标R_delta分布', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].set_facecolor('white')  
    axes[2].grid(False) 
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def analyze_sync_states(statistics_df):
    """
    分析同步状态统计
    
    参数:
        statistics_df: 包含统计数据的DataFrame
        
    返回:
        dict: 同步状态统计结果
    """
    # 统计各种同步状态的数量
    sync_counts = Counter(statistics_df['sync_state'])
    
    # 计算百分比
    total = len(statistics_df)
    sync_percentages = {state: count/total*100 for state, count in sync_counts.items()}
    
    # 创建统计结果字典
    analysis_results = {
        '总数': total,
        '同步状态计数': dict(sync_counts),
        '同步状态百分比': sync_percentages,
        '主要同步状态': max(sync_counts, key=sync_counts.get) if sync_counts else "无数据"
    }
    
    # 打印分析结果
    print("="*60)
    print("同步状态统计分析")
    print("="*60)
    print(f"总运行次数: {total}")
    print("\n同步状态分布:")
    for state, count in sync_counts.items():
        percentage = sync_percentages[state]
        print(f"  {state}: {count}次 ({percentage:.2f}%)")
    
    print(f"\n主要同步状态: {analysis_results['主要同步状态']}")
    print("="*60)
    
    return analysis_results



