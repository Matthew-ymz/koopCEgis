"""
动力学系统数据生成器的基类
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem(ABC):
    """动力学系统的抽象基类"""
    
    def __init__(self):
        """初始化动力学系统"""
        self.name = "Dynamical System"
        self.dim = 0  # 系统维度
        self.parameters = {}
    
    @abstractmethod
    def _derivatives(self, t, state):
        """
        计算系统在给定状态下的导数
        
        Parameters:
        -----------
        t : float
            时间
        state : array-like
            当前状态向量
            
        Returns:
        --------
        derivatives : array-like
            状态向量的导数
        """
        pass
    
    @abstractmethod
    def get_default_initial_conditions(self):
        """
        获取默认初始条件
        
        Returns:
        --------
        initial_state : array-like
            初始状态向量
        """
        pass
    
    def generate_data(self, t_span=(0, 50), n_points=5000, 
                     initial_conditions=None, noise_level=0.0):
        """
        生成时间序列数据
        
        Parameters:
        -----------
        t_span : tuple
            时间范围 (t_start, t_end)
        n_points : int
            采样点数
        initial_conditions : array-like, optional
            初始条件，如果为None则使用默认值
        noise_level : float
            添加到数据的高斯噪声标准差
            
        Returns:
        --------
        t : ndarray
            时间数组，shape (n_points,)
        x : ndarray
            状态数据，shape (n_points, dim)
        """
        if initial_conditions is None:
            initial_conditions = self.get_default_initial_conditions()
        
        # 生成时间点
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # 求解 ODE
        solution = solve_ivp(
            self._derivatives,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )
        
        # 转置使形状为 (n_points, dim)
        x = solution.y.T
        
        # 添加噪声（如果需要）
        if noise_level > 0:
            x += np.random.normal(0, noise_level, x.shape)
        
        return t_eval, x
    
    def get_equations_text(self):
        """
        获取系统方程的文本描述
        
        Returns:
        --------
        equations : str
            方程的文本描述
        """
        return "System equations not specified"
    
    def get_parameters(self):
        """
        获取系统参数
        
        Returns:
        --------
        parameters : dict
            系统参数字典
        """
        return self.parameters.copy()
    
    def __str__(self):
        """返回系统的字符串表示"""
        return f"{self.name} (dim={self.dim})"
