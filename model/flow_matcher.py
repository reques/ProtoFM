import torch


class FlowMatcher:
    """
    此类实现了"最优传输"（OT）路径或"整流流"（Rectified Flow）的
    核心逻辑，用于流匹配。

    它定义了一条从 x0 (噪声) 到 x1 (数据) 的直线路径：
    x_t = (1-t) * x0 + t * x1

    该路径上的目标向量场 (速度) 是恒定的：
    u_t = x1 - x0
    """

    def __init__(self, sigma_min=1e-4):
        """
        Args:
            sigma_min: 一个非常小的值，用于避免 t=0 或 t=1 时的数值不稳定。
        """
        self.sigma_min = sigma_min

    def get_train_tuple(self, x0, x1, t):
        """
        给定 x0 (噪声), x1 (数据) 和 t (时间)，计算 x_t 和 u_t。

        Args:
            x0 (torch.Tensor): 噪声样本, 形状 (B, ...)。
            x1 (torch.Tensor): 真实数据样本, 形状 (B, ...)。
            t (torch.Tensor): 随机时间步, 形状 (B,)。

        Returns:
            x_t (torch.Tensor): 路径上的插值点 (B, ...)。
            u_t (torch.Tensor): 目标向量场 (B, ...)。
        """

        # 将 t 的形状从 (B,) 调整为 (B, 1, 1) 以便广播
        # 假设 x0/x1 的形状是 (B, 1, C_in)
        t_broadcast = t.view(-1, 1, 1)

        # 1. 计算插值点 x_t
        # x_t = (1-t) * x0 + t * x1
        # x_t = (1 - (1-self.sigma_min)*t_broadcast) * x0 + t_broadcast * x1
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1

        # 2. 计算目标向量场 u_t
        u_t = x1 - x0
        # u_t = x1 - (1-self.sigma_min)*x0

        return x_t, u_t

    def sample_t(self, shape, device):
        """
        为训练采样随机时间 t。
        在 [sigma_min, 1.0] 范围内均匀采样。
        """
        return torch.rand(shape, device=device) * (1.0 - self.sigma_min) + self.sigma_min