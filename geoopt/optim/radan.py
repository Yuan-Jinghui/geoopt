import torch
from torch.optim import Adam
from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor

__all__ = ["RiemannianAdan"]

class RiemannianAdan(OptimMixin, Adam):
    r"""
    Riemannian Adan with the same API as :class:`torch.optim.Adam`,
    but replacing Adam 的更新为 Adan（Adaptive Nesterov Momentum）。
    """
    def __init__(self, params, lr=1e-3, betas=(0.02, 0.1, 0.001), eps=1e-8,
                 weight_decay=0, stabilize=None):
        """
        参数说明同 Adam，betas=(β1,β2,β3) 分别对应 Adan 中的一阶动量、
        差分动量和二阶动量衰减系数。
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=False)
        self.stabilize = stabilize

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            β1, β2, β3 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for point in group["params"]:
                grad = point.grad
                if grad is None:
                    continue

                # 1. 选择流形：支持 ManifoldParameter/ManifoldTensor，否则用默认流形
                if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                    manifold = point.manifold
                else:
                    manifold = self._default_manifold

                if grad.is_sparse:
                    raise RuntimeError("RiemannianAdan 不支持稀疏梯度")

                state = self.state[point]
                # State 初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(point)       # m_t
                    state["exp_avg_diff"] = torch.zeros_like(point)  # 存储 g_{t-1}
                    state["exp_avg_sq"] = torch.zeros_like(point)    # n_t

                state["step"] += 1
                t = state["step"]

                # 2. 欧氏梯度 => 黎曼梯度
                grad = grad.add(point, alpha=wd)  # decoupled weight decay
                rgrad = manifold.egrad2rgrad(point, grad)  # :contentReference[oaicite:0]{index=0}

                # 3. 计算 Nesterov 动量修正后的梯度 g'_t
                g_prev = state["exp_avg_diff"]
                g_prime = rgrad.add(rgrad - g_prev, alpha=(1 - β1))

                # 4. 更新一阶动量和二阶动量
                exp_avg     = state["exp_avg"]
                exp_avg_sq  = state["exp_avg_sq"]

                exp_avg.mul_(β1).add_(g_prime, alpha=1 - β1)  # m_t ← β1 m_{t-1} + (1-β1)g′_t
                # component_inner 返回切空间内积 g′_t ∘ g′_t
                v_t = manifold.component_inner(point, g_prime)  # :contentReference[oaicite:1]{index=1}
                exp_avg_sq.mul_(β3).add_(v_t,          alpha=1 - β3)

                # 5. 偏置校正
                bias1 = 1 - β1**t
                bias3 = 1 - β3**t
                m_hat  = exp_avg    .div(bias1)
                v_hat  = exp_avg_sq .div(bias3)

                # 6. 计算复合步长和更新方向
                step1 = m_hat.div(v_hat.sqrt().add(eps))
                step2 = (β2 / bias1) * (exp_avg.sub(m_hat)) / (v_hat.sqrt().add(eps))
                direction = step1.add(step2)

                # 7. 重traction + 并行运输 m_t
                # retr_transp: 一次性完成 retr 和 transport
                new_point, m_trans = manifold.retr_transp(
                    point, -lr * direction, exp_avg
                )  # :contentReference[oaicite:2]{index=2}

                # 8. 写回参数与状态
                point.copy_(new_point)
                state["exp_avg"]       = m_trans
                state["exp_avg_diff"]  = rgrad.clone()  # 用于下一步计算 g_{t-1}
                state["exp_avg_sq"]    = exp_avg_sq

                # 9. 可选：稳态校正
                if self.stabilize and t % self.stabilize == 0:
                    point.copy_(manifold.projx(point))
                    state["exp_avg"].copy_(manifold.proju(point, state["exp_avg"]))

        return loss
