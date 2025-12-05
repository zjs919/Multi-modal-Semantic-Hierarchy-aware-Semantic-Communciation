import torch
import torch.distributions as dist

def sample_from_multivariate_normal(params):
    """
    从多元正态分布中采样，并返回采样值及其对应的概率密度。

    参数:
    params (torch.Tensor): 2*N维的输入向量，前N维是均值，后N维是标准差。

    返回:
    samples (torch.Tensor): N个采样值。
    log_probs (torch.Tensor): N个采样值的对数概率密度。
    """
    N = params.shape[-1] // 2  # 获取N的值
    mean = params[:, :N]  # 前N维是均值
    std_dev = torch.exp(params[:, N:])  # 后N维是标准差，取指数将其转为正数

    # 创建N个正态分布
    normal_distributions = dist.Normal(mean, std_dev)

    # 从每个正态分布中采样
    samples = normal_distributions.sample()

    # 计算每个采样值的对数概率密度
    log_probs = normal_distributions.log_prob(samples)

    return samples, log_probs

def predict_from_option_and_action(option, action):
    phase_action, mod_action = torch.chunk(action, 2, dim=-1)
    prediction = torch.cat((phase_action, \
        torch.nn.functional.normalize(mod_action, dim=-1) * \
        (torch.abs(option) + 1e-10)), dim=-1)
    return prediction


if __name__ == '__main__':
    # 示例输入数据，假设N=3
    input_vector = torch.randn(5, 3)  # 6维输入向量，前3维是均值，后3维是标准差

    # 从多元正态分布中采样
    samples, log_probs = sample_from_multivariate_normal(input_vector)

    print("采样值:", samples)
    print("对数概率密度:", log_probs)