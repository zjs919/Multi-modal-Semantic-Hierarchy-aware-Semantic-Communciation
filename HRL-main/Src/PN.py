import torch
import torch.nn as nn


class HighPolicyNetwork(nn.Module):
    def __init__(self, input_size = 500, hidden_size = 2000, output_size = 1):
        super(HighPolicyNetwork, self).__init__()
        
        # LSTM层，接收e_now和r_now作为输入
        self.lstm = nn.LSTM(input_size*5, hidden_size, batch_first=False, num_layers=1)
        
        # MLP层，接收LSTM的输出和r_q作为输入
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + input_size*3, 2*output_size),
            nn.ReLU(),
            nn.Linear(2*output_size, 2*output_size)
        )
        
    def forward(self, e_trace, r_trace, r_q):
        torch.cuda.empty_cache()
        # LSTM层的前向传播
        lstm_out, _ = self.lstm(torch.cat((e_trace, r_trace), dim=-1))
        
        # MLP层的前向传播，接收LSTM的输出和r_q
        mlp_input = torch.cat((lstm_out[-1], r_q), dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        return mlp_output



class LowPolicyNetwork(nn.Module):
    def __init__(self, input_size = 500, hidden_size = 2000, output_size = 1000):
        super(LowPolicyNetwork, self).__init__()
        
        # LSTM层，接收e_now和r_now作为输入
        self.lstm = nn.LSTM(input_size*5, hidden_size, batch_first=False, num_layers=1)
        
        # MLP层，接收LSTM的输出和r_q作为输入
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + input_size*3, 2*output_size),
            nn.ReLU(),
            nn.Linear(2*output_size, 2*output_size)
        )
        
    def forward(self, e_trace, r_trace, r_q):
        # LSTM层的前向传播
        lstm_out, _ = self.lstm(torch.cat((e_trace, r_trace), dim=-1))
        
        # MLP层的前向传播，接收LSTM的输出和r_q
        mlp_input = torch.cat((lstm_out[-1], r_q), dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        return mlp_output





if __name__ == "__main__":
    # 定义输入维度、隐藏层维度和输出维度
    N = 500
    input_size = N  # N维
    hidden_size = N  # 可根据需要调整
    output_size = N  # N维

    # 创建策略网络
    policy_net = LowPolicyNetwork(input_size, hidden_size, output_size)

    # 示例输入数据
    e_now = torch.randn(1, N)
    r_now = torch.randn(1, N)
    r_q = torch.randn(1, N)

    # 前向传播
    output = policy_net(e_now, r_now, r_q)

    print("网络输出维度:", output.shape)  # 应该是[1, N, 2*N]


