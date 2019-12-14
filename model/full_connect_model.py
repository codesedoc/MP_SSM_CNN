import torch


class FullConnectModel(torch.nn.Module):
    def __init__(self, scales):
        super().__init__()
        layer_list = []
        active_list = []
        for i in range(1,len(scales)):
            layer_list.append(torch.nn.Linear(scales[i-1],scales[i], bias=True))
            active_list.append(torch.nn.Tanh())
        active_list[-1] = torch.nn.LogSoftmax(dim=-1)
        self.layer_list =  torch.nn.ModuleList(layer_list)
        self.active_list = torch.nn.ModuleList(active_list)
        
    def forward(self, input_data):
        # input_data = input_data.permute(1, 0, 2)
        result = input_data
        if torch.isnan(result).sum()>0:
            print(torch.isnan(result))
            raise ValueError
        for model, active in zip(self.layer_list, self.active_list):
            result = model(result)
            result = active(result)
        return result

