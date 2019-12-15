import torch
import model


def calculate_scale():
    input_count = 0
    for select in model.sub_model_select_dict:
        measure_model = select['compare_model']
        input_count += model.output_size_of_compare_model_dict[measure_model]
    return (input_count,model.hidden_cell_number,2)

class FullConnectModel(torch.nn.Module):
    def __init__(self, scales=None):
        if scales is None:
            scales = calculate_scale()
        super().__init__()
        layer_list = []
        active_list = []
        for i in range(1,len(scales)):
            layer_list.append(torch.nn.Linear(scales[i-1], scales[i], bias=True))
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

