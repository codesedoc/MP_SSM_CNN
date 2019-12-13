import torch
import model
from abc import abstractmethod

class MyMinPool1d(torch.nn.MaxPool1d):


    def forward(self, input):
        input = -1* input
        return super().forward(input)


pooling_model_dict = {
    'max': torch.nn.MaxPool1d,
    'min': MyMinPool1d,
    'mean': torch.nn.AvgPool1d
}


class Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.wws = kwargs['wss']
        self.poolings = kwargs['poolings']
        self.filter_number = kwargs["filter_number"]
        self.word_vector_dim = kwargs['word_vector_dim']
        self.con_model_list, self.pooling_list = self.create_submodels()

    @abstractmethod
    def create_submodels(self):
        raise RuntimeError("Do implement this method!")

# blook_type = ['BlockA', 'BlockB']

def create_block_a(wss, poolings, number, word_vector_dim):
    con_model_list = []
    pooling_model_list = []
    for i, ws in enumerate(wss):
        con_model_list.append(torch.nn.Conv1d(in_channels=word_vector_dim, out_channels=number, kernel_size=ws))
    for i, pooling in enumerate(poolings):
        pooling_model_list.append(pooling(kernel_size=1000, stride=1))

    return con_model_list,pooling_model_list

class BlockA(torch.nn.Module):
    def __init__(self, con_model_list, pooling_list):
        super().__init__()
        self.con_model_list = torch.nn.ModuleList(con_model_list)
        self.pooling_list = pooling_list

    def forward(self, input_data):
        result = []

        for pooling in self.pooling_list:
            temp1 = []
            for con_model in self.con_model_list:
                try:
                    con_output = con_model(input_data)
                except RuntimeError as e:
                    print(e)
                    raise
                pooling.kernel_size = int(con_output.size()[-1])
                temp1.append(pooling(con_output).squeeze(-1))
            temp2 = torch.stack(temp1, dim=1)
            result.append(temp2)

        result = torch.stack(result, dim=1)
        return result


def create_block_b(wss, poolings, number, word_vector_dim):
    con_model_list = []
    pooling_model_list = []
    for i, ws in enumerate(wss):
        dim_model_list = []
        for j in range(word_vector_dim):
            dim_model_list.append(torch.nn.Conv1d(in_channels=1, out_channels=number, kernel_size=ws))
        dim_model_list = torch.nn.ModuleList(dim_model_list)
        con_model_list.append(dim_model_list)
    for i, pooling in enumerate(poolings):
        pooling_model_list.append(pooling(kernel_size=0, stride=1))
    return con_model_list, pooling_model_list


class BlockB(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_submodels(self):
        con_model_list = []
        pooling_model_list = []
        for i, ws in enumerate(self.wss):
            dim_model_list = []
            for j in range(self.word_vector_dim):
                dim_model_list.append(torch.nn.Conv1d(in_channels=1, out_channels=self.filter_number, kernel_size=ws))
            dim_model_list = torch.nn.ModuleList(dim_model_list)
            con_model_list.append(dim_model_list)
        for i, pooling in enumerate(self.poolings):
            pooling_model_list.append(pooling(kernel_size=0, stride=1))
        return con_model_list, pooling_model_list

    def forward(self, input_data):
        result = torch.Tensor(len(input_data), len(self.pooling_list), len(self.con_model_list),
                              len(self.con_model_list[0]))
        result = torch.autograd.Variable(result)
        for pool_index, pooling in enumerate(self.pooling_list):
            for con_index, con_model in enumerate(self.con_model_list):
                for dim_index, dim_model in enumerate(con_model):
                    dim_output = dim_model(input_data.index_select(1,torch.LongTensor([dim_index])).squeeze(1))
                    pooling.kernel_size = len(dim_output[0][0])
                    pooling_result = pooling(dim_output)
                    for i in range(len(input_data)):
                        result[i][pool_index][con_index][dim_index] = pooling_result[i].squeeze[1]
        return result





class SentenceModel(torch.nn.Module):
    def __init__(self, block_type, wss, poolings, filter_number, word_vector_dim):
        super().__init__()
        if block_type == "BlockA":
            con_model_list,pooling_model_list = create_block_a(wss=wss, poolings=poolings,number=number,word_vector_dim=word_vector_dim)
            self.Block = BlockA(con_model_list,pooling_model_list)
        else:
            con_model_list,pooling_model_list =create_block_b()
            self.Block = BlockB(wss=wss, poolings=poolings, filter_number=filter_number,word_vector_dim=word_vector_dim)

    def forward(self, input_data):
        return self.Block(input_data)


def test():
    m = torch.nn.MaxPool1d(3, stride=2, ceil_mode=False)
    input = torch.randn(20, 16, 2)
    output = m(input)
    print(output.size())
if __name__ == '__main__':
    test()