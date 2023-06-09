import torch
import model
from abc import abstractmethod
import time
import math


class MyMinPool1d(torch.nn.MaxPool1d):
    def forward(self, input_data):
        input_data = -1 * input_data
        return super().forward(input_data)


pooling_model_dict = {
    'max': torch.nn.MaxPool1d,
    'min': MyMinPool1d,
    'mean': torch.nn.AvgPool1d
}


class Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.wss = kwargs['wss']
        self.poolings = kwargs['poolings']
        self.filter_number = kwargs["filter_number"]
        self.word_vector_dim = kwargs['word_vector_dim']
        self.con_model_list, self.pooling_list = self.create_submodels()

    @abstractmethod
    def create_submodels(self):
        raise RuntimeError("Do implement this method!")


# def create_block_a(wss, poolings, number, word_vector_dim):
#     con_model_list = []
#     pooling_model_list = []
#     for i, ws in enumerate(wss):
#         con_model_list.append(torch.nn.Conv1d(in_channels=word_vector_dim, out_channels=number, kernel_size=ws))
#     for i, pooling in enumerate(poolings):
#         pooling_model_list.append(pooling(kernel_size=1000, stride=1))
#
#     return con_model_list,pooling_model_list


class BlockA(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_submodels(self):
        con_model_list = []
        pooling_model_list = []
        for i, ws in enumerate(self.wss):
            con_model_list.append(torch.nn.Conv1d(in_channels=self.word_vector_dim, out_channels=self.filter_number, kernel_size=ws))
        for i, pooling in enumerate(self.poolings):
            if pooling not in pooling_model_dict:
                raise ValueError
            pooling_model_list.append(pooling_model_dict[pooling](kernel_size=0, stride=1))
        con_model_list = torch.nn.ModuleList(con_model_list)
        pooling_model_list = torch.nn.ModuleList(pooling_model_list)
        return con_model_list, pooling_model_list

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


# class BlockB_old(Block):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #
    # def create_submodels(self):
    #     con_model_list = []
    #     pooling_model_list = []
    #     for i, ws in enumerate(self.wss):
    #         dim_model_list = []
    #         for j in range(self.word_vector_dim):
    #             temp =torch.nn.Conv1d(in_channels=1, out_channels=self.filter_number, kernel_size=ws)
    #
    #             temp.weight = torch.nn.Parameter(torch.ones_like(temp.weight)/2)
    #             temp.bias = torch.nn.Parameter(torch.ones_like(temp.bias)/2)
    #             dim_model_list.append(temp)
    #         dim_model_list = torch.nn.ModuleList(dim_model_list)
    #         con_model_list.append(dim_model_list)
    #     for i, pooling in enumerate(self.poolings):
    #         if pooling not in pooling_model_dict:
    #             raise ValueError
    #         pooling_model = pooling_model_dict[pooling]
    #         pooling_model_list.append(pooling_model(kernel_size=0, stride=1))
    #     con_model_list = torch.nn.ModuleList(con_model_list)
    #     pooling_model_list = torch.nn.ModuleList(pooling_model_list)
    #     return con_model_list, pooling_model_list
    #
    # def forward(self, input_data, gpu = False):
    #
    #     result = torch.Tensor(len(input_data), len(self.pooling_list), len(self.con_model_list),
    #                           len(self.con_model_list[0]), self.filter_number)
    #     if gpu:
    #         result = result.cuda()
    #     result = torch.autograd.Variable(result)
    #     for pool_index, pooling in enumerate(self.pooling_list):
    #         for con_index, con_model in enumerate(self.con_model_list):
    #             for dim_index, dim_model in enumerate(con_model):
    #                 index = torch.LongTensor([dim_index])
    #                 if gpu:
    #                     index = index.cuda()
    #                 dim_output = dim_model(input_data.index_select(1,index))
    #                 pooling.kernel_size = len(dim_output[0][0])
    #                 pooling_result = pooling(dim_output)
    #                 for i in range(len(input_data)):
    #                     result[i][pool_index][con_index][dim_index] = pooling_result[i].squeeze(1)
    #     return result

class BlockB(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_submodels(self):
        con_model_list = []
        pooling_model_list = []
        for i, ws in enumerate(self.wss):
            temp =torch.nn.Conv1d(in_channels=self.word_vector_dim, out_channels=self.word_vector_dim*self.filter_number, kernel_size=ws, groups=self.word_vector_dim)
            con_model_list.append(temp)
            # temp.weight = torch.nn.Parameter(torch.ones_like(temp.weight) / 2)
            # temp.bias = torch.nn.Parameter(torch.ones_like(temp.bias) / 2)
        con_model_list = torch.nn.ModuleList(con_model_list)
        for i, pooling in enumerate(self.poolings):
            if pooling not in pooling_model_dict:
                raise ValueError
            pooling_model = pooling_model_dict[pooling]
            pooling_model_list.append(pooling_model(kernel_size=0, stride=1))
        pooling_model_list = torch.nn.ModuleList(pooling_model_list)
        return con_model_list, pooling_model_list

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
        result = result.reshape(len(input_data), len(self.pooling_list), len(self.wss), self.word_vector_dim, self.filter_number)
        return result

block_type_dict = {
    'BlockA': {'Block': BlockA, 'kwargs': {'wss': model.wss_a, 'poolings': model.poolings_a, 'filter_number': model.filter_number_a, 'word_vector_dim': model.word_vector_dim}},
    'BlockB': {'Block': BlockB, 'kwargs': {'wss': model.wss_b, 'poolings': model.poolings_b, 'filter_number': model.filter_number_b, 'word_vector_dim': model.word_vector_dim}}
}


class SentenceModel(torch.nn.Module):
    def __init__(self):
        block_types = model.block_model_names
        super().__init__()

        self.block_dict = {}
        self.block_model_list = []
        for i, block_type in enumerate(block_types):
            if block_type not in block_type_dict:
                raise ValueError

            block = block_type_dict[block_type]['Block']
            kwargs = block_type_dict[block_type]['kwargs']
            block_model = block(**kwargs)
            self.block_model_list.append(block_model)
            self.block_dict[block_type] = i
        self.block_model_list = torch.nn.ModuleList(self.block_model_list)
        self.current_block = None

    def change_current_block(self, block_name):
        self.current_block = self.block_model_list[self.block_dict[block_name]]

    def forward(self, input_data):
        return self.current_block(input_data)

    # def cuda(self, *args, **kwargs):
    #     super().cuda(*args, **kwargs)
    #     for block in self.block_dict.values():
    #         block.cuda()
    #
    # def cpu(self):
    #     super().cpu()
    #     for block in self.block_dict.values():
    #         block.cpu()
def compara_tensor_data(data):
    n = 3
    temp = data * math.pow(10,n)
    temp = round(temp)
    return temp == 0

def test():
    # m = BlockB(**{'wss': model.wss_b, 'poolings': model.poolings_b, 'filter_number': model.filter_number_b, 'word_vector_dim': model.word_vector_dim})
    # ma = BlockA(**{'wss': model.wss_a, 'poolings': model.poolings_a, 'filter_number': model.filter_number_a, 'word_vector_dim': model.word_vector_dim})
    # m1 = BlockB1(**{'wss': model.wss_b, 'poolings': model.poolings_b, 'filter_number': model.filter_number_b,
    #               'word_vector_dim': model.word_vector_dim})
    # input = torch.randn(64, 300, 5)
    # star =time.time()
    # end = time.time()
    # output1 = m(input)
    # end = time.time()
    # print('m1:{}'.format(end - star))
    # star = time.time()
    # output2 = m1(input)
    # end = time.time()
    # print('m2:{}'.format(end-star))
    #
    # star = time.time()
    # output3 = ma(input)
    # end = time.time()
    # print('ma1:{}'.format(end - star))
    #
    # if not ((output2 - output1).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
    #     raise ValueError
    #
    #
    # m = BlockB(**{'wss': model.wss_b, 'poolings': model.poolings_b, 'filter_number': model.filter_number_b,
    #               'word_vector_dim': model.word_vector_dim}).cuda()
    # m1 = BlockB1(**{'wss': model.wss_b, 'poolings': model.poolings_b, 'filter_number': model.filter_number_b,
    #                 'word_vector_dim': model.word_vector_dim}).cuda()
    #
    # ma = ma.cuda()
    # input = input.cuda()
    # star = time.time()
    # end = time.time()
    # output1 = m(input,gpu= True)
    # end = time.time()
    # print('gpu_m1:{}'.format(end - star))
    # star = time.time()
    # output2 = m1(input)
    # end = time.time()
    # print('gpu_m2:{}'.format(end - star))
    #
    # star = time.time()
    # output3 = ma(input)
    # end = time.time()
    # print('gpu_ma1:{}'.format(end - star))
    # # print(output1, output2)
    # # print(output.size())
    # if not ((output2 - output1).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
    #     raise ValueError
    pass
if __name__ == '__main__':
    test()