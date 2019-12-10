import torch
import model.sentence_model as sentence_model
from abc import abstractmethod

import math
import time
def check_distance_input_data(x,y):
    if torch.isnan(x).sum() + torch.isnan(y).sum()> 0:
        print(torch.isnan(x).sum()), torch.isnan(y).sum()
        raise ValueError

    if (not isinstance(x[0], torch.autograd.Variable)) or (not isinstance(y[0], torch.autograd.Variable)):
        raise ValueError
    if (len(x.size()) != 1) or (len(y.size()) != 1):
        raise ValueError
    if len(x) != len(y):
        raise ValueError
def cos_distance(x,y):
    check_distance_input_data(x,y)
    # sum_inner = 0
    # sum_x = 0
    # sum_y = 0
    # for i in range(len(x)):
    #     sum_x += x[i]**2
    #     sum_y += y[i]**2
    #     sum_inner += x[i]*y[i]
    # return sum_inner / math.sqrt(sum_x) * math.sqrt(sum_y)
    molecule = torch.mul(x,y).sum()
    denominator = torch.sqrt(torch.pow(x,2).sum()) * torch.sqrt(torch.pow(y,2).sum())
    result =  torch.div(molecule, denominator)
    if torch.isnan(result).sum() > 0:
        print(torch.isnan(result).sum())
        raise ValueError
    return result

def l2_distance(x,y):
    check_distance_input_data(x, y)
    # sum = 0
    # for i in range(len(x)):
    #     sum += (x[i]+y[i])**2
    return torch.sqrt(torch.pow(x-y, 2).sum())


def l1_distance(x,y):
    check_distance_input_data(x, y)
    # sum = 0
    # for i in range(len(x)):
    #     sum += torch.sqrt((x[i] + y[i]) ** 2)

    return torch.sqrt(torch.pow(x-y, 2)).sum()


def calculate_compare_units(x,y,compare_units):
    result = []
    for compare_unit in compare_units:
        temp = compare_unit(x, y)
        if torch.isnan(temp).sum() > 0:
            print(torch.isnan(temp).sum())
            # raise ValueError
        result.append(temp)
    result = torch.stack(result, dim=0)
    return result


compare_unit_itme_dic = {
    'cos':cos_distance,
    'l2':l2_distance,
    'l1':l1_distance
}

def create_compare_units(compare_units):
    result = []
    for com in compare_units:
        result.append(compare_unit_itme_dic[com])
    return result


class ComparisonModel(torch.nn.Module):
    def __init__(self, compare_units):
        super().__init__()
        self.compare_units = compare_units

    def forward(self, input_data1, input_data2):
        result = self.compare_algorithm(input_data1, input_data2)
        return result

    @abstractmethod
    def compare_algorithm(self):
        pass


class HorizontalComparisonModel(ComparisonModel):
    def compare_algorithm_method1(self, input_data1, input_data2):

        result_batch = []
        input_data1 = input_data1.permute(0,1,3,2)
        input_data2 = input_data2.permute(0,1,3,2)
        for s in range(len(input_data1)):
            result = []
            for pool_index in range(len(input_data1[s])):
                for num in range(len(input_data1[s][pool_index])):

                    temp=calculate_compare_units(input_data1[s][pool_index][num], input_data2[s][pool_index][num], self.compare_units)
                    result.append(temp)
            result = torch.stack(result, dim=0)
            result_batch.append(result)
        result_batch = torch.stack(result_batch, dim=0)
        # result_batch = torch.autograd.Variable(result_batch, requirs_grad=requirs_grad)
        return result_batch

    def compare_algorithm_method2(self, input_data1, input_data2):
        input_data1 = input_data1.permute(0, 1, 3, 2)
        input_data2 = input_data2.permute(0, 1, 3, 2)
        shape1 = input_data1.size()
        shape2 = input_data2.size()

        shape = shape1
        if shape1 != shape2:
            raise ValueError

        input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
        input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
        pairs_count = input_data1_temp.size()[1]
        result = torch.Tensor(shape[0], pairs_count, len(self.compare_units))
        if input_data1.is_cuda:
            result =result.cuda()
        for s in range(shape[0]):
            for i in range(pairs_count):
                temp = calculate_compare_units(input_data1_temp[s][i], input_data2_temp[s][i],
                                               self.compare_units)
                result[s][i] = temp
        return result

    def compare_algorithm_method3(self, input_data1, input_data2):
        input_data1 = input_data1.permute(0, 1, 3, 2)
        input_data2 = input_data2.permute(0, 1, 3, 2)
        shape1 = input_data1.size()
        shape2 = input_data2.size()

        shape = shape1
        if shape1 != shape2:
            raise ValueError

        input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
        input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
        pairs_count = input_data1_temp.size()[1]

        result = torch.Tensor(shape[0], pairs_count, 3)
        if input_data1.is_cuda:
            result =result.cuda()
        for s in range(shape[0]):
            for i in range(pairs_count):
                temp2 = torch.cdist(input_data1_temp[s][i].unsqueeze(dim=0), input_data2_temp[s][i].unsqueeze(dim=0), 2).squeeze()
                temp3 = torch.cdist(input_data1_temp[s][i].unsqueeze(dim=0), input_data2_temp[s][i].unsqueeze(dim=0), 1).squeeze()
                # temp1 = cos_distance(input_data1_temp[s][i], input_data2_temp[s][i]).squeeze()
                temp1 = torch.nn.functional.normalize(input_data1_temp[s][i],dim=0).dot(torch.nn.functional.normalize(input_data2_temp[s][i],dim=0)).squeeze()
                result[s][i] = torch.stack([temp1, temp2,temp3],dim=0)
        return result


    def compare_algorithm_method4(self, input_data1, input_data2):
        input_data1 = input_data1.permute(0, 1, 3, 2)
        input_data2 = input_data2.permute(0, 1, 3, 2)
        shape1 = input_data1.size()
        shape2 = input_data2.size()

        shape = shape1
        if shape1 != shape2:
            raise ValueError
        input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
        input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
        result = []
        for s in range(shape[0]):
            # temp1 = torch.nn.functional.normalize(input_data1_temp[s], dim=1).mm(torch.nn.functional.normalize(input_data2_temp[s], dim=1).T).squeeze()
            # temp1 = temp1.diag(diagonal=0)
            # temp2 = torch.cdist(input_data1_temp[s], input_data2_temp[s], 2).squeeze()
            # temp2 = temp2.diag(diagonal=0)
            # temp3 = torch.cdist(input_data1_temp[s], input_data2_temp[s], 1).squeeze()
            # temp3 = temp3.diag(diagonal=0)

            x = input_data1_temp[s]
            y = input_data2_temp[s]
            temp1 = torch.nn.functional.normalize(x, dim=1).mm(
                torch.nn.functional.normalize(y, dim=1).T).squeeze()
            temp1 = temp1.diag(diagonal=0)

            temp2 = torch.sqrt(torch.abs(torch.pow(x - y, 2).mm(torch.ones_like((x - y).T)).diag(diagonal=0)))
            temp21 = torch.sum(torch.pow(x - y, 2), dim=1)
            temp3 = torch.abs((x - y)).mm(torch.ones_like((x - y).T)).diag(diagonal=0)
            temp31 = torch.sum(torch.abs(x - y), dim=1)
            if (temp21<0).any():
                raise ValueError
            result.append(torch.stack([temp31, temp21,temp31],dim=1))
        result = torch.stack(result, dim=0)
        return result

    def compare_algorithm(self, input_data1, input_data2):
        # start = time.time()
        # result1 = self.compare_algorithm_method1(input_data1, input_data2)
        # end = time.time()
        # print("method1:{}".format(end - start))
        # # start = time.time()
        # # result2 =self.compare_algorithm_method2(input_data1, input_data2)
        # # end = time.time()
        # # print("method2:{}".format(end-start))
        #
        # start = time.time()
        # result3 = self.compare_algorithm_method3(input_data1, input_data2)
        # end = time.time()
        # print("method3:{}".format(end - start))
        # # print(result3)
        # start = time.time()
        result4 = self.compare_algorithm_method4(input_data1, input_data2)
        # end = time.time()
        # print("method4:{}".format(end - start))
        # print(reslut1)
        # print(result2)

        # print(result4)
        # for i, f in enumerate((result1-result3).cpu().detach().apply_(compara_tensor_data).tolist()):
        #     for j, f1 in enumerate(f):
        #         for k, f2 in enumerate(f1):
        #             if not f2:
        #                 print(i,j,k)
        # if not ((result1-result3).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
        #     raise ValueError
        # if not ((result4-result3).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
        #     raise ValueError
        return result4
def compara_tensor_data(data):
    n = 4
    temp = data * math.pow(10,n)
    temp = round(temp)
    return temp == 0
class VerticalComparisonModelForBlockA(ComparisonModel):
    def compare_algorithm_old(self, input_data1, input_data2):
        result_batch = []
        for s in range(len(input_data1)):
            result = []
            for i in range(len(input_data1[s])):
                for j in range(len(input_data1[s][i])):
                    for k in range(len(input_data2[s][i])):
                        # for compare_unit in self.compare_units:
                        #     temp = compare_unit(input_data1[s][i][j], input_data2[s][i][k])
                        #     requirs_grad = temp.requirs_grad
                        #     temp = temp.data
                        #     result.append(temp)
                        temp = calculate_compare_units(input_data1[s][i][j], input_data2[s][i][k],
                                                       self.compare_units)
                        result.append(temp)
            result = torch.stack(result, dim=0)
            result_batch.append(result)
        result_batch = torch.stack(result_batch, dim=0)
        # result_batch = torch.autograd.Variable(result_batch,requirs_grad=requirs_grad)
        return result_batch
    def compare_algorithm(self, input_data1, input_data2):
        shape1 = input_data1.size()
        shape2 = input_data2.size()
        if shape1 != shape2:
            raise ValueError
        shape = shape1

        input_data1 = input_data1.reshape(shape[0], -1, shape[-1])
        input_data2 = input_data2.reshape(shape[0], -1, shape[-1])
        pairs_count = input_data1.size()[1]
        result = torch.Tensor(shape[0], pairs_count*pairs_count, shape[-1])
        for s in range(shape[0]):
            for i in range(pairs_count):
                for j in range(pairs_count):
                    temp = calculate_compare_units(input_data1[s][i], input_data2[s][j], self.compare_units)
                    result[s][i*pairs_count+j] = temp
        result_old = self.compare_algorithm_old(input_data1, input_data2)
        return result

class VerticalComparisonModelForBlockB(ComparisonModel):
    def compare_algorithm(self, input_data1, input_data2):
        result_batch = []
        for s in range(len(input_data1)):
            result = []
            for pool_index in range(len(input_data1[s])):
                for ws in range(len(input_data1[s][pool_index])):
                    for num in range(len(input_data1[s][pool_index][ws])):
                        # for compare_unit in self.compare_units:
                        #     temp = compare_unit(input_data1[s][pool_index][ws][num], input_data2[s][pool_index][ws][num])
                        #     requirs_grad = temp.requirs_grad
                        #     temp = temp.data
                        #     result.append(temp)
                        temp = calculate_compare_units(input_data1[s][pool_index][ws][num], input_data2[s][pool_index][ws][num],
                                                       self.compare_units)
                        result.append(temp)
            result = torch.stack(result, dim=0)
            result_batch.append(result)
        result_batch = torch.stack(result_batch, dim=0)
        # result_batch = torch.autograd.Variable(result_batch,requirs_grad=requirs_grad)
        return result_batch


class SimilarityMeasureModel(torch.nn.Module):
    def __init__(self, block_type, compare_units, horizontal_compare):
        super().__init__()
        compare_units=create_compare_units(compare_units)
        if horizontal_compare:
            self.compare_model = HorizontalComparisonModel(compare_units)
        else:
            if block_type == 'BloackA':
                self.compare_model = VerticalComparisonModelForBlockA(compare_units)
            else:
                self.compare_model = VerticalComparisonModelForBlockB(compare_units)

    def forward(self, input_data1, input_data2):
        result = self.compare_model(input_data1,input_data2)
        result = result.reshape(result.size()[0], -1)
        return result