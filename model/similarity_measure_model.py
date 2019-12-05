import torch
import model.sentence_model as sentence_model
from abc import abstractmethod

import math

def check_distance_input_data(x,y):
    if torch.isnan(x).sum() + torch.isnan(y).sum()> 0:
        print(torch.isnan(x).sum()) + torch.isnan(y).sum()
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
    def compare_algorithm(self, input_data1, input_data2):

        result_batch = []
        input_data1 = input_data1.data.permute(0,1,3,2)
        input_data2 = input_data2.data.permute(0,1,3,2)
        for s in range(len(input_data1)):
            result = []
            for pool_index in range(len(input_data1[s])):
                for num in range(len(input_data1[s][pool_index])):
                    # for compare_unit in self.compare_units:
                    #     temp = compare_unit(input_data1[s][pool_index][num], input_data2[s][pool_index][num])
                    #     requirs_grad = temp.requirs_grad
                    #     temp = temp.data
                    #     result.append(temp)
                    # if torch.isnan(temp).sum()>0:
                    #     print(torch.isnan(temp).sum())
                    #     raise ValueError
                    temp=calculate_compare_units(input_data1[s][pool_index][num], input_data2[s][pool_index][num], self.compare_units)
                    result.append(temp)
            result = torch.stack(result, dim=0)
            result_batch.append(result)
        result_batch = torch.stack(result_batch, dim=0)
        # result_batch = torch.autograd.Variable(result_batch, requirs_grad=requirs_grad)
        return result_batch


class VerticalComparisonModelForBlockA(ComparisonModel):
    def compare_algorithm(self, input_data1, input_data2):
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
         return  self.compare_model(input_data1,input_data2)