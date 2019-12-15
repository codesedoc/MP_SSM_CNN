import torch
import time
import math


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


def compare_v1(input_data1, input_data2):
    shape1 = input_data1.size()
    shape2 = input_data2.size()
    if shape1 != shape2:
        raise ValueError
    shape = shape1
    compare_units = [cos_distance, l2_distance, l1_distance]

    result = torch.Tensor(shape[0], shape[1], shape1[2]*shape2[2], len(compare_units))
    for s in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape1[2]):
                for k in range(shape2[2]):
                    temp = calculate_compare_units(input_data1[s][i][j], input_data2[s][i][k], compare_units)
                    result[s][i][j * shape1[2] + k] = temp
    result = result.reshape(shape[0],-1, len(compare_units))
    return result


def compare_v2(input_data1, input_data2):
    shape1 = input_data1.size()
    shape2 = input_data2.size()
    if shape1 != shape2:
        raise ValueError
    shape = shape1
    result = torch.Tensor(shape[0], shape[1], shape1[2]*shape2[2], 3)
    for s in range(shape[0]):
        for p in range(shape[1]):
            temp1 = torch.nn.functional.normalize(input_data1[s][p], dim=1).mm(
                torch.nn.functional.normalize(input_data2[s][p], dim=1).T).reshape(-1, 1)
            temp2 = torch.cdist(input_data1[s][p], input_data2[s][p]).reshape(-1,1)

            # temp3 = torch.abs(input_data1[s][p], dim=1)
            temp3 = torch.cdist(input_data1[s][p], input_data2[s][p], p =1).reshape(-1,1)
            temp =torch.cat([temp1,temp2,temp3],dim=1)
            result[s][p] = temp
    result = result.reshape(shape[0],-1, 3)
    return result

def compare_v3(input_data1, input_data2):
    shape1 = input_data1.size()
    shape2 = input_data2.size()
    if shape1 != shape2:
        raise ValueError
    shape = shape1
    # result = torch.Tensor(shape[0], shape[1], shape1[2]*shape2[2], 3)
    result = []
    for s in range(shape[0]):
        result_temp=[]
        for p in range(shape[1]):
            temp1 = torch.nn.functional.normalize(input_data1[s][p], dim=1).mm(
                torch.nn.functional.normalize(input_data2[s][p], dim=1).T).reshape(-1, 1)
            temp2 = torch.cdist(input_data1[s][p], input_data2[s][p]).reshape(-1,1)
            temp3 = torch.cdist(input_data1[s][p], input_data2[s][p], p =1).reshape(-1,1)
            temp = torch.cat([temp1,temp2,temp3],dim=1)
            result_temp.append(temp)
        result_temp = torch.stack(result_temp,dim=0)
        result.append(result_temp)
    result = torch.stack(result, dim=0)
    result = result.reshape(shape[0],-1, 3)
    return result


def VerticalB1(input_data1, input_data2):
    result_batch = []
    compare_units = [cos_distance, l2_distance, l1_distance]
    for s in range(len(input_data1)):
        result = []
        for pool_index in range(len(input_data1[s])):
            for ws in range(len(input_data1[s][pool_index])):
                for num in range(len(input_data1[s][pool_index][ws][0])):
                    # for compare_unit in self.compare_units:
                    #     temp = compare_unit(input_data1[s][pool_index][ws][num], input_data2[s][pool_index][ws][num])
                    #     requirs_grad = temp.requirs_grad
                    #     temp = temp.data
                    #     result.append(temp)
                    index = torch.LongTensor([num])
                    if input_data1.is_cuda:
                        index = index.cuda()
                    temp = calculate_compare_units(input_data1[s][pool_index][ws].index_select(dim=1,index=index).squeeze(dim=1), input_data2[s][pool_index][ws].index_select(dim=1,index=index).squeeze(dim=1),
                                                   compare_units)
                    result.append(temp)
        result = torch.stack(result, dim=0)
        result_batch.append(result)
    result_batch = torch.stack(result_batch, dim=0)
    # result_batch = torch.autograd.Variable(result_batch,requirs_grad=requirs_grad)
    return result_batch

def VerticalB2(input_data1, input_data2):
    result_batch = []
    compare_units = [cos_distance, l2_distance, l1_distance]
    for s in range(len(input_data1)):
        result = []
        for pool_index in range(len(input_data1[s])):
            for ws in range(len(input_data1[s][pool_index])):
                x = input_data1[s][pool_index][ws].T
                y = input_data2[s][pool_index][ws].T
                temp1 = torch.nn.functional.normalize(x, dim=1).mm(
                    torch.nn.functional.normalize(y, dim=1).T).diag(diagonal=0)
                temp2 = torch.sum(torch.pow(x - y, 2), dim=1)
                temp2 = revise_zero_data(temp2)
                temp2 = torch.sqrt(temp2)

                temp3 = torch.sum(torch.abs(x - y), dim=1)
                temp = torch.stack([temp1, temp2, temp3], dim=1)

                result.append(temp)
        result = torch.cat(result, dim=0)
        result_batch.append(result)
    result_batch = torch.stack(result_batch, dim=0)
    # result_batch = torch.autograd.Variable(result_batch,requirs_grad=requirs_grad)
    return result_batch

def VerticalB3(input_data1, input_data2):
    result_batch = []
    compare_units = [cos_distance, l2_distance, l1_distance]
    for s in range(len(input_data1)):
        for pool_index in range(len(input_data1[s])):
            for ws in range(len(input_data1[s][pool_index])):
                x = input_data1[s][pool_index][ws].T
                y = input_data2[s][pool_index][ws].T
                temp1 = torch.nn.functional.normalize(x, dim=1).mm(
                    torch.nn.functional.normalize(y, dim=1).T).diag(diagonal=0)
                temp2 = torch.sum(torch.pow(x - y, 2), dim=1)
                temp2 = revise_zero_data(temp2)
                temp2 = torch.sqrt(temp2)

                temp3 = torch.sum(torch.abs(x - y), dim=1)
                temp = torch.stack([temp1, temp2, temp3], dim=1)

                result_batch.append(temp)
    result_batch = torch.cat(result_batch, dim=0)
    result_batch = result_batch.reshape(input_data1.shape[0], -1 , len(compare_units))
    # result_batch = torch.autograd.Variable(result_batch,requirs_grad=requirs_grad)
    return result_batch

def revise_zero_data(tensor):
    boundary = 0.0001**2
    flag_tensor = (tensor < boundary).type(torch.float)
    tensor = tensor + flag_tensor*boundary
    return tensor


def compara_tensor_data(data):
    n = 3
    temp = data * math.pow(10,n)
    temp = round(temp)
    return temp == 0

# t1 = torch.rand(64,3,3,50, requires_grad=True)
# t2 = torch.rand(64,3,3,50, requires_grad=True)
# start = time.time()
# result1 = compare_v1(t1, t2)
# result2 = compare_v2(t1, t2)
# result2.backward(torch.ones(*result2.size()))
# end = time.time()
# print("cpu:{}".format(end - start))
#
# t1 = t1.cuda()
# t2 = t2.cuda()
#
# start = time.time()
# # result1 = compare_v1(t1, t2)
# result2 = compare_v2(t1, t2)
# result2.backward(torch.ones(*result2.size()))
# end = time.time()
# print("gpu:{}".format(end - start))
#
# if not ((result1-result2).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
#     raise ValueError


t1 = torch.rand(1,2,3,300,25, requires_grad=True)
t2 = torch.rand(1,2,3,300,25, requires_grad=True)

start = time.time()
result1 = VerticalB1(t1,t2)
result2 = VerticalB3(t1, t2)
end = time.time()
result2.backward(torch.ones(*result2.size()))
if not ((result1-result2).cpu().detach().apply_(compara_tensor_data)).type(torch.bool).all():
    raise ValueError


print("cpu:{}".format(end - start))
#
t1 = t1.cuda()
t2 = t2.cuda()
#
start = time.time()
# result1 = VerticalB1(t1, t2)
result2 = VerticalB3(t1, t2)
result2.backward(torch.ones(*result2.size()).cuda())
end = time.time()
print("gpu:{}".format(end - start))



