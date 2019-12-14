# import torch
# import time
#
#
# def calculate2( input_data1, input_data2):
#     input_data1 = input_data1.permute(0, 1, 3, 2)
#     input_data2 = input_data2.permute(0, 1, 3, 2)
#     shape1 = input_data1.size()
#     shape2 = input_data2.size()
#
#     shape = shape1
#     if shape1 != shape2:
#         raise ValueError
#
#     input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
#     input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
#     pairs_count = input_data1_temp.size()[1]
#
#     # result = torch.Tensor(shape[0], pairs_count, 3)
#     # if input_data1.is_cuda:
#     #     result = result.cuda()
#     result = []
#     for s in range(shape[0]):
#         result_temp1 = []
#         for i in range(pairs_count):
#             temp2 = torch.cdist(input_data1_temp[s][i].unsqueeze(dim=0), input_data2_temp[s][i].unsqueeze(dim=0),
#                                 2).squeeze()
#             temp3 = torch.cdist(input_data1_temp[s][i].unsqueeze(dim=0), input_data2_temp[s][i].unsqueeze(dim=0),
#                                 1).squeeze()
#             # temp1 = cos_distance(input_data1_temp[s][i], input_data2_temp[s][i]).squeeze()
#             temp1 = torch.nn.functional.normalize(input_data1_temp[s][i], dim=0).dot(
#                 torch.nn.functional.normalize(input_data2_temp[s][i], dim=0)).squeeze()
#             result_temp1.append(torch.stack([temp1, temp2, temp3], dim=0))
#
#         result.append(torch.stack(result_temp1, dim=0))
#     result = torch.stack(result, dim=0)
#     return result
#
#
# def calculate1(input_data1, input_data2):
#     input_data1 = input_data1.permute(0, 1, 3, 2)
#     input_data2 = input_data2.permute(0, 1, 3, 2)
#     shape1 = input_data1.size()
#     shape2 = input_data2.size()
#
#     shape = shape1
#     if shape1 != shape2:
#         raise ValueError
#
#     input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
#     input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
#     pairs_count = input_data1_temp.size()[1]
#     index_list = [i for i in range(pairs_count)]
#     result = []
#     for s in range(shape[0]):
#         temp1 = torch.nn.functional.normalize(input_data1_temp[s], dim=1).mm(
#             torch.nn.functional.normalize(input_data2_temp[s], dim=1).T).squeeze()
#         temp1 = temp1.diag(diagonal=0)
#         # temp1 = temp1[index_list, index_list]
#         temp2 = torch.cdist(input_data1_temp[s], input_data2_temp[s], 2).squeeze()
#         # temp2 = temp2.diag(diagonal=0)
#         temp2 = temp2[index_list, index_list]
#         temp3 = torch.cdist(input_data1_temp[s], input_data2_temp[s], 1).squeeze()
#         # temp3 = temp3.diag(diagonal=0)
#         temp3 = temp3[index_list, index_list]
#         result.append(torch.stack([temp1, temp2, temp3], dim=1))
#     result = torch.stack(result, dim=0)
#     return result
#
# def calculate1_1(input_data1, input_data2):
#     input_data1 = input_data1.permute(0, 1, 3, 2)
#     input_data2 = input_data2.permute(0, 1, 3, 2)
#     shape1 = input_data1.size()
#     shape2 = input_data2.size()
#
#     shape = shape1
#     if shape1 != shape2:
#         raise ValueError
#
#     input_data1_temp = input_data1.reshape(shape[0], -1, shape[-1])
#     input_data2_temp = input_data2.reshape(shape[0], -1, shape[-1])
#     pairs_count = input_data1_temp.size()[1]
#     index_list = [i for i in range(pairs_count)]
#     result = []
#     for s in range(shape[0]):
#         x = input_data1_temp[s]
#         y = input_data2_temp[s]
#         temp1 = torch.nn.functional.normalize(x, dim=1).mm(
#             torch.nn.functional.normalize(y, dim=1).T).squeeze()
#         temp1 = temp1.diag(diagonal=0)
#
#         temp2 = torch.sqrt(torch.pow(x - y, 2).mm(torch.ones_like((x - y).T)).diag(diagonal=0))
#
#         temp3 = torch.abs((x - y)). mm(torch.ones_like ((x - y).T)).diag(diagonal=0)
#
#         # temp3 = torch.cdist(input_data1_temp[s], input_data2_temp[s], 1).squeeze()
#         # temp3 = temp3.diag(diagonal=0)
#         # temp3 = temp3[index_list, index_list]
#         result.append(torch.stack([temp2,temp2,temp2], dim=1))
#     result = torch.stack(result, dim=0)
#     return result
#
# def calculate3(input_data1,input_data2):
#     input_data1 = input_data1.permute(0, 1, 3, 2)
#     input_data2 = input_data2.permute(0, 1, 3, 2)
#     shape1 = input_data1.size()
#     shape2 = input_data2.size()
#
#     shape = shape1
#     if shape1 != shape2:
#         raise ValueError
#
#     input_data1_temp = input_data1.reshape(-1, shape[-1])
#     input_data2_temp = input_data2.reshape(-1, shape[-1])
#
#     input_data1_temp_list = input_data1_temp.split(1, dim=0)
#     input_data2_temp_list = input_data2_temp.split(1, dim=0)
#     result = []
#     for input_data1_temp,input_data2_temp in zip(input_data1_temp_list, input_data2_temp_list):
#         temp1 = torch.nn.functional.normalize(input_data1_temp, dim=1).mm(torch.nn.functional.normalize(input_data2_temp, dim=1).T).squeeze()
#         temp1 = temp1.diag(diagonal=0)
#         temp2 = torch.cdist(input_data1_temp, input_data2_temp, 2).squeeze()
#         temp2 = temp2.diag(diagonal=0)
#         temp3 = torch.cdist(input_data1_temp, input_data2_temp, 1).squeeze()
#         temp3 = temp3.diag(diagonal=0)
#     # temp3 = temp3[index_list, index_list]
#         result.append(torch.stack([temp1, temp2,temp3],dim=0))
#     result = torch.stack(result, dim=0)
#     result = result.reshape(shape[0], -1, shape[-1])
#     return result
#
#
# t1 = torch.rand(64,3,3,525, requires_grad=True)
# t2 = torch.rand(64,3,3,525, requires_grad=True)
# start = time.time()
# result1 = calculate1(t1, t2)
# # result.backward(torch.ones(*result.size()))
# end = time.time()
# print("cpu:{}".format(end - start))
#
#
# t1 = t1.cuda()
# t2 = t2.cuda()
# t3 = torch.rand(64,3,3,525, requires_grad=True).cuda()
# t4 = torch.rand(64,3,3,525, requires_grad=True).cuda()
# start = time.time()
# # result2 = calculate1(t1, t2)
# result3 = calculate1_1(t1, t2)
# # result = torch.cat([result2,result3],dim=-1)
# result =result3
# result.backward(torch.ones(*result.size()).cuda())
# end = time.time()
# print("gpu:{}".format(end - start))
#
#
import torch
# x=torch.rand(255,255,8,8).cuda()
# up=torch.nn.Upsample(scale_factor=2)
# print(up(x).size())

# x = torch.arange(9).reshape(3,3)
# x=x.type(torch.float)
# x.requires_grad =True
#
# c = (x**3).sum()
#
# y=c
# # y.backward(gradient=torch.tensor(1,dtype=torch.float), keep_graph=True)
# print(y.grad)
# print(c.grad)
# print(x.grad)

x=torch.rand(255,255,16,16).cuda()
up=torch.nn.Upsample(scale_factor=2)
print(up(x).size())