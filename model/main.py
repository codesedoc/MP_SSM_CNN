import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.data_tool as data_tool
import utils.file_tool as file_tool
import model.entire_model as entire_model
import torch
import model as model_py
import utils.simple_progress_bar as simple_progress_bar


def null_grad_fn(x):
    print(x)
    pass
class MSRPLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        l_sum = torch.tensor(0, dtype=torch.float)
        for i, out in enumerate(outputs):
            l = torch.tensor(0, dtype=torch.float)
            ground_score = out[labels[i]]
            for j,yi in enumerate(out):
                if j != labels[i]:
                    l += max(yi-ground_score +1, 0)
            l_sum += l
        l_sum / len(outputs)
        if not isinstance(l_sum, torch.Tensor):
            raise ValueError

        if l_sum.grad_fn is None:
            l_sum.requires_grad =True
        return l_sum


# def evolution(test_loader):
#

def training(train_manager, epoch, gpu_use):
    pb = simple_progress_bar.SimpleProgressBar()
    loader1, loader2 = train_manager.get_data_loader(batch_size=1, drop_last=True)
    model = entire_model.EntireModel(number=model_py.num_filter_a, word_vector_dim=300)
    losser = MSRPLoss()
    if gpu_use:
        model.cuda()
        losser.cuda()
    # file_tool.save_data_pickle(model, file_tool.PathManager.entire_model_file)
    # model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    batch1_list = []
    batch2_list = []
    for batch1, batch2 in zip(loader1, loader2):
        batch1[0] = batch1[0].permute(0, 2, 1)
        batch2[0] = batch2[0].permute(0, 2, 1)
        if gpu_use:
            batch1[0] = batch1[0].cuda()
            batch2[0] = batch2[0].cuda()
            batch1[1] = batch1[1].cuda()
        batch1_list.append(batch1)
        batch2_list.append(batch2)
    for e in range(epoch):
        i = 0
        loss_sum = 0
        for batch1, batch2 in zip(batch1_list, batch2_list):
            i+=1
            # print(sys.getsizeof(batch1))

            # print(round(batch1[0][0][0][0].tolist(),4))
            # if (round(batch1[0][0][0][0].tolist(),4) == 0.2833) and round(batch1[0][0][0][1].tolist(),4) == -0.6571 \
            #         or round(batch2[0][0][0][0].tolist(),4) == 0.2833 and round(batch2[0][0][0][1].tolist(),4) == -0.6571:


            optimizer.zero_grad()
            result = model(batch1[0], batch2[0])
            loss = losser(result, batch1[1])
            loss.backward()
            # try:
            #     loss.backward()
            # except RuntimeError as error:
            #     print(error)
            #     if loss.grad_fn is None:
            #         loss = loss
            #         pass
            #     continue
            optimizer.step()
            loss_sum += loss

            if i % 20 == 0:
                loss_sum = loss_sum/20
                print('epoch:{}  loss:{}'.format(e,loss_sum))
                loss_sum = 0
            pb.update((i % 20) * 100 / 19)
    file_tool.save_data_pickle(model, file_tool.PathManager.entire_model_file)


def main():
    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
    training(train_manager, 500, False)




if __name__ == "__main__":
    main()
