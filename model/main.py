import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.data_tool as data_tool
import utils.file_tool as file_tool
import model.entire_model as entire_model
import torch
import model as model_py
import utils.simple_progress_bar as simple_progress_bar
import model.pre_process as pre_process
import time
def null_grad_fn(x):
    print(x)
    pass


class MSRPLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        if outputs.is_cuda != labels.is_cuda:
            raise TypeError

        l_sum = torch.tensor(0, dtype=torch.float)
        if outputs.is_cuda:
            l_sum =l_sum.cuda()

        for i, out in enumerate(outputs):
            l = torch.tensor(0, dtype=torch.float)
            if outputs.is_cuda:
                l = l.cuda()
            ground_score = out[int(labels[i])]
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


def evaluation(test_loader, train_model):
    correct_count = 0
    count =0
    for input_sentence1s, input_sentence2s, labels in test_loader:
        result = train_model(input_sentence1s, input_sentence2s)
        for i, out in enumerate(result):
            l = result[i].argmax()
            target_index = labels[i]
            if l == target_index:
                correct_count +=1
            count +=1
    accuracy = correct_count/ count
    return accuracy


def training(train_manager, epoch, learn_model, test_manager=None, ):
    pb = simple_progress_bar.SimpleProgressBar()
    train_loader = train_manager.get_data_loader(drop_last=False)
    if test_manager is not None:
        test_loader = train_manager.get_data_loader(drop_last=False)

    losser = MSRPLoss()
    losser.cuda()
    optimizer = torch.optim.SGD(params=learn_model.parameters(), lr=0.01)
    # file_tool.save_data_pickle(model, file_tool.PathManager.entire_model_file)
    # model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    for e in range(epoch):
        loss_sum = 0
        batch_size = len(train_loader)
        for index, training_example in enumerate(train_loader):
            input_sentence1s, input_sentence2s, labels = training_example
            optimizer.zero_grad()
            result = learn_model(input_sentence1s, input_sentence2s)
            # loss = losser(result, labels.type(torch.float).unsqueeze(dim=1).expand_as(result))
            loss = losser(result, labels)
            star_time = time.time()
            loss.backward()
            end_time = time.time()
            print('back_ward_time:{}'.format(end_time-star_time))
            optimizer.step()
            loss_sum += loss
            # pb.update(index * 100 / batch_size)
            print('{}-th epoch, {}-th batch'.format(e+1, index+1))
        if test_manager is not None:
            evaluation(test_loader, learn_model)
        print('epoch:{}  arg_loss:{}'.format(e+1, loss_sum/batch_size))
    file_tool.save_data_pickle(learn_model, file_tool.PathManager.entire_model_file)


def get_learn_model(rebuild=False , use_gpu =False):
    if rebuild:
        learn_model = entire_model.EntireModel(number=model_py.num_filter_a, word_vector_dim=300, compare_units=model_py.compare_units, wss= model_py.wss)
    else:
        learn_model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    if use_gpu:
        learn_model.cuda()
    else:
        learn_model.cpu()
    return learn_model


def main(rebuild_model=False, rebuild_data_manager=False, use_gpu = False):
    begin_time = time.time()
    learn_model = get_learn_model(rebuild=rebuild_model , use_gpu=use_gpu)
    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
    processor = pre_process.Preprocessor()
    train_manager = processor.pre_process(data_manager=train_manager, batch_size= 4, use_gpu=use_gpu, data_align= True, remove_error_word_vector=True, rebuild=rebuild_data_manager)
    test_manager = processor.pre_process(data_manager=test_manager, batch_size= 4, use_gpu=use_gpu, data_align= True, remove_error_word_vector=True, rebuild=rebuild_data_manager)
    end_time = time.time()
    print('prepare_time：{}'.format(end_time-begin_time))
    training(train_manager, 500, learn_model, test_manager)


if __name__ == "__main__":
    main(rebuild_model=True, rebuild_data_manager=True, use_gpu=True)
