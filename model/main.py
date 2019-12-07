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


def training(train_manager, epoch, test_manager=None, rebuild_model = False):
    pb = simple_progress_bar.SimpleProgressBar()
    train_loader = train_manager.get_data_loader(drop_last=False)
    if test_manager is not None:
        test_loader = train_manager.get_data_loader(drop_last=False)
    if rebuild_model:
        model = entire_model.EntireModel(number=model_py.num_filter_a, word_vector_dim=300, compare_units=model_py.compare_units, wss= model_py.wss)
    else:
        model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    losser = MSRPLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # file_tool.save_data_pickle(model, file_tool.PathManager.entire_model_file)
    # model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    for e in range(epoch):
        loss_sum = 0
        batch_size = len(train_loader)
        for index, training_example in enumerate(train_loader):
            input_sentence1s, input_sentence2s, labels = training_example
            optimizer.zero_grad()
            result = model(input_sentence1s, input_sentence2s)
            loss = losser(result, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            pb.update(index * 100 / batch_size)

        print('epoch:{}  loss:{}'.format(e,loss_sum))
        if test_manager is not None:
            evaluation(test_loader, losser)
    file_tool.save_data_pickle(model, file_tool.PathManager.entire_model_file)


def main():
    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
    processor = pre_process.Preprocessor()
    train_manager = processor.pre_process(data_manager=train_manager, batch_size= 32, use_gpu=False, data_align= True, remove_error_word_vector=True)
    test_manager = processor.pre_process(data_manager=test_manager, batch_size= 32, use_gpu=False, data_align= True, remove_error_word_vector=True)
    training(train_manager, 500, test_manager, rebuild_model = True)




if __name__ == "__main__":
    main()
