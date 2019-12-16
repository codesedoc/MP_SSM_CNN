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
import utils.visualization_tool as visualization_tool
import utils.log_tool as log_tool


class MSRPLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        batch_size = outputs.size()[0]
        correct_count = 0
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
                    l += max(yi-ground_score + 1, 0)
            if out.argmax() == labels[i]:
                correct_count +=1
            l_sum += l
        l_sum = l_sum / batch_size
        if not isinstance(l_sum, torch.Tensor):
            raise ValueError

        if l_sum.grad_fn is None:
            l_sum.requires_grad =True
        return l_sum, correct_count


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
    else:
        test_loader =None
    losser = MSRPLoss()
    losser.cuda()
    momentum = model_py.sgd_momentum
    log_tool.model_result_logger.info("momentum = {}".format(momentum))
    optimizer = torch.optim.SGD(params=learn_model.parameters(), lr=model_py.learn_rate, momentum=momentum)

    train_accuracy_list = []
    test_accuracy_list =[]
    for e in range(epoch):
        loss_sum = 0
        batch_number = len(train_loader)
        correct_count = 0
        print('training {}-th epoch'.format(e + 1))
        for index, training_example in enumerate(train_loader):
            input_sentence1s, input_sentence2s, labels = training_example
            optimizer.zero_grad()
            result = learn_model(input_sentence1s, input_sentence2s)
            loss, cor_count = losser(result, labels)
            correct_count += cor_count
            if model_py.show_run_time:
                start = time.time()
                loss.backward()
                end = time.time()
                print("back_ward_time:{}".format(end - start))
            else:
                loss.backward()

            optimizer.step()
            loss_sum += loss
            pb.update((index+1) * 100 / batch_number)
            # print('{}-th epoch, {}-th batch  loss:{}'.format(e+1, index+1, loss))
        print()
        print()
        train_accuracy = correct_count / train_manager.get_count_of_examples()
        if model_py.log_model_data_flag:
            log_tool.model_result_logger.info('epoch:{} train_accuracy:{}  arg_loss:{}'.format(e + 1, train_accuracy, loss_sum/batch_number))
        else:
            print('epoch:{} train_accuracy:{}  arg_loss:{}'.format(e + 1, train_accuracy, loss_sum/batch_number))
        train_accuracy_list.append(('epoch:{}'.format(e+1), train_accuracy))
        if test_manager is not None:
            test_accuracy = evaluation(test_loader, learn_model)
            if model_py.log_model_data_flag:
                log_tool.model_result_logger.info('test_accuracy:{}'.format(test_accuracy))
            else:
                print('test_accuracy:{}'.format(test_accuracy))
            test_accuracy_list.append(('epoch:{}'.format(e+1), test_accuracy))
        print()
    test_accuracy_list_filename = file_tool.PathManager.append_filename_to_dir_path(file_tool.PathManager.model_path, filename='test_accuracy', extent='pkl')
    train_accuracy_list_filename = file_tool.PathManager.append_filename_to_dir_path(file_tool.PathManager.model_path, filename='train_accuracy', extent='pkl')
    file_tool.save_data_pickle(test_accuracy_list, test_accuracy_list_filename)
    # test_accuracy_list_t = file_tool.load_data_pickle(test_accuracy_list_filename)

    file_tool.save_data_pickle(train_accuracy_list, train_accuracy_list_filename)
    # train_accuracy_list_t = file_tool.load_data_pickle(train_accuracy_list_filename)
    file_tool.save_data_pickle(learn_model, file_tool.PathManager.entire_model_file)


def get_learn_model(rebuild=False , use_gpu =False):
    if rebuild:
        learn_model = entire_model.EntireModel()
    else:
        learn_model = file_tool.load_data_pickle(file_tool.PathManager.entire_model_file)
    if use_gpu:
        learn_model.cuda()
    else:
        learn_model.cpu()
    return learn_model


def prepare_data(rebuild_data_manager=False, use_gpu = False):
    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=rebuild_data_manager)
    processor = pre_process.Preprocessor()
    train_manager = processor.pre_process(data_manager=train_manager, batch_size=model_py.batch_size, use_gpu=use_gpu, data_align=True,
                                          remove_error_word_vector=True, rebuild=rebuild_data_manager)
    test_manager = processor.pre_process(data_manager=test_manager, batch_size=model_py.batch_size, use_gpu=use_gpu, data_align=True,
                                         remove_error_word_vector=True, rebuild=rebuild_data_manager)
    return  train_manager,test_manager

def main(rebuild_model=False, rebuild_data_manager=False, use_gpu = False):
    begin_time = time.time()
    learn_model = get_learn_model(rebuild=rebuild_model, use_gpu=use_gpu)
    train_manager, test_manager = prepare_data(rebuild_data_manager=rebuild_data_manager, use_gpu=use_gpu)
    end_time = time.time()
    print('prepare_time：{}'.format(end_time-begin_time))

    # visualize_model(learn_model, train_manager)
    momentum = [0.2, 0.4, 0.6, 0.8]
    log_tool.model_result_logger.info("with test")
    for i, m in enumerate(momentum):
        model_py.sgd_momentum = m
        training(train_manager, 80, learn_model, test_manager)

    # model_py.sgd_momentum = 0.2
    # training(train_manager, , learn_model, test_manager)
    #
    # model_py.sgd_momentum = 0.4
    # training(train_manager, 2, learn_model, test_manager)
    #
    # model_py.sgd_momentum = 0.6
    # training(train_manager, 2, learn_model, test_manager)
    #
    # model_py.sgd_momentum = 0.8
    # training(train_manager, 2, learn_model, test_manager)

    momentum = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    log_tool.model_result_logger.info("without test")
    for i, m in enumerate(momentum):
        model_py.sgd_momentum = m
        training(train_manager, 20, learn_model)


def visualize_model(learn_model, data_manager):
    input_data = data_manager.get_an_input()
    filename = visualization_tool.create_log_filename()
    visualization_tool.log_graph(filename= filename, nn_model=learn_model, input_data=input_data, comment='entire_model_graph')
    # visualization_tool.run_tensorboard_command()


if __name__ == "__main__":
    main(rebuild_model=True, rebuild_data_manager=True, use_gpu=True)
