from abc import abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import file_tool
import torch.utils.data as torch_data
import numpy as np
import re
import torch
import os
from utils import word_embedding

class DataItem:
    def __init__(self, original_data):
        self.original_data = original_data
        self.convert_data_form()

    @abstractmethod
    def convert_data_form(self):
        pass

    @abstractmethod
    def get_tensor(self):
        pass

class MSRPInput(DataItem):
    def convert_data_form(self):
        word_dictionary = word_embedding.get_dictionary_instance()
        self.word_count = 0
        self.no_find_word = {}
        if type(self.original_data) != str:
            raise ValueError
        new_sentence = []
        for i,word in enumerate(self.original_data.split()):
            r=re.finditer(r'[^a-zA-Z]|[a-zA-Z]+',word)
            for part in r:
                substr = word[part.span()[0] : part.span()[1]]
                # if len(re.findall(r'[a-zA-Z]',substr))>1 and substr
                #     raise ValueError
                vector, find_flag = word_dictionary.word2vector(substr)
                self.word_count += 1
                new_sentence.append(vector)
                if not find_flag:
                    self.no_find_word[word] = vector
        self.sentence = np.array(new_sentence, dtype=np.float)

    def get_tensor(self):
        result = torch.from_numpy(self.sentence).type(torch.float32)
        return result

class MSRPLabel(DataItem):
    def convert_data_form(self):
        self.label = int(self.original_data)
        if self.label != 0 and self.label != 1:
            raise ValueError

    def get_tensor(self):
        result =  torch.tensor(self.label)
        return result

class DataElement:
    def __init__(self, **items):
        self.members = items

    def __getitem__(self, item):
        return self.members[item]

    def __setitem__(self, key, value):
        self.members[key] = value

    def __iter__(self):
        return self


class DataExample(DataElement):
    @abstractmethod
    def get_example_pair(self):
        pass


class MSRPDataExample(DataExample):
    def __init__(self, input_sentence:MSRPInput, label:MSRPLabel):
        super().__init__(input_sentence=input_sentence, label=label)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result is None:
            raise ValueError
        return result

    def get_example_pair(self):
        input_data, label_data= self.members['input_sentence'], self.members['label']
        if (type(input_data) is not MSRPInput) or (type(label_data) is not MSRPLabel):
            raise TypeError
        return input_data.get_tensor(), label_data.get_tensor()


# class DataArray:
#     def __init__(self, data_element_list):
#         self.array = np.array(data_element_list, dtype=np.object)
#         #self.array = torch.from_numpy(self.array)


class MyDataSet(torch_data.dataset.Dataset):
    def __init__(self, data_array):
        if not isinstance(data_array[0], DataExample):
            raise TypeError

        super().__init__()
        self.data_array = data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, item):
        result = self.data_array[item].get_example_pair()
        if type(result[0]) != torch.Tensor or type(result[1])!=torch.Tensor:
            raise TypeError
        return self.data_array[item].get_example_pair()


class DataManager:
    def __init__(self, original_file):
        self.file_operator = file_tool.FileOperator(original_file)
        self.original_data = self.file_operator.read(operation_func=lambda rows: rows, model = 'r')
        self.format_original_data()
        pass

    @abstractmethod
    def format_original_data(self):
        pass

    @abstractmethod
    def get_data_loader(self, batch_size):
        pass
        # return torch_data.dataloader.DataLoader(self.data_set, batch_size=batch_size)


class MSRPDataManager(DataManager):
    def __init__(self, original_file):
        self.data_set1 = None
        self.data_set2 = None
        DataManager.__init__(self, original_file)

    def format_original_data(self):
        example1_list = []
        example2_list = []
        self.original_data = self.original_data[1:]
        word_count = 0
        no_find_word = {}
        for data in self.original_data:
            re_pattern = r'.+?[\t\n]'
            result=re.findall(re_pattern, data)
            label = MSRPLabel(result[0])
            input_sentence1 = MSRPInput(result[3])
            input_sentence2 = MSRPInput(result[4])
            example1 = MSRPDataExample(input_sentence=input_sentence1, label=label)
            example2 = MSRPDataExample(input_sentence=input_sentence2,  label=label)

            example1_list.append(example1)
            example2_list.append(example2)
            word_count += input_sentence1.word_count + input_sentence2.word_count
            no_find_word.update(input_sentence1.no_find_word)
            no_find_word.update(input_sentence2.no_find_word)

        log_str = "total word:{}, no_in_dictionary_word:{}\n".format(word_count, len(no_find_word))
        log_str += str(no_find_word)
        file_tool.save_data(log_str, file_tool.PathManager.word_no_in_dictionary_file, 'w')
        example1_array = np.array(example1_list, dtype=np.object)
        example2_array = np.array(example2_list, dtype=np.object)
        self.data_set1 = MyDataSet(example1_array)
        self.data_set2 = MyDataSet(example2_array)

    def get_data_loader(self, batch_size, drop_last):
        return torch_data.dataloader.DataLoader(self.data_set1, batch_size=batch_size, drop_last=drop_last),\
               torch_data.dataloader.DataLoader(self.data_set2, batch_size=batch_size, drop_last=drop_last)


    def get_max_length_of_sentence(self):
        max_length = 0
        for data in self.data_set1:
            input_data = data[0]
            if len(input_data) > max_length:
                max_length = len(input_data)
        for data in self.data_set1:
            input_data = data[0]
            if len(input_data) > max_length:
                max_length = len(input_data)
        return max_length

MSRPC_Train_Manager_Single_Instance = None
MSRPC_Test_Manager_Single_Instance = None


def get_msrpc_manager(re_build = False):
    global MSRPC_Train_Manager_Single_Instance, MSRPC_Test_Manager_Single_Instance
    if MSRPC_Train_Manager_Single_Instance is None:
        if (not re_build) and os.path.isfile(file_tool.PathManager.msrpc_train_data_manager_path):
            train_manager = file_tool.load_data_pickle(file_tool.PathManager.msrpc_train_data_manager_path)
        else:
            train_manager = MSRPDataManager(file_tool.PathManager.msrpc_train_data_set_path)
            file_tool.save_data_pickle(train_manager, file_tool.PathManager.msrpc_train_data_manager_path)
        MSRPC_Train_Manager_Single_Instance = train_manager

    if MSRPC_Test_Manager_Single_Instance is None:

        if (not re_build) and os.path.isfile(file_tool.PathManager.msrpc_test_data_manager_path):
            test_manager = file_tool.load_data_pickle(file_tool.PathManager.msrpc_test_data_manager_path)
        else:
            test_manager = MSRPDataManager(file_tool.PathManager.msrpc_test_data_set_path)
            file_tool.save_data_pickle(test_manager, file_tool.PathManager.msrpc_test_data_manager_path)

        MSRPC_Test_Manager_Single_Instance = test_manager
    return MSRPC_Train_Manager_Single_Instance, MSRPC_Test_Manager_Single_Instance


def test():
    # train_manager = MSRPDataManager(file_tool.PathManager.msrpc_train_data_set_path)
    # loader1, loader2 = train_manager.get_data_loader(batch_size=1, drop_last=True)
    # train_data1 =[]
    # train_data2 = []
    # for batch_datas, batch_labels in loader1:
    #     train_data1.append((batch_datas,batch_labels))
    #
    # for batch_datas, batch_labels in loader2:
    #     train_data2.append((batch_datas,batch_labels))
    #
    # if len(train_data1) != len(train_data2):
    #     raise ValueError
    #
    #
    # count = 10
    # repeat = 0
    # for i in range(count):
    #     index = [random.randint(0, 299) for i in range(count)]
    #     batch1 = train_data1[i][0][0][[0],index]
    #     batch2 = train_data2[i][0][0][[0],index]
    #
    # for i in range(count):
    #     if batch1[i]==batch2[i]:
    #         repeat += 1
    # print('重复率：{}'.format(repeat/count))

    get_msrpc_manager(re_build=False)

if __name__ == '__main__':
    test()