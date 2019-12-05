import utils.data_tool as data_tool


ws_endless_flag = True
input_align = True
num_filter_a = 525
train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
ws_max = max(train_manager.get_max_length_of_sentence(), test_manager.get_max_length_of_sentence())


class Preprocessor:
    def __init__(self):
        pass

    def process(self, data_manager):
