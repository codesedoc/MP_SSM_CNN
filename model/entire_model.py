import model.sentence_model as sentence_model
import model.similarity_measure_model as similarity_measure_model
import model.full_connect_model as full_connect_model
import torch
import model as model_init
import model
import time


class EntireModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sentence_model = sentence_model.SentenceModel()
        self.similarity_measure_model = similarity_measure_model.SimilarityMeasureModel()
        self.full_connect_model = full_connect_model.FullConnectModel()

    def forward_inner(self, input_data1, input_data2):
        if model_init.show_run_time:
            start = time.time()
            result1 = self.sentence_model(input_data1)
            result2 = self.sentence_model(input_data2)
            end = time.time()
            print("sentence_model_run_time:{}".format(end - start))
        else:
            result1 = self.sentence_model(input_data1)
            result2 = self.sentence_model(input_data2)

        if model_init.show_run_time:
            start = time.time()
            result = self.similarity_measure_model(result1,result2)
            end = time.time()
            print("similarity_measure_model_run_time:{}".format(end - start))
        else:
            result = self.similarity_measure_model(result1,result2)
        return result

    def forward(self, input_data1, input_data2):
        result = []
        for select in model.sub_model_select_dict:
            block_model = select['block_model']
            compare_model = select['compare_model']
            self.sentence_model.change_current_block(block_model)
            self.similarity_measure_model.change_current_compare_model(compare_model)
            result.append(self.forward_inner(input_data1, input_data2))
        result = torch.cat(result, dim= -1)
        if model_init.show_run_time:
            start = time.time()
            result = self.full_connect_model(result)
            end = time.time()
            print("full_connect_model_run_time:{}".format(end - start))
        else:
            result = self.full_connect_model(result)

        return result

    # def cuda(self, *args, **kwargs):
    #     super().cuda(*args, **kwargs)


