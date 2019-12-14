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
        poolings = model.poolings
        block_types = model.block_types
        compare_model_names = model.compare_model_names
        compare_unit_names = model.compare_unit_names
        self.sentence_model = sentence_model.SentenceModel(block_types=block_types)
        self.similarity_measure_model = similarity_measure_model.SimilarityMeasureModel(compare_model_names=compare_model_names,
                                                            compare_unit_names=compare_unit_names)

        #
        self.full_connect_model = full_connect_model.FullConnectModel(model.full_layer_scale_a)

        self.sub_part_select = [
            ('BlockA', 'Horizontal')
        ]

    def forward(self, input_data1, input_data2):
        for select in self.sub_part_select:
            self.sentence_model.change_current_block('BlockA')
            self.similarity_measure_model.change_current_compare_model('VerticalForBlockA')
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


