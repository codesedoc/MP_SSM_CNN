import sys
import re
import numpy as np
import utils.file_tool as file_tool


def process_data():
    data = file_tool.load_data('train_result.txt', model='r')
    data = '#'.join(data)
    data = data.replace('\n', '#')
    re_pattern = re.compile(r'with|without')
    match_result = re.split(re_pattern, data)

    re_pattern = re.compile(r'momentum *= *')
    momentum_match_result1 = re.split(re_pattern,match_result[1])
    momentum_match_result2 = re.split(re_pattern,match_result[2])
    result = []
    for momentum_match in momentum_match_result2[1:]:
        momentum_pattern = re.compile(r'.+?#')
        epoch_pattern = re.compile(r'epoch:[0-9]+')
        value_pattern = re.compile(r'[0-9]+\.[0-9]{2,}')
        momentum = float(re.match(momentum_pattern, momentum_match).group(0)[:-1])
        epochs = [int(x.split(':')[-1]) for x in epoch_pattern.findall(momentum_match)]
        values =[float(x) for x in value_pattern.findall(momentum_match)]
        result_momentum = {}
        result_momentum['momentum'] = momentum
        records = []

        for i, e in enumerate(epochs):
            record = []
            record.append(e)
            record.extend(values[2*i:2*(i+1)])
            records.append(record)
        records = np.array(records, dtype=np.float)
        result_momentum['records'] = records
        result.append(result_momentum)
    return result

if __name__ == '__main__':
    without_test_data = process_data()
    # file_tool.save_data_pickle(without_test_data, 'without_test_data.pkl')
    pass
