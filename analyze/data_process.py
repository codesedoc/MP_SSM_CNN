import sys
import re

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

    for momentum_match in momentum_match_result1:
        re_pattern = re.compile(r'')
    pass

if __name__ == '__main__':
    process_data()
