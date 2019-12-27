import utils.data_tool as data_tool
import utils.file_tool as file_tool
import utils.data_tool as data_tool
import re
import random


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def extra_parsing_head(original_str):
    result = {}
    re_pattern = 'Parsing *\[sent.*\]: *'
    result['sentence_split'] = re.split(re_pattern,original_str)[1].split(' ')
    re_pattern = '[0-9]+'
    number_list = re.findall(re_pattern, original_str)
    result['sentence_index'] = int(number_list[0])
    # result['sentence_length'] = int(number_list[1])
    return result


def extra_parsing_dependency(original_str):
    result = {}
    re_pattern = '\('
    temp = re.split(re_pattern, original_str)
    result['name'] = temp[0]
    word_pair = {}
    word_pair_raw = temp[1][:-1]
    word_index_raw_list = word_pair_raw.split(',')
    if len(word_index_raw_list)!=2:
        re_pattern = '-[0-9]+\'*,.+-'
        temp = re.split(re_pattern, word_pair_raw)
        if len(temp) != 2:
            raise ValueError
        first_temp = temp[0]
        second_temp = temp[1]
        temp = re.findall(re_pattern, word_pair_raw)
        if len(temp) != 1:
            raise ValueError
        temp = temp[0].split(',')
        first_temp += temp[0]
        second_temp = ','.join(temp[1:]) + second_temp
        word_index_raw_list = [first_temp, second_temp]

    word_index_list = []
    for raw in word_index_raw_list:
        word_index_temp = raw.split('-')
        word = '-'.join(word_index_temp[:-1])
        try:
            index = word_index_temp[-1]
        except Exception :
            raise
        word_index_list.append({'word': word, 'index': index})
    word_pair['first'] = word_index_list[0]
    word_pair['second'] = word_index_list[1]
    result['word_pair'] = word_pair
    return result

def extra_parsing_info(parsing_itme_list):
    result = {}
    result['head'] = extra_parsing_head(parsing_itme_list[0])
    dependency_list = []
    for item in parsing_itme_list[1:]:
        dependency_list.append(extra_parsing_dependency(item))
    result['dependency_list'] = dependency_list
    return result


def extra_pasing_info(original_content):
    result = []
    parsing_temp = []
    for content in original_content:
        if content == '\n':
            result.append(extra_parsing_info(parsing_temp))
            parsing_temp = []
        else:
            parsing_temp.append(content.replace('\n', '').replace('\r', ''))
    return result


def check_id_sentence(parsing_info):
    result = True
    head = parsing_info['head']
    if len(head['sentence_split'])!=4:
        result = False
    else:
        for word in head['sentence_split'][:-1]:
            if not is_number(word):
                result = False
                break
    return result

def create_sentence(parsing_info):
    head = parsing_info['head']
    sentence = {'words': head['sentence_split'],
                'dependencies': parsing_info['dependency_list']}
    return sentence

def create_example_head(parsing_info):
    head = parsing_info['head']
    temp = head['sentence_split'][:-1]
    if int(temp[1]) == 2452454:
        temp =temp
    return {'id1': int(temp[1]),
            'id2': int(temp[2]),
            'label': int(temp[0])}

def create_sentence_pair(parsing_info_dict):
    example_head = create_example_head(parsing_info_dict['example_head'])
    sentence1 = parsing_info_dict['sentence1']
    sentence2 = parsing_info_dict['sentence2']

    sentence1 = create_sentence(sentence1)
    sentence2 = create_sentence(sentence2)
    sentence1['id'] = example_head['id1']
    sentence2['id'] = example_head['id2']
    return sentence1,sentence2


def deal_with_info(filename):
    def deal_with_info_temp_list():
        if len(pasing_info_temp) != 3:
            example_head = create_example_head(pasing_info_temp[0])
            error_example_parsing_list.append({
                'id1': example_head['id1'],
                'id2': example_head['id2']
            })
        else:
            example_info_temp = {
                'example_head': pasing_info_temp[0],
                'sentence1': pasing_info_temp[1],
                'sentence2': pasing_info_temp[2]
            }
            right_example_parsing_list.append(create_sentence_pair(example_info_temp))
    error_example_parsing_list = []
    right_example_parsing_list = []
    original_content = file_tool.load_data(filename,'r')
    pasing_info_list = extra_pasing_info(original_content)
    pasing_info_temp = []
    for i, p in enumerate(pasing_info_list):
        if i == len(pasing_info_list)-1:
            i=i
        if check_id_sentence(p):
            if pasing_info_temp == []:
                pasing_info_temp.append(p)
                continue
            deal_with_info_temp_list()
            pasing_info_temp = []
        pasing_info_temp.append(p)
    deal_with_info_temp_list()
    print('ok')
    return error_example_parsing_list, right_example_parsing_list


def process_info_func():
    def save_parsing_sentence(name1,name2):
        error_sentence_ids = []
        for error_example in error_example_parsing_list:
            error_sentence_ids.extend(error_example.values())

        error_sentence_ids = list(set(error_sentence_ids))
        right_sentence_dict = {}
        for sent_pair in right_example_parsing_list:
            for sent in sent_pair:
                id = str(sent['id'])
                right_sentence_dict[id]=sent
        file_tool.save_data_pickle(error_sentence_ids, name1)
        file_tool.save_data_pickle(right_sentence_dict, name2)

    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)

    error_example_parsing_list, right_example_parsing_list = deal_with_info('/home/sheng/Documents/study/workspace/python/MP_SSM_CNN /data/msrpc/test_dependency_output.txt')
    sentence_id_pairs = test_manager.sentence_id_pairs
    for id_pair in sentence_id_pairs:
        find_flag = False
        for ep in error_example_parsing_list:
            if (ep['id1'],ep['id2']) == id_pair:
                find_flag =True
                break
        for ep in right_example_parsing_list:
            if (ep[0]['id'],ep[1]['id']) == id_pair:
                find_flag = True
                break
        if not find_flag:
            print(id_pair)

    save_parsing_sentence('test_dependency_error.pkl', 'test_dependency_right.pkl')
    error_example_parsing_list, right_example_parsing_list = deal_with_info(
        "/home/sheng/Documents/study/workspace/python/MP_SSM_CNN /data/msrpc/train_dependency_output.txt")
    sentence_id_pairs = train_manager.sentence_id_pairs
    for id_pair in sentence_id_pairs:
        find_flag = False
        for ep in error_example_parsing_list:
            if (ep['id1'],ep['id2']) == id_pair:
                find_flag =True
                break
        for ep in right_example_parsing_list:
            if (ep[0]['id'],ep[1]['id']) == id_pair:
                find_flag = True
                break
        if not find_flag:
            print(id_pair)
    save_parsing_sentence('train_dependency_error.pkl', 'train_dependency_right.pkl')
    pass



test_dependency_error = file_tool.load_data_pickle('test_dependency_error.pkl')
train_dependency_error = file_tool.load_data_pickle('train_dependency_error.pkl')

test_dependency_right = file_tool.load_data_pickle('test_dependency_right.pkl')
train_dependency_right = file_tool.load_data_pickle('train_dependency_right.pkl')

def create_error_sentence_file():
    error_id_temp = []
    error_id_temp.extend(test_dependency_error)
    error_id_temp.extend(train_dependency_error)
    error_id_temp = map(str,error_id_temp)
    error_id_temp = list(set(error_id_temp))

    right_id_temp = {}
    right_id_temp.update(test_dependency_right)
    right_id_temp.update(train_dependency_right)

    key_temp = right_id_temp.keys()
    temp = []
    for e in error_id_temp:
        if e not in key_temp:
            temp.append(e)
    error_id_temp = temp

    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
    manager_sentence_dict = train_manager.original_sentence_dict.copy()
    manager_sentence_dict.update(test_manager.original_sentence_dict)

    # test_temp = list(test_dependency_right.copy().keys())
    # test_temp.extend(test_dependency_error)
    # test_temp = map(str, test_temp)
    # test_temp = list(set(test_temp))
    #
    #
    # train_temp = list(train_dependency_right.copy().keys())
    # train_temp.extend(train_dependency_error)
    # train_temp = map(str, train_temp)
    # train_temp = list(set(train_temp))
    #
    # count = 0
    #
    # for i,e in enumerate(test_temp):
    #     if e == "633744" :
    #             print(e, i)
    #
    # dit_temp = {}
    # for i,tt in enumerate(test_temp):
    #     count += 1
    #
    #     if str(tt) in dit_temp:
    #         print(dit_temp[str(tt)],i,tt)
    #     tt = str(tt)
    #     dit_temp[tt]=i
    #     t= test_manager.original_sentence_dict[tt]
    #     if str(tt) not in test_manager.original_sentence_dict:
    #         print(tt)
    #
    # count =0
    #
    # dit_temp = {}
    # for i,tt in enumerate(train_temp):
    #     count += 1
    #     if str(tt) in dit_temp:
    #         print(dit_temp[str(tt)],i,tt)
    #
    #     tt = str(tt)
    #     dit_temp[tt]=i
    #     t = train_manager.original_sentence_dict[str(tt)]
    #     if str(tt) not in train_manager.original_sentence_dict:
    #         print(tt)

    if len(error_id_temp)+len(right_id_temp) != len(manager_sentence_dict.keys()):
        temp = temp

    save_data = []
    for sent_id in error_id_temp:
        save_data.append(sent_id+'.')
        save_data.append(manager_sentence_dict[sent_id])
    file_tool.save_list_data(save_data,'error_paring_sentences.txt','w')
    file_tool.save_data_pickle(error_id_temp, 'parsing_error_sentence.pkl')
    file_tool.save_data_pickle(right_id_temp, 'parsing_right_sentence.pkl')

def check_id_error_sentence(parsing_info):
    result = True
    head = parsing_info['head']
    if len(head['sentence_split'])!=2:
        result = False
    else:
        for word in head['sentence_split'][:-1]:
            if not is_number(word):
                result = False
                break
    return result
def deal_with_error_sentence_file(filename):
    original_content = file_tool.load_data(filename, 'r')
    pasing_info_list = extra_pasing_info(original_content)
    parsing_sentence_dict = {}
    for i, p in enumerate(pasing_info_list[:-1]):
        next_p = pasing_info_list[i+1]
        if check_id_error_sentence(p) and (not check_id_error_sentence(next_p)) :
            sent_id = str(p['head']['sentence_split'][0])
            parsing_info = {}
            parsing_info['words'] = next_p['head']['sentence_split']
            parsing_info['dependencies'] = next_p['dependency_list']
            parsing_info['id'] = sent_id
            parsing_sentence_dict[sent_id] = parsing_info

    parsing_error_sentence_id = file_tool.load_data_pickle('parsing_error_sentence.pkl')
    for sent_id in parsing_sentence_dict.keys():
        if sent_id not in parsing_error_sentence_id:
            print(sent_id)
    pass
    file_tool.save_data_pickle(parsing_sentence_dict, "repeat_parsing_sentence.pkl")

def create_parsing_sentence_dict():
    parsing_sentence_dict = {}
    repeat_parsing_sentence = file_tool.load_data_pickle("repeat_parsing_sentence.pkl")
    parsing_right_sentence = file_tool.load_data_pickle("parsing_right_sentence.pkl")
    parsing_sentence_dict.update(repeat_parsing_sentence)
    parsing_sentence_dict.update(parsing_right_sentence)

    file_tool.save_data_pickle(parsing_sentence_dict, '/home/sheng/Documents/study/workspace/python/MP_SSM_CNN /data/msrpc/parsing_sentence_dict.pkl')
    pass
# deal_with_error_sentence_file('repeate_error_parsing_sentence.txt')
# process_info_func()
# create_error_sentence_file()
# re_pattern = '\('
# result = re.split(re_pattern, 'nmod:poss(officer-5, PCCW-1)')
# # re_pattern = '\(.+\)'
# # result = re.findall(re_pattern, 'nmod:poss(officer-5, PCCW-1)')
# result=result
create_parsing_sentence_dict()


def test():
    parsing_sentence_dict = file_tool.load_data_pickle('/home/sheng/Documents/study/workspace/python/MP_SSM_CNN /data/msrpc/parsing_sentence_dict.pkl')
    train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
    manager_sentence_dict = train_manager.original_sentence_dict.copy()
    manager_sentence_dict.update(test_manager.original_sentence_dict)
    if len(parsing_sentence_dict) != len(manager_sentence_dict):
        raise ValueError
    for sent_id in parsing_sentence_dict.keys():
        if sent_id not in manager_sentence_dict:
            raise ValueError
    one_key = list(parsing_sentence_dict.keys())[random.randint(0,len(parsing_sentence_dict))]
    print('{}: {}'.format(one_key, parsing_sentence_dict[one_key]))
    print('{}: {}'.format(one_key,manager_sentence_dict[one_key]))
    print('parsing_process is ok!')

test()
