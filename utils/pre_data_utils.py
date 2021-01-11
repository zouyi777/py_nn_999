import numpy as np

# 定义一些常量
BATCH_SIZE = 512
EPOCH = 200
piece_len = 10
tgt_out_len = 4
tgt_vocab_size = 12

# 如果只有两位数，前面补一个0
def appendTo3(label_sequence_str):
    label_sequence_str = str(label_sequence_str)
    label_sequence = []
    if len(label_sequence_str) == 1:  # 如果只有一位数，前面补两个0
        pre_zero2 = [0, 0]
        pre_zero2.append(int(label_sequence_str[0]))
        label_sequence = pre_zero2
    elif len(label_sequence_str) == 2:  # 如果只有两位数，前面补一个0
        pre_zero1 = [0]
        pre_zero1.append(int(label_sequence_str[0]))
        pre_zero1.append(int(label_sequence_str[1]))
        label_sequence = pre_zero1
    else:
        label_sequence.append(int(label_sequence_str[0]))
        label_sequence.append(int(label_sequence_str[1]))
        label_sequence.append(int(label_sequence_str[2]))
    return label_sequence

# 如果只有两位数，前面补一个1
def appendTo3Add1(label_sequence_str):
    label_sequence_str = str(label_sequence_str)
    label_sequence = []
    if len(label_sequence_str) == 1:  # 如果只有一位数，前面补两个1
        pre_zero2 = [1, 1]
        pre_zero2.append(int(label_sequence_str[0]) + 1)
        label_sequence = pre_zero2
    elif len(label_sequence_str) == 2:  # 如果只有两位数，前面补一个1
        pre_zero1 = [1]
        pre_zero1.append(int(label_sequence_str[0]) + 1)
        pre_zero1.append(int(label_sequence_str[1]) + 1)
        label_sequence = pre_zero1
    else:
        label_sequence.append(int(label_sequence_str[0]) + 1)
        label_sequence.append(int(label_sequence_str[1]) + 1)
        label_sequence.append(int(label_sequence_str[2]) + 1)
    return label_sequence

# 定义训练集预处理函数
def preDataAdd0(src_list,piece_len):
    pre_list = []
    for i in range(len(src_list)-piece_len):
        item = []
        for j in range(piece_len):
            appendTo3Str = src_list[i+j]
            appendTo3List = appendTo3(appendTo3Str)
            item.append(appendTo3List)
        pre_list.append(item)
    return pre_list

def preDataAdd1(src_list,piece_len):
    pre_list = []
    for i in range(len(src_list)-piece_len):
        item = []
        for j in range(piece_len):
            appendTo3Str = src_list[i+j]
            appendTo3List = appendTo3Add1(appendTo3Str)
            item.append(appendTo3List)
        pre_list.append(item)
    return pre_list

# 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
def guiyiAndChannel(dataList):
    encoder_input = np.expand_dims(np.array(dataList).astype(np.float32) / 10.0, axis=-1)
    return encoder_input

#  定义训练标签预处理函数
def preLabels(src_list,piece_len):
    tgt_in_list = []
    tgt_out_list = []
    for j in range(len(src_list)-piece_len):
        src = src_list[j + piece_len]
        tgt_in = [1000,src] # 开头添加1000作为起始元素
        tgt_in_list.append(tgt_in)
        tgt_out = [src,1001] # 结尾添加11作为结束元素
        tgt_out_list.append(tgt_out)
    return tgt_in_list,tgt_out_list

def preLabels1(src_list,piece_len):
    tgt_in_list = []
    tgt_out_list = []
    for j in range(len(src_list)-piece_len):
        src = src_list[j + piece_len]

        tgt_in = appendTo3(src)
        tgt_in.insert(0,10) # 开头添加10作为起始元素
        tgt_in_list.append(tgt_in)

        tgt_out = appendTo3(src)
        tgt_out.append(11)  # 结尾添加11作为结束元素
        tgt_out_list.append(tgt_out)
    return tgt_in_list,tgt_out_list
# 向量化
def gen_sequence(sentence_list,sentence_len,vocab_size):
    sequence =[]
    for _,seq in enumerate(sentence_list):
        sentence_vector = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sentence_vector[char_index, char] = 1
        sequence.append(sentence_vector.tolist())
    return np.array(sequence)

# 生成目标句向量,返回数据是3维的
def gen_sequence_targt(sentence_list,sentence_len,vocab_size):
    sequence_out =[]
    for _,seq in enumerate(sentence_list):
        sen_vec_out = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sen_vec_out[char_index, char] = 1.0
        sequence_out.append(sen_vec_out.tolist())
    return np.array(sequence_out)
