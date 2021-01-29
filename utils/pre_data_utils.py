import numpy as np

# 定义一些常量
BATCH_SIZE = 512
EPOCH = 600
piece_len = 10
tgt_out_len = 4
tgt_vocab_size = 12
hidden_units = 128
embedding_dim = 150

train_first_index = 0
train_last_index = -20  # 从0至倒数第10个样本作为训练，剩下的作为测试样本
test_last_index = -30  # 从0至倒数第10个样本作为训练，剩下的作为测试样本

# 如果只有两位数，前面补一个0
def appendTo3(label_sequence_str):
    label_sequence_str = str(label_sequence_str)
    label_sequence = []
    label_sequence.append(int(label_sequence_str[0]))  # A
    label_sequence.append(int(label_sequence_str[1]))  # B
    label_sequence.append(int(label_sequence_str[2]))  # C
    return sorted(label_sequence) # 组选的话，从小到大排个序
    # return label_sequence

# 如果只有两位数，前面补一个1
def appendTo3Add1(label_sequence_str):
    label_sequence_str = str(label_sequence_str)
    label_sequence = []
    label_sequence.append(int(label_sequence_str[0]))  # A
    label_sequence.append(int(label_sequence_str[1]))  # B
    label_sequence.append(int(label_sequence_str[2]))  # C
    sorted_list = sorted(label_sequence) # 组选的话，从小到大排个序
    # sorted_list = label_sequence

    result_sequence = []
    for i in range(len(sorted_list)):
        bai = sorted_list[i]

        # 邻码
        bai_left = bai - 1
        if bai_left < 0:
            bai_left = 9
        bai_right = bai + 1
        if bai_right > 9:
            bai_right = 0

        # 对码
        dui = bai
        if dui < 5:
            dui = dui + 5
        else:
            dui = dui - 5

        # 跨4码
        # kua3_left = bai - 4
        # if kua3_left < 0:
        #     kua3_left = kua3_left + 10
        # kua3_right = bai + 4
        # if kua3_right > 9:
        #     kua3_right = kua3_right - 10

        # item = [bai_left, bai, bai_right]
        item = [bai, bai_left, bai_right,dui]
        # item = [bai, bai_left, bai_right, kua3_left, kua3_right, dui]
        result_sequence.append(item)
    return result_sequence

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
    # encoder_input = np.expand_dims(np.array(dataList).astype(np.float32) / 10.0, axis=-1)
    encoder_input = np.array(dataList).astype(np.float32) / 10.0
    return encoder_input

#  定义训练标签预处理函数
def preLabels(src_list,piece_len):
    tgt_in_list = []
    tgt_out_list = []
    for j in range(len(src_list)-piece_len):
        src = int(src_list[j + piece_len])
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
        tgt_in.insert(0, 10) # 开头添加10作为起始元素
        tgt_in_list.append(tgt_in)

        tgt_out = appendTo3(src)
        tgt_out.append(11)  # 结尾添加11作为结束元素
        tgt_out_list.append(tgt_out)
    return tgt_in_list, tgt_out_list
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

# 读取txt类型的训练数据
def read_txt_data_train():
    data_path = 'dataLib/nn_999_data.txt'
    source_list = []
    with open(data_path, "r", encoding='utf-8', ) as f:
        for line in f:
            content = line.replace(' ', '')
            content = content.replace('\n', '')  # 去掉换行符
            source_list.append(content)
    return source_list

# 读取txt类型的预测数据
def read_txt_data_predict():
    data_path = 'dataLib/nn_999_data.txt'
    source_list = []
    add1_list = []
    with open(data_path, "r", encoding='utf-8', ) as f:
        for line in f:
            content = line.replace(' ', '')
            content = content.replace('\n', '')  # 去掉换行符
            source_list.append(content)
            add1 = appendTo3Add1(content)
            add1_list.append(add1)
    source_list = source_list[-piece_len:]  # 取最后piece_len个数据进行预测
    add1_list = add1_list[-piece_len:]  # 取最后piece_len个数据进行预测
    return [source_list], [add1_list]


if __name__ == '__main__':
    source_list = read_txt_data_train()[6756:]
    print(source_list)
    source_list = preDataAdd1(source_list, piece_len)
    print(source_list)