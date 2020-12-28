from py_nn_999.net import net_model
from tensorflow import keras
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
import numpy as np

BATCH_SIZE = 64
EPOCH = 100
piece_len = 3

# 如果只有一位数，前面补两个0
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

# 定义训练集预处理函数
def preData(src_list,piece_len):
    pre_list = []
    for i in range(len(src_list)-piece_len):
        item = []
        for j in range(piece_len):
            item = item + appendTo3(src_list[i+j])
        pre_list.append(item)
    return pre_list
#  定义训练标签预处理函数
def preLabels(src_list,piece_len):
    pre_labels = []
    for j in range(len(src_list)-piece_len):
        src = appendTo3(src_list[j + piece_len])
        src.insert(0, 10)  # 开头添加10作为起始元素
        src.append(11)  # 结尾添加11作为结束元素
        pre_labels.append(src)
    return pre_labels

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
    sequence_in =[]
    sequence_out =[]
    for _,seq in enumerate(sentence_list):
        sen_vec_in = np.zeros((sentence_len, vocab_size))
        sen_vec_out = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sen_vec_in[char_index, char] = 1.0
            if char_index > 0:
                sen_vec_out[char_index - 1, char] = 1.0
        sequence_in.append(sen_vec_in.tolist())
        sequence_out.append(sen_vec_out.tolist())
    return np.array(sequence_in),np.array(sequence_out)

dataList = preData(data_zixuan.getTrainData(), piece_len)
labelList = preLabels(data_zixuan.getTrainData(), piece_len) #数据载入
# 向量化
# encoder_input = gen_sequence(dataList,piece_len * 3 ,10)
encoder_input = np.array(dataList)
decoder_input = np.array(labelList)
_,decoder_output = gen_sequence_targt(labelList,5,12)


model_train = net_model.Seq2Seq(12)
# optimizer = keras.optimizers.RMSprop(0.01)
optimizer = keras.optimizers.Adam(0.01)
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.1)
model_train.save_weights("model/seq2seq_直选.h5")

