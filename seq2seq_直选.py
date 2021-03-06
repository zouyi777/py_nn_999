from py_nn_999.net import seq2seq_lstm_net
from tensorflow import keras
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
import numpy as np

BATCH_SIZE = 512
EPOCH = 50
piece_len = 5
tgt_out_len = 2
src_vocab_size = 1000
tgt_vocab_size = 1002

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
            item.append(src_list[i+j])
        pre_list.append(item)
    return pre_list
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

dataList = preData(data_zixuan.getTrainData(), piece_len)
tgt_in,tgt_out = preLabels(data_zixuan.getTrainData(), piece_len) #数据载入
# 向量化
# encoder_input = gen_sequence(dataList,piece_len * 3 ,10)
encoder_input = np.array(dataList)
decoder_input = np.array(tgt_in)
decoder_output = gen_sequence_targt(tgt_out,tgt_out_len,tgt_vocab_size)


model_train = seq2seq_lstm_net.Seq2Seq(src_vocab_size,tgt_vocab_size)
# optimizer = keras.optimizers.RMSprop(0.01)
optimizer = keras.optimizers.Adam(0.01)
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.1)
model_train.save_weights("model/seq2seq_直选.h5")

