from py_nn_999.net import seq2seq_cnn2lstm_net
from tensorflow import keras
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
from py_nn_999.utils import pre_data_utils
import numpy as np


# 源数据(编码器数据)处理
dataList = pre_data_utils.preDataAdd1(pre_data_utils.read_txt_data_train(), pre_data_utils.piece_len)
dataList = dataList[pre_data_utils.train_first_index:pre_data_utils.train_last_index]  # 从0至倒数第10个样本作为训练，剩下的作为测试样本
# 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为通道
encoder_input = pre_data_utils.guiyiAndChannel(dataList)

# 目标数据（解码器数据）处理
tgt_in,tgt_out = pre_data_utils.preLabels1(pre_data_utils.read_txt_data_train(), pre_data_utils.piece_len) #数据载入
tgt_in = tgt_in[pre_data_utils.train_first_index:pre_data_utils.train_last_index]  # 从0至倒数第10个样本作为训练，剩下的作为测试样本
tgt_out = tgt_out[pre_data_utils.train_first_index:pre_data_utils.train_last_index] # 从0至倒数第10个样本作为训练，剩下的作为测试样本
decoder_input = np.array(tgt_in)
decoder_output = pre_data_utils.gen_sequence_targt(tgt_out,pre_data_utils.tgt_out_len,pre_data_utils.tgt_vocab_size)


model_train = seq2seq_cnn2lstm_net.Seq2Seq_CNN2LSTM(pre_data_utils.piece_len)
optimizer = keras.optimizers.Adam(0.005)  # 经过测试，0.005的学习率对目前任务效果最佳
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=pre_data_utils.BATCH_SIZE,epochs=pre_data_utils.EPOCH)
model_train.save_weights("model/seq2seq_cnn2lstm直选txt3.h5")

