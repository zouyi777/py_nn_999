from py_nn_999.net import seq2seq_cnn2lstm_net
from tensorflow import keras
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
from py_nn_999.utils import pre_data_utils
import numpy as np



# 源数据(编码器数据)处理
dataList = pre_data_utils.preDataAdd1(data_zixuan.getTrainData(), pre_data_utils.piece_len)
# 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
encoder_input = pre_data_utils.guiyiAndChannel(dataList)

# 目标数据（解码器数据）处理
tgt_in,tgt_out = pre_data_utils.preLabels1(data_zixuan.getTrainData(), pre_data_utils.piece_len) #数据载入
decoder_input = np.array(tgt_in)
decoder_output = pre_data_utils.gen_sequence_targt(tgt_out,pre_data_utils.tgt_out_len,pre_data_utils.tgt_vocab_size)


model_train = seq2seq_cnn2lstm_net.Seq2Seq_CNN2LSTM(pre_data_utils.piece_len)
# optimizer = keras.optimizers.RMSprop(0.01)
optimizer = keras.optimizers.Adam(0.01)
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
# model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=pre_data_utils.BATCH_SIZE,
#                 epochs=pre_data_utils.EPOCH,validation_split=0.1)
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=pre_data_utils.BATCH_SIZE,epochs=pre_data_utils.EPOCH)
model_train.save_weights("model/seq2seq_cnn2lstm直选.h5")

