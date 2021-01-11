from py_nn_999.net import seq2seq_lstm_net
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
from py_nn_999.dataLib import data_zixuan_eval as data_eval
import numpy as np
import heapq

piece_len = 100
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

def preLabels(src_list,piece_len):
    tgt_out = []
    for j in range(len(src_list)-piece_len):
        src = appendTo3(src_list[j + piece_len])
        tgt_out.append(src)
    return tgt_out
def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    _, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    #第一个10,为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,10] = 1

    output = []
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        if char_index == 11:#预测到了终止符则停下来
            break
        output.append(char_index)
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
    return output

# 预测均衡输出
def infer_beam_search(source,encoder_inference, decoder_inference, n_steps, k=3):
    # 先通过推理encoder获得预测输入序列的隐状态
    enc_outputs, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    # 第一个字符'\t',为起始标志
    predict_seq = np.array([[1000]])
    output = []
    # 开始对encoder获得的隐状态进行推理
    # 每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps - 1):  # n_steps为句子最大长度
        # 给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat, h, c = decoder_inference.predict([enc_outputs, predict_seq] + state)
        # yhat,h,c = decoder_inference.predict([predict_seq]+state)
        # 注意，这里的yhat为Dense之后输出的结果，因此与h不同
        state = [h, c]  # 本次状态做为下一次的初始状态继续传递
        list_prob = yhat[0][0].tolist()
        probs_k = heapq.nlargest(k, list_prob)
        prob_indexs = [list_prob.index(prob_k) for prob_k in probs_k]
        # predict_seq = np.array([[char_index]])
        # if char_index == 1:#预测到了终止符则停下来
        #     break
        for prob_index in prob_indexs:
            if prob_index == 1001:
                continue
            poetry = prob_index
            output.append(poetry)
    return output

model_train = seq2seq_lstm_net.Seq2Seq(src_vocab_size,tgt_vocab_size)
model_train.load_weights("model/seq2seq_直选.h5")
encoder_infer = seq2seq_lstm_net.encoder_infer(model_train)
decoder_infer = seq2seq_lstm_net.decoder_infer(model_train,encoder_infer)

# texts = [[8,8,3, 5,3,1]]
texts = [preData(data_eval.getEvalData(), piece_len)[4]]
# tgt_out = preLabels(data_eval.getEvalData(), piece_len) #数据载入
for i,text in enumerate(texts):
    encoder_input = np.array([text])
    out = infer_beam_search(encoder_input, encoder_infer, decoder_infer, tgt_out_len,5)
    print(texts)
    print(out)


