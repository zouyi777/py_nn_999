from py_nn_999.net import net_model
from py_nn_999.dataLib import data_zixuan_train as data_zixuan
import numpy as np
import heapq

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

# 向量化
def gen_sequence(sentence_list,sentence_len,vocab_size):
    sequence =[]
    for _,seq in enumerate(sentence_list):
        sentence_vector = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sentence_vector[char_index, char] = 1
        sequence.append(sentence_vector.tolist())
    return np.array(sequence)

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
    enc_state_outputs = [enc_state_h, enc_state_c]

    predict_seq = [[10]]
    states_curr = {0: enc_state_outputs}
    seq_scores = [[predict_seq, 1.0, 0]]
    resList = []
    for _ in range(n_steps):
        cands = list()
        states_prev = states_curr
        for i in range(len(seq_scores)):
            seq, score, state_id = seq_scores[i]
            dec_inputs = np.array(seq[-1:])
            dec_states_inputs = states_prev[state_id]
            # 解码器decoder预测输出
            dense_outputs, dec_state_h, dec_state_c = decoder_inference.predict([enc_outputs, dec_inputs] + dec_states_inputs)
            prob = dense_outputs[0][0]
            states_curr[i] = [dec_state_h, dec_state_c]

            k_prob = heapq.nlargest(k,prob)
            list_prob = prob.tolist()
            for item_prob in k_prob:
                cand = [seq + [[list_prob.index(item_prob)]], score * item_prob, i]
                cands.append(cand)
        seq_scores = heapq.nlargest(k, cands, lambda d: d[1])
    for i in range(len(seq_scores)):
        res = []
        for item in seq_scores[i][0]:
            if item[0] != 10 and item[0] != 11:
                res.append(item[0])
        resList.append("".join(str(res)))
    return resList

model_train = net_model.Seq2Seq(12)
model_train.load_weights("model/seq2seq_直选.h5")
encoder_infer = net_model.encoder_infer(model_train)
decoder_infer = net_model.decoder_infer(model_train,encoder_infer)

texts = [[0, 7, 3, 2, 3, 7, 0, 7, 1]]
# encoder_input = gen_sequence(texts,piece_len * 3,10)
# texts = [preData(data_zixuan.getTrainData(), piece_len)[0]]
encoder_input = np.array(texts)
out = infer_beam_search(encoder_input,encoder_infer,decoder_infer,3,5)
print(texts)
for item in out:
    print(item)

