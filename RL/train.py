import tensorflow as tf
import numpy as np
from model import Seq2Seq
from loader import Loader
import pickle
import os
import shutil
from id2sent import id2sent
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

checkpoint=False

def main():
    max_seq_len = 20
    # voca_size = 10000
    voca_size = 800
    embed_size = 300
    # rnn_size = 256
    rnn_size = 1024
    n_layers = 3

    n_epoch = 300
    start_epoch = 0
    batch_size = 32
    learning_rate = 1e-6
    # learning_rate = 1e-3

    work_dir = 'RL/'


    s2s = Seq2Seq(max_seq_len, voca_size, embed_size, rnn_size, n_layers)

    input_tensors, output_tensors = s2s.build_model()

    text_input = input_tensors['text_input']
    input_len = input_tensors['input_len']
    target = input_tensors['target']
    reward = input_tensors['reward']

    loss = output_tensors['loss']
    text_output = output_tensors['text_output']

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    voca_size = s2s.voca_size
    max_seq_len = s2s.max_seq_len

    if checkpoint:
        loader = pickle.load(open(work_dir + 'models/loader.p', 'rb'))
        print('Loaded RL model pre-trained')
    else:
        if os.path.exists(work_dir + 'loss_log'):
            os.remove(work_dir + 'loss_log')
        if os.path.exists(work_dir + 'models'):
            shutil.rmtree(work_dir + 'models')
        os.mkdir(work_dir + 'models')

        # loader = pickle.load(open(work_dir + 'seq2seq/models/loader.p', 'rb'))
        # load seq2seq model pre-trained
        loader = pickle.load(open('seq2seq/models/loader.p', 'rb'))
        print('Loaded Seq2Seq model pre-trained')
        # store the loader for test
        pickle.dump(loader, open(work_dir + 'models/loader.p', 'wb'))# 

    # save graph model for test
    pickle.dump(s2s, open(work_dir + 'models/s2s.p', 'wb'))
    dic = loader.dic
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        saver = tf.train.Saver()
        if checkpoint:
            saver_path = tf.train.latest_checkpoint(work_dir + 'models')
        else:
            saver_path = tf.train.latest_checkpoint('seq2seq/models')

        saver.restore(sess, saver_path)

        for epoch in range(start_epoch, n_epoch):
            batch_loss = []
            for i, (ques, lens, ans) in enumerate(loader.train_data(batch_size=batch_size)):
                
                # seq2seq step
                feed = {
                    text_input: ques,
                    input_len: lens
                }

                _output = sess.run(text_output, feed_dict=feed)

                # reinforcement step TODO
                _output_sent = id2sent(dic, _output)
                _ans_sent = id2sent(dic, ans)
                _reward = rewardfunc(_output_sent, _ans_sent)

                feed = {
                    text_input: ques,
                    input_len: lens,
                    target: _output,
                    reward: _reward
                }

                _loss, _ = sess.run([loss, optimizer], feed_dict=feed)
                batch_loss.append(_loss)

                if (i+1) % 100 == 0:
                    print(np.mean(batch_loss))

            epoch_loss = np.mean(batch_loss)
            epoch_rl_loss = np.mean(batch_loss)
            print('Epoch %s, Loss: %s' % (epoch, epoch_loss))
            with open(work_dir + 'loss_log', 'a') as f:
                f.write('%s\n' % (epoch_loss))
            saver.save(sess, work_dir + "models/model_epoch_%d.ckpt" % epoch)

def rewardfunc(_output_sent, _ans_sent):
    sm = SmoothingFunction()
    score_list = [sentence_bleu([a], o,smoothing_function=sm.method1) for a, o in zip(_ans_sent, _output_sent)]

    for i in range(len(_ans_sent)):
        sent = _ans_sent[i]
        if "không biết" not in sent.lower() or "tôi không" not in sent.lower() or "không không" not in sent.lower():
            score_list[i] += 0.1

        if '<unk>' not in sent:
            score_list[i] += 0.1
    return score_list

if __name__ == '__main__':
    main()