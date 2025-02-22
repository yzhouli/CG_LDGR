import os

import numpy as np

import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras import layers, optimizers, datasets, models
from sklearn.metrics import f1_score as F1_com
from keras import backend as K
from tqdm import tqdm


def load_train_data(path):
    att_matrix = np.load(file=f'{path}/att_matrix.npy')
    att_matrix = tf.cast([att_matrix], dtype=tf.float32)

    rel_matrix = np.load(file=f'{path}/rel_matrix.npy')
    rel_depth = max(rel_matrix.max(-1)) + 1
    rel_matrix = tf.cast([rel_matrix], dtype=tf.int32)
    depth_matrix = tf.cast([rel_depth], dtype=tf.int32)

    label_matrix = np.load(file=f'{path}/label_matrix.npy')
    total = (label_matrix >= 0).sum()

    label_matrix = tf.cast(label_matrix, dtype=tf.int32)

    return att_matrix, rel_matrix, depth_matrix, label_matrix, total


def mormal_acc(label_matrix, out_matrix):
    out_matrix = out_matrix.numpy()
    pred_li, label_li = [], []
    for i in range(len(label_matrix)):
        if label_matrix[i] == -1:
            continue
        pred_li.append(out_matrix[i])
        label_li.append(label_matrix[i])
    pred_li = tf.cast(pred_li, dtype=tf.float32)
    label_li = tf.cast(label_li, dtype=tf.int32)
    label_li = tf.one_hot(label_li, depth=3)
    return pred_li, label_li


def max_index(pred_li):
    index = 0
    temp = -1000
    for i, num in enumerate(pred_li):
        if num > temp:
            temp = num
            index = i
    return index


def acc(label_matrix, out_matrix):
    out_matrix = out_matrix.numpy()
    true_total = 0
    pred_li = []
    for i in range(len(label_matrix)):
        if label_matrix[i] == -1:
            continue
        pred = tf.nn.softmax(out_matrix[i])
        pred_index = max_index(pred_li=pred)
        pred_li.append(pred_index)
        if pred_index == label_matrix[i]:
            true_total += 1
    return true_total, pred_li


def normal(predict_li):
    result_li = []
    for att_li in predict_li.numpy():
        index = max_index(att_li)
        result_li.append(index)
    result_li = np.asarray(result_li, dtype=np.int32)
    result_li = tf.cast(result_li, dtype=tf.int32)
    result_li = tf.one_hot(result_li, depth=3)
    return result_li


def evaluation(y_test, y_predict):
    y_predict = normal(predict_li=y_predict)
    metrics = classification_report(y_test, y_predict, output_dict=True)
    precision = metrics['0']['precision'], metrics['1']['precision'], metrics['2']['precision']
    recall = metrics['0']['recall'], metrics['1']['recall'], metrics['2']['recall']
    f1_score = metrics['0']['f1-score'], metrics['1']['f1-score'], metrics['2']['f1-score']
    return precision, recall, f1_score


def expend(data_li, size):
    result_li = []
    for item in tqdm(data_li, desc='auc'):
        for i in range(size):
            if i == item:
                result_li.append(1)
            else:
                result_li.append(0)
    return result_li


def com_auc(pred, label):
    pred_li = expend(data_li=pred, size=3)
    label_li = expend(data_li=label, size=3)
    auc = roc_auc_score(y_true=label_li, y_score=pred_li)
    return auc


def train(path, model, learning_rate, save_path, epochs, train_rate):
    optimizer = optimizers.Adam(learning_rate)
    line_li = []
    max_acc = -1
    temp_path = ''

    task_li = [i for i in os.listdir(path) if not i.__contains__('.D')]
    for i in range(epochs):
        true, finish_total, loss_total = 0, 0, 0
        true_test, finish_total_test, loss_total_test = 0, 0, 0
        label_matrix_all, pred_matrix_all = None, None
        true_li, pred_li = [], []
        for index, train_name in enumerate(tqdm(task_li, desc=f'train epoch: {i + 1}')):
            train_id = int(str(train_name).split('step')[-1])
            if train_id >= len(task_li) - 1:
                continue
            att_matrix, rel_matrix, depth_matrix, label_matrix, total = load_train_data(path=f'{path}/{train_name}')

            with tf.GradientTape() as tape:
                out_matrix = model((att_matrix, rel_matrix, depth_matrix, total))

                true_total, temp_li = acc(out_matrix=out_matrix, label_matrix=label_matrix)

                label_li = label_matrix

                label_matrix = tf.one_hot(label_matrix, depth=3)

                loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                loss = tf.reduce_mean(loss)

                if train_id < len(task_li) * train_rate:
                    true += true_total
                    finish_total += total
                    loss_total += loss
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                else:
                    [true_li.append(int(i)) for i in label_li]
                    [pred_li.append(int(i)) for i in temp_li]
                    true_test += true_total
                    finish_total_test += total
                    loss_total_test += loss
                    if label_matrix_all is None:
                        label_matrix_all = label_matrix
                        pred_matrix_all = out_matrix
                    else:
                        label_matrix_all = tf.concat([label_matrix_all, label_matrix], axis=0)
                        pred_matrix_all = tf.concat([pred_matrix_all, out_matrix], axis=0)

        macro_f1 = F1_com(true_li, pred_li, average='macro')
        micro_f1 = F1_com(true_li, pred_li, average='micro')
        auc = com_auc(pred=pred_li, label=true_li)
        print('auc:', auc)
        print('macro_f1:', macro_f1, 'micro_f1:', micro_f1)
        precision, recall, f1_score = evaluation(y_test=label_matrix_all, y_predict=pred_matrix_all)
        train_loss, train_acc = loss_total / finish_total, true / finish_total
        test_loss, test_acc = loss_total_test / finish_total_test, true_test / finish_total_test
        if test_acc > max_acc:
            max_acc = test_acc

        print(f'train accuracy: {train_acc}, train loss: {train_loss}')
        print(f'test accuracy: {test_acc}, test loss: {test_loss}')
        print(f'non-precision: {precision[0]}, non-recall: {recall[0]}, non-f1_score: {f1_score[0]}')
        print(f'rumor-precision: {precision[1]}, rumor-recall: {recall[1]}, rumor-f1_score: {f1_score[1]}')
        print(f'promote-precision: {precision[2]}, promote-recall: {recall[2]}, promote-f1_score: {f1_score[2]}\n')
        att_line = f'{macro_f1}\t{micro_f1}\t{train_loss}\t{train_acc}\t{test_loss}\t{test_acc}\t{precision[0]}\t{precision[1]}\t{precision[2]}'
        att_line += f'\t{recall[0]}\t{recall[1]}\t{recall[2]}\t{f1_score[0]}\t{f1_score[1]}\t{f1_score[2]}\n'
        line_li.append(att_line)
    metric_path = f'{save_path}/{max_acc}'
    if not os.path.exists(metric_path):
        os.mkdir(metric_path)
    model.load_weights(temp_path)
    model.save_weights(f'{metric_path}/model')
    print('max-acc:', max_acc)
    with open(f'{metric_path}/metric.txt', 'w+') as f:
        for line in line_li:
            f.write(line)
