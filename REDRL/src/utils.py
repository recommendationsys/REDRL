import pickle
import numpy as np
import tensorflow as tf

RATINGFILE = dict({'MI': 'ratings.txt'})
LR = dict({'MI': 1e-2})
L2 = dict({'MI': 1e-2})
EMB = dict({'MI': 10})
BOUND = dict({'MI': 4.0})


def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f)


def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)


def get_envobjects(ratingfile='MI', max_step=5000, train_rate=0.95,
                   max_stop_count=30):

    rating_file_path = '../data/processed_data/'+ratingfile + '/'+RATINGFILE[ratingfile]
    rating = np.loadtxt(fname=rating_file_path, delimiter='\t')
    lr = LR[ratingfile]
    l2_factor = L2[ratingfile]
    emb_size = EMB[ratingfile]
    boundary_rating = BOUND[ratingfile]

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = len(item_set)
    item_list = list(item_set)

    data = np.array(rating)
    np.random.shuffle(data)

    t = int(len(data) * train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1) + tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                     feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse,
                             feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
            0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                         feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                    ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1],
                                                  ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
                i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test > pre_rmse_test:
                stop_count += 1
                if stop_count == max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        user_embeddings_value, item_embeddings_value, ibias_bias_value = sess.run([user_embeddings, item_embeddings, item_bias])
        mf_rating = np.dot(user_embeddings_value, item_embeddings_value.T) + ibias_bias_value.T
        boundary_item_id = int(item_num * 0.8)
        train_rating = mf_rating[:, :boundary_item_id]
        test_rating = mf_rating[:, boundary_item_id:]
        rela_num = np.sum(np.where(test_rating > boundary_rating, 1, 0), axis=1)

        print('done with full stop count' if stop_count_flag else 'done with full training step')
        pickle_save(train_rating, '../data/processed_data/'+ratingfile+'/train_rating')
        pickle_save({'train_matrix': train_rating, 'user_num': user_num, 'item_num': item_num, 'rela_num': rela_num, 'item_list': item_list, 'r_matrix': mf_rating},
                    '../data/run_time/%s_env_objects' % ratingfile)



def popular_in_train_user(ratingfile='movielens', topk=300, boundary = 3.0):
    rating_file_path = '../data/processed_data/'+ratingfile + '/'+RATINGFILE[ratingfile]
    rating = np.loadtxt(fname=rating_file_path, delimiter='\t')

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = len(item_set)
    r_matrix = np.zeros(shape=[user_num,item_num])

    for i, j, k in rating:
        if int(k)>boundary:
            r_matrix[int(i),int(j)] = 1.0
        elif int(k)<boundary:
            r_matrix[int(i),int(j)] = -1.0

    boundary_item_id = int(item_num * 0.8)
    ave_rating = np.mean(r_matrix[:, :boundary_item_id], axis=0)
    topk_item = np.argsort(ave_rating)[-topk:]

    pickle_save(topk_item, '../data/run_time/%s' % ratingfile + '_pop%d' % topk)