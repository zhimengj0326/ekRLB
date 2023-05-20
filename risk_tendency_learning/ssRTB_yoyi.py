# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import itertools
import time
import pickle
from rtb_environment import RTB_environment
from rtb_environment import get_data
import config
import argparse
from utility import calc_m_pdf
from collections import deque
from joblib import Parallel, delayed
from risk_tendency_initializer import rt_initializer
from buffer import SortedBuffer
from deep_q_network import MLP_model
from plotfigure import plot_risk_tendency, plot_click

src = "yoyi"

if src == "ipinyou":
	camps = config.ipinyou_camps
	data_path = config.ipinyouPath
	max_market_price = config.ipinyou_max_market_price
elif src == "vlion":
	camps = config.vlion_camps
	data_path = config.vlionPath
	max_market_price = config.vlion_max_market_price
elif src == "yoyi":
	camps = config.yoyi_camps
	data_path = config.yoyiPath
	max_market_price = config.yoyi_max_market_price

#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# tf.compat.v1.random.set_seed(0)
#np.random.seed(0)
## Test
# np.random.seed(123)
# tf.random.set_seed(0)

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Reinforcement Learning via Imitation")
    parser.add_argument('--env_id', help='environment ID', default='InvertedPendulum-v2')
    parser.add_argument('--epsilon_max', help='epsilon_max', type=float, default=0.95)
    parser.add_argument('--epsilon_min', help='epsilon_min', default=0.05)
    parser.add_argument('--epsilon_decay_rate', help='epsilon_decay_rate', type=float, default=0.000025)
    # Hyperparameters
    parser.add_argument('--discount_factor', help='discount_factor', type=float, default=1)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=32)
    parser.add_argument('--buffer_size', help='the size of the sorted buffer', type=int, default=1e5)
    parser.add_argument('--update_frequency', help='update_frequency', type=int, default=100)
    parser.add_argument('--hid_size', help='the hidden size of the network', default=64)
    parser.add_argument('--num_hid_layers', help='the number of the hiddenlayers of the network', default=2)
    parser.add_argument('--learning_rate', help='the learning rate', type=float, default=1e-3)
    # parser.add_argument('--rollout_steps', help='the number of rollouts in each iteration', type=int,  default=1)
    # parser.add_argument('--train_steps', help='the number of training updates in each iteration', type=int, default=5)
    # parser.add_argument('--log_every', help='log every timestep', default=2000)
    # parser.add_argument('--eval_num', help='the number of evaluation number', default=0)
    parser.add_argument('--noise_type', help='the type of the noise', default='none')

    return parser

def run(camp, src, buffer_time, train_steps, batch_size, episodes_n, episodes_num, state_size, action_size, learning_rate, hidden_num, hidden_unit, c0, max_bid, clk_stat_interval):
    train_file, test_file = config.get_camp_info(camp, src, ifmix=True)
    avg_m_price = np.mean(train_file['data'][:, 1])
    m_pdf = calc_m_pdf(train_file['mprice_count'])
    m_pdf = np.array(m_pdf)[:, 0]
    graph = tf.compat.v1.get_default_graph()
    init_budget = int(train_file['budget'] / train_file['imp'] * c0 * episodes_n)
    norm_state = np.array([min(episodes_n, train_file['imp']), init_budget]).T

    with tf.compat.v1.Session() as sess:
        with graph.as_default():
            mlp_network = MLP_model(state_size, action_size, learning_rate, hidden_num, hidden_unit, norm_state)
        sess.run(tf.compat.v1.global_variables_initializer())
        rtb_environment = RTB_environment(train_file, episodes_n, init_budget, camp, learning_rate,
                                          norm_state, sess)
        rtb_environment.load_data(train_file)
        log_in = open(data_path + "bid-performance/camp={}_ss_N={}_c0={}.txt".format(camp, episodes_n, c0), "w")


        rtb_environment.reset(init_budget)

        # Replay_memory = replay_memory(memory_cap, batch_size)
        # Buffer
        buffer_size = 1e5
        buf = SortedBuffer(size=int(buffer_size),
                           ob_dim=2,
                           ac_dim=1)

        ret_buf = deque(maxlen=buffer_time*100)
        episode_counter = 0
        global_step_counter = 0
        risk_tendency_ini = rt_initializer(train_file, 1, avg_m_price, episodes_n, init_budget)
        risk_tendency_table = risk_tendency_ini.calc_risk_tendency(episodes_n, init_budget, max_market_price, m_pdf, 0, ifrisk=True,
                                             ifconst_risktendency=False)
        path = data_path + "result/camp={}_c0={}_risk_tendency_ori.png".format(camp, c0)
        plot_risk_tendency(risk_tendency_table, path)

        # train_writer = tf.compat.v1.summary.FileWriter("results/train")
        # test_writer = tf.compat.v1.summary.FileWriter("results/test")

        # #Train initial MLP
        train_ini_steps = 1000
        for i in range(train_ini_steps):
            sample_batch = np.random.randint((1+episodes_n)*(1+init_budget), size=batch_size)
            time_step = sample_batch // (1+init_budget)
            remain_budget = sample_batch % (1+init_budget)
            obs = (np.array([time_step, remain_budget]).T).astype(int)
            ac = []
            for i in range(len(time_step)):
                ac_sample = risk_tendency_table[time_step[i], remain_budget[i]]
                ac.append(ac_sample)
            ac = np.reshape(np.array(ac), (len(ac), -1))
            mlp_network.train_batch(sess, obs, ac)

        # hidden_layer, output_layer = sess.run([mlp_network.hidden_layer, mlp_network.output_layer], feed_dict={mlp_network.input_pl: obs})
        # print("hidden_layer={}, output_layer={}".format(hidden_layer, output_layer))

        risk_tendency_table = risktendency_get(sess, mlp_network, episodes_n, init_budget)

        path = data_path + "result/camp={}_c0={}_risk_tendency_ini_learn.png".format(camp, c0)
        plot_risk_tendency(risk_tendency_table, path)

        while (episode_counter < episodes_num):
            # update value function
            start_time = time.time()
            if (episode_counter % 10 == 0):
                rtb_environment.calc_optimal_value_function_with_approximation_i(episodes_n, init_budget, max_bid, m_pdf, risk_tendency_table)

            for i in range(buffer_time):
                rtb_environment.reset(init_budget)
                state, risk_tendency, ret_list = rtb_environment.episode(sess, mlp_network, max_bid,
                                                                         risk_tendency_table, ifintial={episode_counter==0})
                ret_buf.append(ret_list)
                episode_counter += 1

                for obs, rt, ret in zip(state, risk_tendency, itertools.repeat(ret_list)):
                    # print(obs.shape, acs.shape, ret)
                    buf.insert(obs, rt, ret)

            if (episode_counter % 10 == 0):
                print('Episode {} of {}'.format(episode_counter, episodes_num))

            #rtb_environment.reset(init_budget)
            for _ in range(train_steps):
                obs, acs = buf.sample(batch_size, k=buffer_size)
                mlp_network.train_batch(sess, obs, acs)

            #risk_tendency_table = risktendency_get(sess,mlp_network, episodes_n, init_budget)
            print("camp={} c0={} and Training: Episode={}".format(camp, c0, episode_counter))
            print("running training time: {}".format(time.time() - start_time))


        ## Test
        # DQN
        risk_tendency_table2 = risktendency_get(sess, mlp_network, episodes_n, init_budget)
        #risk_tendency_table_zero = np.zeros([episodes_n + 1, init_budget + 1])

        # auction, imp, clk, cost, clk_stat = test_camp(test_file, rtb_environment, risk_tendency_table, episodes_n,\
        #                                             init_budget, camp, c0, max_bid, m_pdf, clk_stat_interval)
        # auction, imp, clk, cost, clk_stat = test_camp(test_file, rtb_environment, risk_tendency_table_zero, episodes_n,\
        #                                               init_budget, camp, c0, max_bid, m_pdf, clk_stat_interval)
        auction, imp, clk, cost, clk_stat = test_camp(test_file, rtb_environment, risk_tendency_table2, episodes_n, \
                                                      init_budget, camp, c0, max_bid, m_pdf, clk_stat_interval)
    sess.close()
    return auction, imp, clk, cost, clk_stat

def test_camp(test_file, rtb_environment, risk_tendency_table2, episodes_n, init_budget, camp, c0, max_bid, m_pdf, clk_stat_interval):

    path = data_path + "result/camp={}_c0={}_risk_tendency_learn.png".format(camp, c0)
    plot_risk_tendency(risk_tendency_table2, path)

    # risk_tendency_table2 = np.zeros([episodes_n+1, init_budget+1])

    rtb_environment.calc_optimal_value_function_with_approximation_i(episodes_n, init_budget, max_bid, m_pdf,
                                                                     risk_tendency_table2)
    start_time = time.time()
    test_budget = init_budget
    print("test_budget:{}".format(test_budget))
    src = "yoyi"

    setting = "{}, camp={}, algo={}, N={}, c0={}" \
        .format(src, camp, "RRTB", episodes_n, c0)

    (auction, imp, clk, cost, clk_stat) = rtb_environment.run(test_file, risk_tendency_table2, episodes_n, init_budget,
                                                              max_bid, clk_stat_interval, ifconst_risk=False)

    # ecpc = (cost / 1000) / clk_train

    print("c0= {} and click number:{}".format(c0, clk))
    print("test time={}".format(time.time() - start_time))

    return auction, imp, clk, cost, clk_stat


def risktendency_get(sess, mlp_network, N, B):
    risk_tendency = np.zeros([N+1, B+1])
    for i in range(N+1):
        states = np.array([[i, j] for j in range(B+1)])
        risk_inter = mlp_network.predict_batch(sess, states)
        for j in range(B+1):
            risk_tendency[i, j] = risk_inter[j]
    return risk_tendency

def main():
    start_time = time.time()
    os.system("taskset -p -c 0-46 %d" % os.getpid())
    clk_stat_interval = 30
    update_frequency = 100
    episodes_n = 1000
    episodes_num = 10  # num_episodes
    max_bid = 300
    buffer_time = 5
    train_steps = 5
    batch_size = int(1e2)
    hidden_num = 2
    hidden_unit = 100


    state_size = 2
    action_size = 1
    learning_rate = 0.001

    #ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
    yoyi_camps = ["1458"]
    #c0_vec = [1 / 32, 1 / 16, 1/8, 1/4]
    c0_vec = [1/2]
    camps = yoyi_camps

    log = "{:<55}\t {:>10}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}" \
        .format("setting", "auctions", "impressions", "click", "cost", "win-rate", "eCPC", "CPM")
    print(log)
    #summary result
    # click_tot = np.zeros([len(camps), len(c0_vec)])  #
    # winrate_tot = np.zeros([len(camps), len(c0_vec)])
    # cpm_tot = np.zeros([len(camps), len(c0_vec)])
    # ecpc_tot = np.zeros([len(camps), len(c0_vec),])
    # clk_stat_tot = np.zeros([len(camps), np.ceil(max_market_price / clk_stat_interval).astype(int), len(c0_vec)])

    processNum = 5
    #Parallel(processNum, 'multiprocessing', max_nbytes=None)(delayed(run)(camps[camp_index]) for camp_index in range(len(camps)))

    auction_tot, imp_tot, click_tot, cost_tot, clk_stat_tot = zip(*Parallel(processNum, 'multiprocessing', max_nbytes=None)(
		delayed(run)(camps, src, buffer_time, train_steps, batch_size, episodes_n, episodes_num, state_size, action_size, learning_rate, hidden_num, hidden_unit, c0_vec[c0_index], max_bid, clk_stat_interval)
        for c0_index in range(len(c0_vec))))

    click_tot = np.array(click_tot).reshape((len(c0_vec)))
    auction_tot = np.array(auction_tot).reshape((len(c0_vec)))
    imp_tot = np.array(imp_tot).reshape((len(c0_vec)))
    cost_tot = np.array(cost_tot).reshape((len(c0_vec)))
    #clk_stat_tot = np.array(clk_stat_tot).reshape((len(c0_vec), len(camps)))
    #print("clicl_tot shape={}".format(click_tot.shape()))

    winrate_tot = imp_tot / auction_tot * 100
    cpm_tot = (cost_tot / 1000) / imp_tot * 1000
    #ecpc_tot = (cost_tot / 1000) / np.max(click_tot, 1)

    results_file = data_path + "result/ssRTB_results_1/2.pickle"
    print("click_tot={}".format(click_tot))
    with open(results_file, 'wb') as f:
        ssRTB_results = {'click': click_tot, 'win_rate': winrate_tot, 'cpm': cpm_tot}
        pickle.dump(ssRTB_results, f)

    winrate_avg = np.sum(imp_tot, 0) / np.sum(auction_tot, 0) * 100
    cpm_avg = (np.sum(cost_tot, 0) / 1000) / np.sum(imp_tot, 0) * 1000
    ecpc_avg = (np.sum(cost_tot, 0) / 1000) / np.sum(click_tot, 0)
    click_avg = np.mean(click_tot, 0)

    print("c0= {} and click number:{}".format(c0_vec, click_avg))

    results_file = data_path + "result/ssRTB_avg_results.pickle"
    with open(results_file, 'wb') as f:
        ssRTB_avg_results = {'click': click_avg, 'win_rate': winrate_avg, 'cpm': cpm_avg, 'ecpc': ecpc_avg}
        pickle.dump(ssRTB_avg_results, f)

    # path = "results/"
    # plot_click(click_avg, winrate_avg, cpm_avg, ecpc_avg, c0_vec, path)

    print("running time={}".format(time.time() - start_time))

    # log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
    #     .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
    # print(log)


if __name__=="__main__":
    main()
