
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

src = "ipinyou"

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

def run(camp, src, buffer_time, train_steps, batch_size, episodes_n, episodes_num, state_size, action_size, learning_rate, hidden_num, hidden_unit, c0, max_bid, clk_stat_interval):
	train_file, test_file = config.get_camp_info(camp, src, ifmix=True)
	avg_m_price = np.mean(train_file['data'][:, 1])
	m_pdf = calc_m_pdf(train_file['mprice_count'])
	m_pdf = np.array(m_pdf)[:, 0]
	graph = tf.compat.v1.get_default_graph()
	init_budget = int(float(train_file['budget']) / float(train_file['imp']) * c0 * episodes_n)
	norm_state = np.array([min(episodes_n, train_file['imp']), init_budget]).T

	with tf.compat.v1.Session() as sess:
		with graph.as_default():
			mlp_network = MLP_model(state_size, action_size, learning_rate, hidden_num, hidden_unit, norm_state)
		sess.run(tf.compat.v1.global_variables_initializer())
		rtb_environment = RTB_environment(train_file, episodes_n, init_budget, camp, learning_rate,
										  norm_state, sess)
		rtb_environment.load_data(train_file)
		print("load data done")
		log_in = open(data_path + "/bid-performance/camp={}_ss_N={}_c0={}.txt".format(camp, episodes_n, c0), "w")


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
		clk_trace = []
		risk_tendency_ini = rt_initializer(train_file, 1, avg_m_price, episodes_n, init_budget)
		print("risk tendency initilization")
		risk_tendency_table = risk_tendency_ini.calc_risk_tendency(episodes_n, init_budget, max_market_price, m_pdf, 0, ifrisk=True,
											 ifconst_risktendency=False)
		with open("results/risk_tendency_ori.pickle", 'wb') as f:
			pickle.dump(risk_tendency_table, f)
		path = "results/camp={}_c0={}_risk_tendency_ori.png".format(camp, c0)
		plot_risk_tendency(risk_tendency_table, path)
		print("print risk tendency ori figure")

		# train_writer = tf.compat.v1.summary.FileWriter("results/train")
		# test_writer = tf.compat.v1.summary.FileWriter("results/test")

		# #Train initial MLP
		train_ini_steps = 1000
		trace_solu = 10
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
			print("train network")

			global_step_counter += 1

		# hidden_layer, output_layer = sess.run([mlp_network.hidden_layer, mlp_network.output_layer], feed_dict={mlp_network.input_pl: obs})
		# print("hidden_layer={}, output_layer={}".format(hidden_layer, output_layer))

		risk_tendency_table = risktendency_get(sess, mlp_network, episodes_n, init_budget)
		with open("results/risk_tendency_ini.pickle", 'wb') as f:
			pickle.dump(risk_tendency_table, f)
		path = "results/camp={}_c0={}_risk_tendency_ini_learn.png".format(camp, c0)
		plot_risk_tendency(risk_tendency_table, path)
		print("print risk tendency learn figure")

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
				global_step_counter += 1

			#risk_tendency_table = risktendency_get(sess,mlp_network, episodes_n, init_budget)
			print("camp={} c0={} and Training: Episode={}".format(camp, c0, episode_counter))
			print("running training time: {}".format(time.time() - start_time))

		risk_tendency_table = risktendency_get(sess, mlp_network, episodes_n, init_budget)
	with open("results/risk_tendency_learn.pickle", 'wb') as f:
		pickle.dump(risk_tendency_table, f)
	path = "results/camp={}_c0={}_risk_tendency_learn.png".format(camp, c0)
	plot_risk_tendency(risk_tendency_table, path)
	print("print risk tendency learn2 figure")
		## Test
		# DQN





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
	episodes_num = 30  # num_episodes
	max_bid = 300
	buffer_time = 5
	train_steps = 5
	batch_size = int(1e2)
	hidden_num = 2
	hidden_unit = 100

	state_size = 2
	action_size = 1
	learning_rate = 0.001

	# ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
	ipinyou_camps = "1458"
	c0 = 1 / 32
	# c0_vec = [1 / 32]
	camps = ipinyou_camps

	log = "{:<55}\t {:>10}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}" \
		.format("setting", "auctions", "impressions", "click", "cost", "win-rate", "eCPC", "CPM")
	print(log)
	# summary result
	# click_tot = np.zeros([len(camps), len(c0_vec)])  #
	# winrate_tot = np.zeros([len(camps), len(c0_vec)])
	# cpm_tot = np.zeros([len(camps), len(c0_vec)])
	# ecpc_tot = np.zeros([len(camps), len(c0_vec),])
	# clk_stat_tot = np.zeros([len(camps), np.ceil(max_market_price / clk_stat_interval).astype(int), len(c0_vec)])


	run(ipinyou_camps, src, buffer_time, train_steps, batch_size, episodes_n, episodes_num, state_size, action_size,
		learning_rate, hidden_num, hidden_unit, c0, max_bid, clk_stat_interval)




if __name__ == "__main__":
	main()

