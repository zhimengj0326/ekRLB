import _pickle as pickle

import time
import numpy as np
import tensorflow as tf
import math


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def getTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def write_log(log_path, line, echo=False):
	with open(log_path, "a") as log_in:
		log_in.write(line + "\n")
		if echo:
			print(line)


def activate(act_func, x):
	if act_func == 'tanh':
		return tf.tanh(x)
	elif act_func == 'relu':
		return tf.nn.relu(x)
	else:
		return tf.sigmoid(x)


def activate_calc(act_func, x):
	if act_func == "tanh":
		return np.tanh(x)
	elif act_func == "relu":
		return max(0, x)
	else:
		return sigmoid(x)


def init_var_map(init_path, _vars):
	if init_path:
		var_map = pickle.load(open(init_path, "rb"))
	else:
		var_map = {}

	for i in range(len(_vars)):
		key, shape, init_method, init_argv = _vars[i]
		if key not in var_map.keys():
			if init_method == "normal":
				mean, dev, seed = init_argv
				var_map[key] = tf.random_normal(shape, mean, dev, seed=seed)
			elif init_method == "uniform":
				min_val, max_val, seed = init_argv
				var_map[key] = tf.random_uniform(shape, min_val, max_val, seed=seed)
			else:
				var_map[key] = tf.zeros(shape)

	return var_map


def build_optimizer(opt_argv, loss):
	opt_method = opt_argv[0]
	if opt_method == 'adam':
		_learning_rate, _epsilon = opt_argv[1:3]
		opt = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
	elif opt_method == 'ftrl':
		_learning_rate = opt_argv[1]
		opt = tf.train.FtrlOptimizer(learning_rate=_learning_rate).minimize(loss)
	else:
		_learning_rate = opt_argv[1]
		opt = tf.train.GradientDescentOptimizer(learning_rate=_learning_rate).minimize(loss)
	return opt


# obj_type: clk, profit, imp
class Opt_Obj:
	def __init__(self, obj_type="clk", clk_v=500):
		self.obj_type = obj_type
		self.clk_v = clk_v
		if obj_type == "clk":
			self.v1 = 1
			self.v0 = 0
			self.w = 0
		elif obj_type == "profit":
			self.v1 = clk_v
			self.v0 = 1
			self.w = 0
		else:
			self.v1 = 0
			self.v0 = 0
			self.w = 1

	def get_obj(self, imp, clk, cost):
		return self.v1 * clk - self.v0 * cost + self.w * imp


def calc_m_pdf(m_counter, laplace=1):
	m_pdf = [0] * len(m_counter)
	sum = 0
	for i in range(0, len(m_counter)):
		sum += m_counter[i]
	for i in range(0, len(m_counter)):
		m_pdf[i] = (m_counter[i] + laplace) / (
			sum + len(m_counter) * laplace)
	return m_pdf


def str_list2float_list(str_list):
	res = []
	for _str in str_list:
		res.append(float(_str))
	return res


def plot(click_tot, winrate_tot, cpm_tot, ecpc_tot, clk_stat_tot):
	log_in.flush()
	log_in.close()

	filename_click = config.ipinyouPath + 'result/' + 'click_summary.pickle'
	filename_winrate = config.ipinyouPath + 'result/' + 'winrate_summary.pickle'
	filename_cpm = config.ipinyouPath + 'result/' + 'cpm_summary.pickle'
	filename_ecpc = config.ipinyouPath + 'result/' + 'ecpc_summary.pickle'
	i = 0
	for camp in camps:
		filename_clk_stat = config.ipinyouPath + 'result/' + 'clk_stat_camp={}.pickle'.format(camp)
		with open(filename_clk_stat, 'wb') as f5:
			pickle.dump(clk_stat_tot[i, :, :, :], f5)
		i += 1
	with open(filename_click, 'wb') as f1:
		pickle.dump(click_tot, f1)
	with open(filename_winrate, 'wb') as f2:
		pickle.dump(winrate_tot, f2)
	with open(filename_cpm, 'wb') as f3:
		pickle.dump(cpm_tot, f3)
	with open(filename_ecpc, 'wb') as f4:
		pickle.dump(ecpc_tot, f4)

	### plot
	fig1 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, click_tot[:, 0], color="r", linestyle="-", marker="^", linewidth=1, label="Lin")
	plt.plot(c0_vec, click_tot[:, 1], color="b", linestyle="-", marker="*", linewidth=1, label="RLB")
	plt.plot(c0_vec, click_tot[:, 2], color="k", linestyle="-", marker="o", linewidth=1, label="RRLB")
	plt.plot(c0_vec, click_tot[:, 3], color="m", linestyle="-", marker="d", linewidth=1, label="CRTRLB")
	plt.plot(c0_vec, click_tot[:, 4], color="y", linestyle="-", marker="s", linewidth=1, label="CRRLB")
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("Total Clicks")
	plt.legend(loc='lower right')
	plt.savefig(config.ipinyouPath + "result/click.png")
	plt.close(fig1)
	print("click")

	fig2 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, winrate_tot[:, 0], color="r", linestyle="-", marker="^", linewidth=1, label="Lin")
	plt.plot(c0_vec, winrate_tot[:, 1], color="b", linestyle="-", marker="*", linewidth=1, label="RLB")
	plt.plot(c0_vec, winrate_tot[:, 2], color="k", linestyle="-", marker="o", linewidth=1, label="RRLB")
	plt.plot(c0_vec, winrate_tot[:, 3], color="m", linestyle="-", marker="d", linewidth=1, label="CRTRLB")
	plt.plot(c0_vec, winrate_tot[:, 4], color="y", linestyle="-", marker="s", linewidth=1, label="CRRLB")
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("Win Rate")
	plt.legend(loc='lower right')
	plt.savefig(config.ipinyouPath + "result/winrate.png")
	plt.close(fig2)
	print("win rate")

	fig3 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, cpm_tot[:, 0], color="r", linestyle="-", marker="^", linewidth=1, label="Lin")
	plt.plot(c0_vec, cpm_tot[:, 1], color="b", linestyle="-", marker="*", linewidth=1, label="RLB")
	plt.plot(c0_vec, cpm_tot[:, 2], color="k", linestyle="-", marker="o", linewidth=1, label="RRLB")
	plt.plot(c0_vec, cpm_tot[:, 3], color="m", linestyle="-", marker="d", linewidth=1, label="CRTRLB")
	plt.plot(c0_vec, cpm_tot[:, 4], color="y", linestyle="-", marker="s", linewidth=1, label="CRRLB")
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("CPM")
	plt.legend(loc='lower right')
	plt.savefig(config.ipinyouPath + "result/cpm.png")
	plt.close(fig3)
	print("cpm")

	fig4 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, ecpc_tot[:, 0], color="r", linestyle="-", marker="^", linewidth=1, label="Lin")
	plt.plot(c0_vec, ecpc_tot[:, 1], color="b", linestyle="-", marker="*", linewidth=1, label="RLB")
	plt.plot(c0_vec, ecpc_tot[:, 2], color="k", linestyle="-", marker="o", linewidth=1, label="RRLB")
	plt.plot(c0_vec, ecpc_tot[:, 3], color="m", linestyle="-", marker="d", linewidth=1, label="CRTRLB")
	plt.plot(c0_vec, ecpc_tot[:, 4], color="y", linestyle="-", marker="s", linewidth=1, label="CRRLB")
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("eCPC")
	plt.legend(loc='lower right')
	plt.savefig(config.ipinyouPath + "result/ecpc.png")
	plt.close(fig4)
	print("ecpc")

	np.zeros([len(camps), np.ceil(max_market_price / clk_stat_interval).astype(int), len(c0_vec), Algo_num])
	bid_vec = range(clk_stat_interval, max_market_price + 1, clk_stat_interval)
	i = -1
	for camp in camps:
		i += 1
		for index in range(len(c0_vec)):
			c0 = c0_vec[index]
			fig0 = plt.figure(figsize=(6.4, 4.8))
			plt.plot(bid_vec, clk_stat_tot[i, :, index, 0], color="r", linestyle="-", marker="^", linewidth=1,
					 label="Lin")
			plt.plot(bid_vec, clk_stat_tot[i, :, index, 1], color="b", linestyle="-", marker="*", linewidth=1,
					 label="RLB")
			plt.plot(bid_vec, clk_stat_tot[i, :, index, 2], color="k", linestyle="-", marker="o", linewidth=1,
					 label="RRLB")
			plt.plot(bid_vec, clk_stat_tot[i, :, index, 3], color="m", linestyle="-", marker="d", linewidth=1,
					 label="CRTRLB")
			plt.plot(bid_vec, clk_stat_tot[i, :, index, 4], color="y", linestyle="-", marker="s", linewidth=1,
					 label="CRRLB")
			plt.grid(color="k", linestyle=":")
			plt.xlabel("Bidding price")
			plt.ylabel("Click number")
			plt.title("camp={} and c0={}".format(camp, c0))
			plt.legend(loc='upper left')
			plt.savefig(config.ipinyouPath + "result/clk_stat_camp={}_c0={}.png".format(camp, c0))
			plt.close(fig0)

	for index in range(len(c0_vec)):
		c0 = c0_vec[index]
		fig0 = plt.figure(figsize=(6.4, 4.8))
		plt.plot(bid_vec[:-1], np.mean(clk_stat_tot[:, :-1, index, 0], 0), color="r", linestyle="-", marker="^",
				 linewidth=1, label="Lin")
		plt.plot(bid_vec[:-1], np.mean(clk_stat_tot[:, :-1, index, 1], 0), color="b", linestyle="-", marker="*",
				 linewidth=1, label="RLB")
		plt.plot(bid_vec[:-1], np.mean(clk_stat_tot[:, :-1, index, 2], 0), color="k", linestyle="-", marker="o",
				 linewidth=1, label="RRLB")
		plt.plot(bid_vec[:-1], np.mean(clk_stat_tot[:, :-1, index, 3], 0), color="m", linestyle="-", marker="d",
				 linewidth=1,
				 label="CRTRLB")
		plt.plot(bid_vec[:-1], np.mean(clk_stat_tot[:, :-1, index, 4], 0), color="y", linestyle="-", marker="s",
				 linewidth=1,
				 label="CRRLB")
		plt.grid(color="k", linestyle=":")
		plt.xlabel("Bidding price")
		plt.ylabel("Click number")
		plt.title("$c_0$={}".format(c0))
		plt.legend(loc='upper right')
		left, bottom, width, height = 0.5, 0.6, 0.2, 0.2
		ax2 = fig0.add_axes([left, bottom, width, height])
		ax2.plot(bid_vec[-1], np.mean(clk_stat_tot[:, -1, index, 0], 0), color="r", linestyle="-", marker="^",
				 linewidth=1, label="Lin")
		ax2.plot(bid_vec[-1], np.mean(clk_stat_tot[:, -1:, index, 1], 0), color="b", linestyle="-", marker="*",
				 linewidth=1,
				 label="RLB")
		ax2.plot(bid_vec[-1], np.mean(clk_stat_tot[:, -1:, index, 2], 0), color="k", linestyle="-", marker="o",
				 linewidth=1,
				 label="RRLB")
		ax2.plot(bid_vec[-1], np.mean(clk_stat_tot[:, -1:, index, 3], 0), color="m", linestyle="-", marker="d",
				 linewidth=1,
				 label="CRTRLB")
		ax2.plot(bid_vec[-1], np.mean(clk_stat_tot[:, -1:, index, 4], 0), color="y", linestyle="-", marker="s",
				 linewidth=1,
				 label="CRRLB")
		# ax2.set_xlabel("Bidding price")
		# ax2.set_ylabel("Click number")

		plt.savefig(config.ipinyouPath + "result/clk_stat_c0={}.png".format(c0))
		plt.close(fig0)
