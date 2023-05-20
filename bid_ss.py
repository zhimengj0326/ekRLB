#!/usr/bin/python
import config
import matplotlib.pyplot as plt
from ss_mdp import SS_MDP
#from mcpc import Mcpc
from lin_bid import Lin_Bid
from rlb_dp_i import RLB_DP_I
from utility import *
import numpy as np
from joblib import Parallel, delayed
import sys
import os

obj_type = "clk"
clk_vp = 1
N = 1000
c0_vec = [1 / 16]
#c0_vec = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1/2]
Algo_num = 5
clk_stat_interval = 10
#c0_vec = [1/16]
gamma = 1

src = "ipinyou"

log_in = open(config.projectPath + "bid-performance/{}_N={}_obj={}_clkvp={}.txt".format(src, N, obj_type, clk_vp),
			  "w")
print("logs in {}".format(log_in.name))
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}" \
	.format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
print(log)
log_in.write(log + "\n")

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

def run(camp):
	# summary result
	click_tot = np.zeros([len(c0_vec), Algo_num])  #
	winrate_tot = np.zeros([len(c0_vec), Algo_num])
	cpm_tot = np.zeros([len(c0_vec), Algo_num])
	ecpc_tot = np.zeros([len(c0_vec), Algo_num])

	profit_tot = np.zeros([len(c0_vec), Algo_num])
	ROI_tot = np.zeros([len(c0_vec), Algo_num])


	camp_info, auction_in, eCPC = config.get_camp_info(camp, src, ifmix=True)
	#auction_in = open(data_path + camp + "/test.theta.txt", "r")
	opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info['budget'] / camp_info['click']))
	avg_m_price = np.mean(camp_info['data'][:, 1])
	m_pdf = calc_m_pdf(camp_info['mprice_count'])
	m_pdf = np.array(m_pdf)[:, 0]
	const_risk_tendency = -0.1
	clk_stat = np.zeros([np.ceil(max_market_price / clk_stat_interval).astype(int), len(c0_vec), Algo_num])
	for c0_index in range(len(c0_vec)):
		c0 = c0_vec[c0_index]
		B = int(camp_info['budget'] / camp_info['imp'] * c0 * N)
		print(f'B={B} c0={c0}')
		rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma, avg_m_price, auction_in['data'], N, B)
		# if c0 == 1/2:
		# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
		# 			data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=False) # RRLB
		# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
		# 			data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=False, ifconst_risktendency=False) #RLB
		# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
		# 			data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=True) # CRTRLB

		# # Mcpc
		# auction_in = open(data_path + camp + "/test.theta.txt", "r")
		# mcpc = Mcpc(camp_info)
		# setting = "{}, camp={}, algo={}, N={}, c0={}"\
		# 	.format(src, camp, "mcpc", N, c0)
		# bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
		# (auction, imp, clk, cost) = mcpc.run(auction_in, bid_log_path, N, c0,
		#                                      max_market_price, delimiter=" ", save_log=False)
		# win_rate = imp / auction * 100
		# cpm = (cost / 1000) / imp * 1000
		# ecpc = (cost / 1000) / clk
		# obj = opt_obj.get_obj(imp, clk, cost)
		# log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		# 	.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		# print(log)
		# log_in.write(log + "\n")
		#
		# Lin-Bid
		lin_bid = Lin_Bid(camp_info)
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "lin_bid", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
		model_path = data_path + "bid_log/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
		#valid_path = config.ipinyouPath + 'iPinYou/' + 'train_' + camp + 'n_MR_meanstd.pickle'
		lin_bid.parameter_tune(opt_obj, camp_info['data'], model_path, N, c0, max_market_price, max_market_price, clk_stat_interval, load=True)
		(auction, imp, clk, return_ctr, cost, clk_stat[:, c0_index, 0]) = lin_bid.run(auction_in['data'], bid_log_path, N, c0, max_market_price, clk_stat_interval, save_log=True)


		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

		click_tot[c0_index, 0] = clk
		winrate_tot[c0_index, 0] = win_rate
		cpm_tot[c0_index, 0] = cpm
		ecpc_tot[c0_index, 0] = ecpc
		profit_tot[c0_index, 0] = return_ctr * eCPC - cost
		ROI_tot[c0_index, 0] = (return_ctr * eCPC - cost) / cost

		# RLB
		#rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "RLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)
		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, False, ifconst_risktendency=False)
		(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, c0_index, 1]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, c0,
												 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=False)#False

		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

		click_tot[c0_index, 1] = clk
		winrate_tot[c0_index, 1] = win_rate
		cpm_tot[c0_index, 1] = cpm
		ecpc_tot[c0_index, 1] = ecpc
		profit_tot[c0_index, 1] = return_ctr * eCPC - cost
		ROI_tot[c0_index, 1] = (return_ctr * eCPC - cost) / cost

		# RLB with risk
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "RRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)
		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)

		(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, c0_index, 2]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, c0,
												 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=False)  # False

		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

		click_tot[c0_index, 2] = clk

		winrate_tot[c0_index, 2] = win_rate
		cpm_tot[c0_index, 2] = cpm
		ecpc_tot[c0_index, 2] = ecpc
		profit_tot[c0_index, 2] = return_ctr * eCPC - cost
		ROI_tot[c0_index, 2] = (return_ctr * eCPC - cost) / cost

		# RLB with constant risk
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "CRRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
		#rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)

		(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, c0_index, 3]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, c0,
												 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=True)  # False

		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

		click_tot[c0_index, 4] = clk

		winrate_tot[c0_index, 4] = win_rate
		cpm_tot[c0_index, 4] = cpm
		ecpc_tot[c0_index, 4] = ecpc
		profit_tot[c0_index, 4] = return_ctr * eCPC - cost
		ROI_tot[c0_index, 4] = (return_ctr * eCPC - cost) / cost

		# RLB with constant risk tendency
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "CRTRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
		model_path = data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)
		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=True)

		(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, c0_index, 4]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, c0,
												 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=False)  # False

		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

		click_tot[c0_index, 3] = clk

		winrate_tot[c0_index, 3] = win_rate
		cpm_tot[c0_index, 3] = cpm
		ecpc_tot[c0_index, 3] = ecpc
		profit_tot[c0_index, 3] = return_ctr * eCPC - cost
		ROI_tot[c0_index, 3] = (return_ctr * eCPC - cost) / cost

	return click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot, clk_stat



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

def main():
	# processNum = 2
	# p = Pool(processNum)
	# for camp_index in range(len(camps)):
	# 	camp = camps[camp_index]
	# 	p.apply_async(run, [camp, camp_index])
	# p.close
	# p.join
	# plot()
	os.system("taskset -p -c 0-10 %d" % os.getpid())
	start_time = time.time()
	# summary result
	click_tot = np.zeros([len(camps), len(c0_vec), Algo_num])  #
	winrate_tot = np.zeros([len(camps), len(c0_vec), Algo_num])
	cpm_tot = np.zeros([len(camps), len(c0_vec), Algo_num])
	ecpc_tot = np.zeros([len(camps), len(c0_vec), Algo_num])
	clk_stat_tot = np.zeros([len(camps), np.ceil(max_market_price / clk_stat_interval).astype(int), len(c0_vec), Algo_num])

	processNum = 20
	#Parallel(processNum, 'multiprocessing', max_nbytes=None)(delayed(run)(camps[camp_index]) for camp_index in range(len(camps)))
	click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot, clk_stat_tot = zip(*Parallel(processNum, 'multiprocessing', max_nbytes=None)(
		delayed(run)(camps[camp_index]) for camp_index in range(len(camps))))

	click_tot = np.array(click_tot)
	winrate_tot = np.array(winrate_tot)
	cpm_tot = np.array(cpm_tot)
	ecpc_tot = np.array(ecpc_tot)
	profit_tot = np.array(profit_tot)
	ROI_tot = np.array(ROI_tot)
	clk_stat_tot = np.array(clk_stat_tot)
	print("profit_tot={}".format(profit_tot))

	log_in = open(config.ipinyouPath + "result/proroi.txt", "w")
	#a0, a1, a2 = profit_tot.shape()
	log_in.write('profit' + "\n")
	log_in.write("{:>8}\t {:>8}\t {:>8}\t {:>8}\t {:>8}\t".format("Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
	for i in range(profit_tot.shape[0]):
		for j in range(profit_tot.shape[2]):
			log_in.write(str(profit_tot[i, 0, j]) + "\t")
		log_in.write('\n')
	#### ROI
	log_in.write('ROI' + "\n")
	log_in.write("{:>8}\t {:>8}\t {:>8}\t {:>8}\t {:>8}\t".format("Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
	for i in range(ROI_tot.shape[0]):
		for j in range(ROI_tot.shape[2]):
			log_in.write(str(ROI_tot[i, 0, j]) + '\t')
		log_in.write('\n')
	log_in.close()

	results_file = config.ipinyouPath + "result/results_total.pickle"
	with open(results_file, 'wb') as f:
		RTB_results = {'click': click_tot, 'win_rate': winrate_tot, 'cpm': cpm_tot, 'ecpc': ecpc_tot}
		pickle.dump(RTB_results, f)

	cost_sum = ecpc_tot * click_tot * 1000
	imp_sum = cost_sum / cpm_tot
	auction_sum = imp_sum / winrate_tot * 100

	clk_sum = np.sum(click_tot, 0)
	cost_sum = np.sum(cost_sum, 0)
	imp_sum = np.sum(imp_sum, 0)
	auction_sum = np.sum(auction_sum, 0)


	winrate_tot = imp_sum / auction_sum * 100
	cpm_tot = (cost_sum / 1000) / imp_sum * 1000
	ecpc_tot = (cost_sum / 1000) / clk_sum
	click_tot = np.mean(click_tot, 0)


	print("shape:{}".format(click_tot.shape))
	print("shape:{}".format(winrate_tot.shape))
	print("shape:{}".format(cpm_tot.shape))
	print("shape:{}".format(ecpc_tot.shape))
	print("shape:{}".format(clk_stat_tot.shape))

	plot(click_tot, winrate_tot, cpm_tot, ecpc_tot, clk_stat_tot)
	print("time:{}".format(time.time()-start_time))

if __name__=="__main__":
	main()
