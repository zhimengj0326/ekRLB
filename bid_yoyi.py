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
#c0_vec = [1 / 32]
c0_vec = [1 / 32, 1/16, 1/8, 1/4, 1/2]
Algo_num = 5
clk_stat_interval = 10
#c0_vec = [1/16]
gamma = 1

src = "yoyi"

log_in = open(config.projectPath + "YOYI/data/bid-performance/{}_N={}_obj={}_clkvp={}.txt".format(src, N, obj_type, clk_vp),
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

def run(c0):
	# summary result
	click_tot = np.zeros([Algo_num])  #len(c0_vec)
	winrate_tot = np.zeros([Algo_num])
	cpm_tot = np.zeros([Algo_num])
	ecpc_tot = np.zeros([Algo_num])

	profit_tot = np.zeros([Algo_num])
	ROI_tot = np.zeros([Algo_num])

	camp ="1458"
	camp_info, auction_in, eCPC = config.get_camp_info(camp, src, ifmix=True)
	camp_info1, auction_in1, eCPC = config.get_camp_info(camp, "ipinyou", ifmix=True)
	print("read data done")
	print("shape of data={}".format(np.array(camp_info['data']).shape))
	#auction_in = open(data_path + camp + "/test.theta.txt", "r")
	cpm = clk_vp * camp_info['budget'] / camp_info['click']
	print(cpm)
	opt_obj = Opt_Obj(obj_type, cpm)
	avg_m_price = np.mean(camp_info['data'][:, 1])
	m_pdf = calc_m_pdf(camp_info['mprice_count'])
	m_pdf = np.array(m_pdf)[:, 0]
	const_risk_tendency = -0.1
	clk_stat = np.zeros([np.ceil(max_market_price / clk_stat_interval).astype(int), Algo_num])

	#for c0_index in range(len(c0_vec)):
	#c0 = c0_vec[c0_index]
	B = int(cpm * c0 * N)
	## risk = np.mean(np.array(auction_in[:, 3]).astype(float))
	rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma, avg_m_price, auction_in['data'], N, B)
	const_risk = -0.1

	if c0 == 1/2:
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=False) # RRLB
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=False, ifconst_risktendency=False) #RLB
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=True) # CRTRLB

	print("V function update done")

	# Lin-Bid
	lin_bid = Lin_Bid(camp_info)
	setting = "{},  algo={}, N={}, c0={}" \
		.format(src, "lin_bid", N, c0)
	bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
	model_path = config.ipinyouPath + "bid_log/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
	# valid_path = config.ipinyouPath + 'iPinYou/' + 'train_' + camp + 'n_MR_meanstd.pickle'
	lin_bid.parameter_tune(opt_obj, camp_info['data'], model_path, N, B, c0, max_market_price, max_market_price,
						   clk_stat_interval, load=True)
	rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, False,
								ifconst_risktendency=False)
	(auction, imp, clk, return_ctr, cost, clk_stat[:, 0]) = lin_bid.run(auction_in['data'], bid_log_path,
																				  N, B, max_market_price,
																				  clk_stat_interval, save_log=True)

	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	click_tot[0] = clk
	winrate_tot[0] = win_rate
	cpm_tot[0] = cpm
	ecpc_tot[0] = ecpc
	profit_tot[0] = return_ctr * eCPC - cost
	ROI_tot[0] = (return_ctr * eCPC - cost) / cost

	# RLB
	# rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
	setting = "{}, algo={}, N={}, c0={}" \
		.format(src, "RLB", N, c0)
	bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
															  data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(
																  camp, N, c0), const_risk, ifrisk=False,
															  ifconst_risktendency=False)  # RLB

	model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
	rlb_dp_i.load_value_function(N, B, model_path)
	rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, False,
								ifconst_risktendency=False)
	(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, 1]) = rlb_dp_i.run(auction_in['data'],
																								bid_log_path, N, B,
																								max_market_price,
																								clk_stat_interval,
																								save_log=True,
																								ifconst_risk=False)  # False

	win_rate = imp / auction * 100
	cpm = 1 #(cost / 1000) / imp * 1000
	ecpc = 1 #(cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	click_tot[1] = clk
	winrate_tot[1] = win_rate
	cpm_tot[1] = cpm
	ecpc_tot[1] = ecpc
	profit_tot[1] = return_ctr * eCPC - cost
	ROI_tot[1] = (return_ctr * eCPC - cost) / cost

	# RLB with risk
	setting = "{}, algo={}, N={}, c0={}" \
		.format(src, "RRLB", N, c0)
	bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

	#model_path = data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
	#rlb_dp_i.load_value_function(N, B, model_path)

	rlb_dp_i.alpha_tune(src, camp, camp_info, m_pdf, const_risk_tendency, camp_info['data'], bid_log_path, N, B, max_market_price, clk_stat_interval)

	rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)

	(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, 2]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, B,
											 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=False)  # False

	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	click_tot[2] = clk

	winrate_tot[2] = win_rate
	cpm_tot[2] = cpm
	ecpc_tot[2] = ecpc
	profit_tot[2] = return_ctr * eCPC - cost
	ROI_tot[2] = (return_ctr * eCPC - cost) / cost

	# RLB with constant risk
	setting = "{}, camp={}, algo={}, N={}, c0={}" \
		.format(src, camp, "CRRLB", N, c0)
	bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
	#rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)
	aaa = np.mean(np.array(auction_in['data'][:, 3]).astype(float))
	print("test={}".format(aaa))

	#######
	rlb_dp_i.risk_tune(src, camp, camp_info['data'], bid_log_path, N, B, max_market_price, clk_stat_interval)
	(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, 3]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, B,
											 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=True)  # False

	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	click_tot[4] = clk

	winrate_tot[4] = win_rate
	cpm_tot[4] = cpm
	ecpc_tot[4] = ecpc
	profit_tot[4] = return_ctr * eCPC - cost
	ROI_tot[4] = (return_ctr * eCPC - cost) / cost

	# RLB with constant risk tendency
	setting = "{}, camp={}, algo={}, N={}, c0={}" \
		.format(src, camp, "CRTRLB", N, c0)
	bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
	#model_path = data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)

	rlb_dp_i.risk_tendency_tune(src, camp, camp_info, model_path, camp_info['data'], bid_log_path, N, B, max_market_price, clk_stat_interval)
	#rlb_dp_i.load_value_function(N, B, model_path)
	(auction, imp, clk, return_ctr, return_risk, cost, clk_stat[:, 4]) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, B,
											 max_market_price, clk_stat_interval, save_log=True, ifconst_risk=False)  # False

	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	click_tot[3] = clk

	winrate_tot[3] = win_rate
	cpm_tot[3] = cpm
	ecpc_tot[3] = ecpc
	profit_tot[3] = return_ctr * eCPC - cost
	ROI_tot[3] = (return_ctr * eCPC - cost) / cost

	return click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot, clk_stat

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

	processNum = 5
	#Parallel(processNum, 'multiprocessing', max_nbytes=None)(delayed(run)(camps[camp_index]) for camp_index in range(len(camps)))

	click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot, clk_stat_tot = zip(*Parallel(processNum, 'multiprocessing', max_nbytes=None)(
		delayed(run)(c0_vec[c0_index]) for c0_index in range(len(c0_vec)))) ### len(camps)

	#click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot, clk_stat_tot = run(camps[0])

	click_tot = np.array(click_tot)
	winrate_tot = np.array(winrate_tot)
	cpm_tot = np.array(cpm_tot)
	ecpc_tot = np.array(ecpc_tot)
	profit_tot = np.array(profit_tot)
	ROI_tot = np.array(ROI_tot)
	clk_stat_tot = np.array(clk_stat_tot)
	print("profit_tot={}".format(profit_tot))

	results_file = config.yoyiPath + "result/proroi.pickle"
	with open(results_file, 'wb') as f:
		RTB_perf = {'profit': profit_tot, 'ROI': ROI_tot, 'click': click_tot}
		pickle.dump(RTB_perf, f)

	log_in = open(config.yoyiPath + "result/proroi.txt", "w")
	#a0, a1, a2 = profit_tot.shape()

	log_in.write('Average result_click' + "\n")
	for c0_index in range(len(c0_vec)):
		log_in.write("{}\t{}\t{}\t{}\t{}\t{}\t".format("c0", "Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
		log_in.write(str(c0_vec[c0_index]) + '\t')
		for j in range(click_tot.shape[-1]):
			log_in.write(str(click_tot[0, j]) + '\t')
		log_in.write('\n')

	log_in.write('Average result_profit' + "\n")
	for c0_index in range(len(c0_vec)):
		log_in.write("{}\t{}\t{}\t{}\t{}\t{}\t".format("c0", "Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
		log_in.write(str(c0_vec[c0_index]) + '\t')
		for j in range(click_tot.shape[-1]):
			log_in.write(str(profit_tot[0, j]) + '\t')
		log_in.write('\n')

	log_in.write('Average result_ROI' + "\n")
	for c0_index in range(len(c0_vec)):
		log_in.write("{}\t{}\t{}\t{}\t{}\t{}\t".format("c0", "Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
		log_in.write(str(c0_vec[c0_index]) + '\t')
		for j in range(click_tot.shape[-1]):
			log_in.write(str(ROI_tot[0, j]) + '\t')
		log_in.write('\n')
	log_in.close()

	results_file = config.yoyiPath + "result/results_total.pickle"
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

	# plot(click_tot, winrate_tot, cpm_tot, ecpc_tot, clk_stat_tot)
	print("time:{}".format(time.time()-start_time))

if __name__=="__main__":
	main()
