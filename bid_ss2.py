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
import argparse

obj_type = "clk"
clk_vp = 1
N = 1000
# c0_vec = [1 / 32, 1/16]
c0_vec = [1 / 32, 1 / 16, 1/8, 1/4, 1/2]
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

def run(camp, c0, args):
	camp_info, auction_in, eCPC = config.get_camp_info(camp, src, ifmix=True)
	#auction_in = open(data_path + camp + "/test.theta.txt", "r")
	opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info['budget'] / camp_info['click']))
	avg_m_price = np.mean(camp_info['data'][:, 1])
	m_pdf = calc_m_pdf(camp_info['mprice_count'])
	m_pdf = np.array(m_pdf)[:, 0]
	const_risk_tendency = -0.1

	#for c0_index in range(len(c0_vec)):
	#c0 = c0_vec[c0_index]
	B = int(camp_info['budget'] / camp_info['imp'] * c0 * N)
	aaa = camp_info['data'][:, 1].shape[0]
	print(f'camp_info_len={aaa}')
	print(f'B={B*aaa/N} c0={c0}')
	## risk = np.mean(np.array(auction_in[:, 3]).astype(float))
	rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma, avg_m_price, auction_in['data'], N, B)
	const_risk = -0.1
	# if args.alg=="RRLB": # and not os.path.exists(data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):
	# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
	# 			data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=False) # RRLB
	# if args.alg=="RLB": # and not os.path.exists(data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):	
	# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
	# 			data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=False, ifconst_risktendency=False) #RLB
	# if args.alg=="CRTRLB": # and not os.path.exists(data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):		
	# 	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
	# 			data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=True) # CRTRLB

	# Lin-Bid
	if args.alg=="Lin":
		lin_bid = Lin_Bid(camp_info)
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "lin_bid", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)
		model_path = data_path + "bid_log/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
		# valid_path = config.ipinyouPath + 'iPinYou/' + 'train_' + camp + 'n_MR_meanstd.pickle'
		lin_bid.parameter_tune(opt_obj, camp_info['data'], model_path, N, B, c0, max_market_price, max_market_price,
							load=True)
		# rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, False,
		# 							ifconst_risktendency=False)
		(auction, imp, clk, return_ctr, cost) = lin_bid.run(auction_in['data'], bid_log_path,
															N, B, max_market_price, save_log=True)
	# RLB
	elif args.alg=="RLB":
		# rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "RLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)

		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, False,
									ifconst_risktendency=False)
		(auction, imp, clk, return_ctr, return_risk, cost) = rlb_dp_i.run(auction_in['data'],
																		bid_log_path, N, B,
																		max_market_price,
																		save_log=True,
																		ifconst_risk=False)  # False
	# RLB with risk
	elif args.alg=="RRLB":
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "RRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)

		rlb_dp_i.alpha_tune(src, camp, camp_info, m_pdf, const_risk_tendency, camp_info['data'], bid_log_path, N, B, c0, max_market_price)

		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)

		(auction, imp, clk, return_ctr, return_risk, cost) = rlb_dp_i.run(auction_in['data'], 
																		bid_log_path, N, B,
																		max_market_price, save_log=True, 
																		ifconst_risk=False)  # False

	# RLB with constant risk
	elif args.alg=="CRRLB":
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "CRRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)

		rlb_dp_i.calc_risk_tendency(N, B, max_market_price, m_pdf, const_risk_tendency, True, ifconst_risktendency=False)
		
		#######
		rlb_dp_i.risk_tune(src, camp, camp_info['data'], bid_log_path, N, B, c0, max_market_price)
		(auction, imp, clk, return_ctr, return_risk, cost) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, B,
												max_market_price, save_log=True, ifconst_risk=True)  # False
	
	
	# RLB with constant risk tendency
	else:
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "CRTRLB", N, c0)
		bid_log_path = config.projectPath + "bid_log/{}.txt".format(setting)

		model_path = data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)
		rlb_dp_i.load_value_function(N, B, model_path)

		rlb_dp_i.risk_tendency_tune(src, camp, camp_info, model_path, camp_info['data'], bid_log_path, N, B, c0, max_market_price)
		
		(auction, imp, clk, return_ctr, return_risk, cost) = rlb_dp_i.run(auction_in['data'], bid_log_path, N, B,
												max_market_price, save_log=True, ifconst_risk=False)  # False
	
	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000 if imp>0 else 0
	ecpc = (cost / 1000) / clk if clk>0 else 0
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")
	
	profit = return_ctr * ecpc - cost
	ROI = (return_ctr * ecpc - cost) / cost if cost>0 else 0

	return clk, win_rate, cpm, ecpc, profit, ROI

def parse_args():
    parser = argparse.ArgumentParser(description='CNN on MNIST')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--alg', default='RLB', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--save', default='cnn_mnist.pth', type=str)

    return parser.parse_args()

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
	args = parse_args()
	# summary result
	click_tot = np.zeros([len(camps), len(c0_vec)])  #
	winrate_tot = np.zeros([len(camps), len(c0_vec)])
	cpm_tot = np.zeros([len(camps), len(c0_vec)])
	ecpc_tot = np.zeros([len(camps), len(c0_vec)])

	processNum = 45
	#Parallel(processNum, 'multiprocessing', max_nbytes=None)(delayed(run)(camps[camp_index]) for camp_index in range(len(camps)))

	click_tot, winrate_tot, cpm_tot, ecpc_tot, profit_tot, ROI_tot = zip(*Parallel(processNum, 'multiprocessing', max_nbytes=None)(
		delayed(run)(camps[camp_index], c0_vec[c0_index], args)
        for camp_index in range(len(camps)) for c0_index in range(len(c0_vec)))) ### len(camps)


	click_tot = np.array(click_tot).reshape((len(camps), len(c0_vec)))
	winrate_tot = np.array(winrate_tot).reshape((len(camps), len(c0_vec)))
	cpm_tot = np.array(cpm_tot).reshape((len(camps), len(c0_vec)))
	ecpc_tot = np.array(ecpc_tot).reshape((len(camps), len(c0_vec)))
	profit_tot = np.array(profit_tot).reshape((len(camps), len(c0_vec)))
	ROI_tot = np.array(ROI_tot).reshape((len(camps), len(c0_vec)))
	print("profit_tot={}".format(profit_tot))

	results_file = config.ipinyouPath + "result/{}_proroi.pickle".format(args.alg)
	with open(results_file, 'wb') as f:
		RTB_perf = {'profit': profit_tot, 'ROI': ROI_tot, 'click': click_tot}
		pickle.dump(RTB_perf, f)

	log_in = open(config.ipinyouPath + "result/{}_proroi.txt".format(args.alg), "w")
	#a0, a1, a2 = profit_tot.shape()
	log_in.write('profit' + "\n")
	for i in range(len(camps)):
		log_in.write("{}".format(camps[i]) + "\t")
		for c0_index in range(len(c0_vec)):
			log_in.write(str(profit_tot[i, c0_index]) + "\t")
		log_in.write('\n')
		# #### ROI
		# log_in.write('ROI' + "\n")
		# log_in.write("{:>8}\t {:>8}\t {:>8}\t {:>8}\t {:>8}\t {:>8}\t".format("camp", "Lin", "RLB", "RRLB", "CRTRLB", "CRRLB") + "\n")
		# for i in range(len(camps)):
		# 	log_in.write("{}".format(camps[i]) + "\t")
		# 	for j in range(ROI_tot.shape[-1]):
		# 		log_in.write(str(ROI_tot[i * len(c0_vec) + c0_index, j]) + '\t')
		# 	log_in.write('\n')
		#### click number
	log_in.write('click_number' + "\n")
	for i in range(len(camps)):
		log_in.write("{}".format(camps[i]) + "\t")	
		for c0_index in range(len(c0_vec)):
			log_in.write(str(click_tot[i, c0_index]) + '\t')
		log_in.write('\n')
	log_in.write("avg" + "\t")
	for c0_index in range(len(c0_vec)):
		log_in.write(str(np.mean(click_tot[:, c0_index])) + '\t')
	log_in.write('\n')	
	log_in.close

	results_file = config.ipinyouPath + "result/results_total.pickle"
	with open(results_file, 'wb') as f:
		RTB_results = {'click': click_tot, 'win_rate': winrate_tot, 'cpm': cpm_tot, 'ecpc': ecpc_tot}
		pickle.dump(RTB_results, f)

	# cost_sum = ecpc_tot * click_tot * 1000
	# imp_sum = cost_sum / cpm_tot
	# auction_sum = imp_sum / winrate_tot * 100

	# clk_sum = np.sum(click_tot, 0)
	# cost_sum = np.sum(cost_sum, 0)
	# imp_sum = np.sum(imp_sum, 0)
	# auction_sum = np.sum(auction_sum, 0)


	# winrate_tot = imp_sum / auction_sum * 100
	# cpm_tot = (cost_sum / 1000) / imp_sum * 1000
	# ecpc_tot = (cost_sum / 1000) / clk_sum
	# click_tot = np.mean(click_tot, 0)


	print("time:{}".format(time.time()-start_time))

if __name__=="__main__":
	main()
