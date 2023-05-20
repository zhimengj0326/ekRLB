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
c0_vec = [1 /2]
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
	const_risk = -0.1

	#for c0_index in range(len(c0_vec)):
	#c0 = c0_vec[c0_index]
	B = int(camp_info['budget'] / camp_info['imp'] * c0 * N)
	## risk = np.mean(np.array(auction_in[:, 3]).astype(float))
	rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma, avg_m_price, auction_in['data'], N, B)
	if args.alg=="RRLB": # and not os.path.exists(data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/R_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=False) # RRLB
	if args.alg=="RLB": # and not os.path.exists(data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):	
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/NR_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=False, ifconst_risktendency=False) #RLB
	if args.alg=="CRTRLB": # and not os.path.exists(data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0)):		
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + "bid_model_V/Rcons_{}_v_nb_N={}_c0={}.txt".format(camp, N, c0), const_risk, ifrisk=True, ifconst_risktendency=True) # CRTRLB
	print("update {}_{} done".format(args.alg, c0))

def parse_args():
	parser = argparse.ArgumentParser(description='CNN on MNIST')
	parser.add_argument('--gpu', default='0', type=str)
	parser.add_argument('--alg', default='RLB', type=str)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--times', default=5, type=int)

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
	os.system("taskset -p -c 0-63 %d" % os.getpid())
	start_time = time.time()
	args = parse_args()
	# summary result
	click_tot = np.zeros([len(camps), len(c0_vec)])  #
	winrate_tot = np.zeros([len(camps), len(c0_vec)])
	cpm_tot = np.zeros([len(camps), len(c0_vec)])
	ecpc_tot = np.zeros([len(camps), len(c0_vec)])

	processNum = 45
	#Parallel(processNum, 'multiprocessing', max_nbytes=None)(delayed(run)(camps[camp_index]) for camp_index in range(len(camps)))

	Parallel(processNum, 'multiprocessing', max_nbytes=None)(
		delayed(run)(camps[camp_index], c0_vec[c0_index], args)
		for camp_index in range(len(camps)) for c0_index in range(len(c0_vec))) ### len(camps)



	print("time:{}".format(time.time()-start_time))

if __name__=="__main__":
	main()
