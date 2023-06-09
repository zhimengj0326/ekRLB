from utility import *
import os
import numpy as np

class Lin_Bid:
	def __init__(self, camp_info):
		self.theta_avg = camp_info['click'] / camp_info['imp']
		self.b0 = 0
		self.step = 5
		self.valid_rate = 1
		self.min_valid = 300000

	def parameter_tune(self, opt_obj, valid_set, save_path, N, B, c0, max_b0, max_bid, load=True):
		if load and os.path.isfile(save_path):
			var_map = pickle.load(open(save_path, "rb"))
			self.b0 = var_map["b0"]
			obj = var_map["best_obj"]
		else:
			#valid_set = []
			#B = int(self.cpm * c0 * N)
			# with open(valid_path, "r") as fin:
			# 	n = N
			# 	episode_set = []
			# 	for line in fin:
			# 		line = line[:len(line) - 1].split(delimiter)
			# 		click = int(line[0])
			# 		price = int(line[1])
			# 		theta = float(line[2])
			# 		episode_set.append((click, price, theta))
			#
			# 		n -= 1
			# 		if n == 0:
			# 			n = N
			# 			valid_set.extend(episode_set)
			# 			episode_set.clear()
			# valid_set = valid_set[:max(self.min_valid, int(len(valid_set) * self.valid_rate))]
			valid_set = valid_set[:max(self.min_valid, int(len(valid_set) * self.valid_rate)), :]
			obj = 0
			bb = 0
			tune_list = []
			kp_dc = 0
			for bc in range(self.step, max_b0 + self.step, self.step):
				self.b0 = bc
				(auction, imp, clk, return_ctr, cost) = self.run(valid_set, None, N, B, max_bid, False)
				perf = opt_obj.get_obj(imp, clk, cost)
				tune_list.append((bc, perf))
				if perf >= obj:
					obj = perf
					bb = bc
					kp_dc = 0
				else:
					kp_dc += 1
				if kp_dc >= 5:
					break
			# if bb == max_b0:
			# 	bc = max_b0 + self.step
			# 	while True:
			# 		self.b0 = bc
			# 		(auction, imp, clk, return_ctr, cost, clk_stat) = self.run(valid_set, None, N, B, max_bid, clk_stat_interval,\
			# 		                                     save_log=False)
			# 		perf = opt_obj.get_obj(imp, clk, cost)
			# 		tune_list.append((bc, perf))
			# 		if perf > obj:
			# 			obj = perf
			# 			bb = bc
			# 			bc += self.step
			# 		else:
			# 			break
			# 	for _i in range(5):
			# 		bc += self.step
			# 		self.b0 = bc
			# 		(auction, imp, clk, return_ctr, cost, clk_stat) = self.run(valid_set, None, N, B, max_bid, clk_stat_interval,
			# 		                                     save_log=False)
			# 		perf = opt_obj.get_obj(imp, clk, cost)
			# 		tune_list.append((bc, perf))
			self.b0 = bb
			pickle.dump({"b0": self.b0, "best_obj": obj, "tune_list": tune_list}, open(save_path, "wb"))
			print("Lin-Bid parameter tune: b0={}, best_obj={} and save in {}".format(self.b0, obj, save_path))

	def run(self, auction_in, bid_log_path, N, B, max_bid, save_log=False):
		auction = 0
		imp = 0
		clk = 0
		cost = 0
		return_ctr = 0

		if save_log:
			log_in = open(bid_log_path, "w")
		#B = int(self.cpm * c0 * N)

		episode = 1
		n = N
		b = B
		for line in range(np.array(auction_in).shape[0]):
			# if input_type == "file reader":
			# 	line = line[:len(line) - 1].split(delimiter)
			# 	click = int(line[0])
			# 	price = int(line[1])
			# 	theta = float(line[2])
			# 	risk = float(line[3])
			# else:
			# 	(click, price, theta) = line

			click = int(auction_in[line, 0])
			price = int(auction_in[line, 1])
			theta = float(auction_in[line, 2])
			risk = float(auction_in[line, 3])

			a = min(int(theta * self.b0 / self.theta_avg), max_bid)
			a = min(b, a)

			log = getTime() + "\t{}\t{}_{}\t{}_{}_{}\t{}_{}\t".format(
				episode, b, n, a, price, click, clk, imp)
			if save_log:
				log_in.write(log + "\n")

			if a >= price:
				imp += 1
				return_ctr += theta
				if click == 1:
					clk += 1
				b -= price
				cost += price
			n -= 1
			auction += 1

			if n == 0:
				episode += 1
				n = N
				b = B
		if save_log:
			log_in.flush()
			log_in.close()

		return auction, imp, clk, return_ctr, cost
