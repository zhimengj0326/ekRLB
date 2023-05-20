from utility import *
import config
import numpy as np

class RLB_DP_I:
	up_precision = 1e-10
	zero_precision = 1e-12

	def __init__(self, camp_info, opt_obj, gamma, avg_m_price, N, B):
		self.cpm = camp_info['budget'] / camp_info['imp']
		self.theta_avg = camp_info['click'] / camp_info['imp']
		self.ctr_avg = np.mean(camp_info['data'][:, 2])
		self.risk_avg = np.mean(camp_info['data'][:, 3])
		self.opt_obj = opt_obj
		self.gamma = gamma
		self.avg_m_price = avg_m_price
		self.v1 = self.opt_obj.v1
		self.v0 = self.opt_obj.v0
		self.risk_tendency = np.ones([N + 1, B + 1])
		self.V = []
		self.D = []

	def calc_risk_tendency(self, N, B, max_bid, m_pdf, const_risk,  ifrisk = True, ifconst_risktendency=True):
		if ifrisk and not ifconst_risktendency:
			e_b_max = 0
			e_b_cul = np.array([0] * (max_bid + 1))
			e_b_bar = self.avg_m_price
			alpha = 1
			for i in range(1, len(m_pdf)):
				e_b_cul[i] = e_b_cul[i - 1] + i * m_pdf[i]
				e_b_max += i * m_pdf[i]

			risk_tendency = np.zeros([N + 1, B + 1])
			e_b = np.zeros([N+1, B+1])
			for i in range(1, N+1):
				for j in range(1, B+1):
					if float(j) / float(i) > e_b_max:
						e_b[i, j] = e_b_max
					else:
						price_err = np.abs(e_b_cul - float(j) / float(i))
						bat = np.where(price_err == np.min(price_err))
						e_b[i, j] = bat[0][0]
						# for index in range(0, len(m_pdf)):
						# 	if e_b_cul[index] > float(j) / float(i):
						# 		break
						# e_b[i, j] = index
			risk_tendency = np.tanh(alpha * (e_b - e_b_bar) / e_b_bar)
			self.risk_tendency = risk_tendency
		elif ifrisk and ifconst_risktendency:
			self.risk_tendency = np.ones([N + 1, B + 1]) * const_risk
		else:
			self.risk_tendency = np.zeros([N + 1, B + 1])


	def calc_optimal_value_function_with_approximation_i(self, N, B, max_bid, m_pdf, const_risk, ifrisk, ifconst_risktendency):
		#print(getTime() + "\tvalue function with approx_i, N={}, B={}, save in {}".format(N, B, save_path))
		V = [0] * (B + 1)
		nV = [0] * (B + 1)
		V_max = 0
		V_inc = 0
		self.calc_risk_tendency(N, B, max_bid, m_pdf, const_risk, ifrisk, ifconst_risktendency)
		if self.v0 != 0:
			a_max = min(int(self.v1 * self.theta_avg / self.v0), max_bid)
		else:
			a_max = max_bid
		for b in range(0, a_max + 1):
			V_inc += m_pdf[b] * (self.v1 * self.theta_avg - self.v0 * b)
		for n in range(1, N):
			a = [0] * (B + 1)
			bb = B - 1
			for b in range(B, 0, -1):
				while bb >= 0 and self.gamma * (V[bb] - V[b]) + self.v1 * (self.theta_avg + self.risk_tendency[n, b] * self.risk_avg) - self.v0 * (b - bb) >= 0:
					bb -= 1
				if bb < 0:
					a[b] = min(max_bid, b)
				else:
					a[b] = min(max_bid, b - bb - 1)

			V_max = self.gamma * V_max + V_inc
			flag = False
			for b in range(1, B + 1):
				nV[b] = self.gamma * V[b]
				for delta in range(0, a[b] + 1):
					nV[b] += m_pdf[delta] * (self.v1 * (self.theta_avg + self.risk_tendency[n, b] * self.risk_avg) + self.gamma * (V[b - delta] - V[b]) - self.v0 * delta)
				if abs(nV[b] - V_max) < self.up_precision:
					for bb in range(b + 1, B + 1):
						nV[bb] = V_max
					flag = True
					break
			V = nV[:]
			# if flag:
			# 	print(getTime() + "\tround {} end with early stop.".format(n))
			# else:
			# 	print(getTime() + "\tround {} end.".format(n))


	def calc_Dnb(self, N, B, max_bid, m_pdf, save_path):
		print(getTime() + "\tD(n, b), N={}, B={}, save in {}".format(N, B, save_path))
		D_out = open(save_path, "w")
		V = [0] * (B + 1)
		nV = [0] * (B + 1)
		V_max = 0
		V_inc = 0
		if self.v0 != 0:
			a_max = min(int(self.v1 * self.theta_avg / self.v0), max_bid)
		else:
			a_max = max_bid
		for b in range(0, a_max + 1):
			V_inc += m_pdf[b] * (self.v1 * self.theta_avg - self.v0 * b)
		for n in range(1, N):
			a = [0] * (B + 1)
			for b in range(B, 0, -1):
				bb = B - 1
				while bb >= 0 and self.gamma * (V[bb] - V[b]) + self.v1 * self.theta_avg - self.v0 * (b - bb) >= 0:
					bb -= 1
				if bb < 0:
					a[b] = min(max_bid, b)
				else:
					a[b] = min(max_bid, b - bb - 1)

			for b in range(0, B):
				dtb = V[b + 1] - V[b]
				if abs(dtb) < self.zero_precision:
					dtb = 0
				if b == B - 1:
					D_out.write("{}\n".format(dtb))
				else:
					D_out.write("{}\t".format(dtb))

			V_max = self.gamma * V_max + V_inc
			flag = False
			for b in range(1, B + 1):
				nV[b] = self.gamma * V[b]
				for delta in range(0, a[b] + 1):
					nV[b] += m_pdf[delta] * (self.v1 * self.theta_avg + self.gamma * (V[b - delta] - V[b]) - self.v0 * delta)
				if abs(nV[b] - V_max) < self.up_precision:
					for bb in range(b + 1, B + 1):
						nV[bb] = V_max
					flag = True
					break
			V = nV[:]
			if flag:
				print(getTime() + "\tround {} end with early stop.".format(n))
			else:
				print(getTime() + "\tround {} end.".format(n))
		for b in range(0, B):
			dtb = V[b + 1] - V[b]
			if abs(dtb) < self.zero_precision:
				dtb = 0
			if b == B - 1:
				D_out.write("{}\n".format(dtb))
			else:
				D_out.write("{}\t".format(dtb))
		D_out.flush()
		D_out.close()

	def Vnb2Dnb(self, v_path, d_path):
		with open(v_path, "r") as fin:
			with open(d_path, "w") as fout:
				for line in fin:
					line = line[:len(line) - 1].split("\t")
					nl = ""
					for b in range(len(line) - 1):
						d = float(line[b + 1]) - float(line[b])
						if abs(d) < RLB_DP_I.zero_precision:
							d = 0
						if b == len(line) - 2:
							nl += "{}\n".format(d)
						else:
							nl += "{}\t".format(d)
					fout.write(nl)

	def load_value_function(self, N, B, model_path):
		self.V = [[0 for i in range(B + 1)] for j in range(N)]
		with open(model_path, "r") as fin:
			n = 0
			for line in fin:
				line = line[:len(line) - 1].split("\t")
				for b in range(B + 1):
					self.V[n][b] = float(line[b])
				n += 1
				if n >= N:
					break

	def load_Dnb(self, N, B, model_path):
		self.D = [[0 for i in range(B)] for j in range(N)]
		with open(model_path, "r") as fin:
			n = 0
			for line in fin:
				line = line[:len(line) - 1].split("\t")
				for b in range(B):
					self.D[n][b] = float(line[b])
				n += 1
				if n >= N:
					break

	def bid(self, n, b, theta, max_bid):
		a = 0
		if len(self.V) > 0:
			for delta in range(1, min(b, max_bid) + 1):
				if self.v1 * theta + self.gamma * (self.V[n - 1][b - delta] - self.V[n - 1][b]) - self.v0 * delta >= 0:
					a = delta
				else:
					break
		elif len(self.D) > 0:
			value = self.v1 * theta
			for delta in range(1, min(b, max_bid) + 1):
				value -= self.gamma * self.D[n - 1][b - delta] + self.v0
				if value >= 0:
					a = delta
				else:
					break
		return a

	def run(self, auction_in, risk_tendency, N, c0, max_bid, clk_stat_interval, save_log=False, ifconst_risk=False):
		auction = 0
		imp = 0
		clk = 0
		cost = 0
		clk_stat = np.zeros([np.ceil(max_bid / clk_stat_interval).astype(int)])

		B = int(self.cpm * c0 * N)

		episode = 1
		n = N
		b = B
		risk_avg = np.mean(np.array(auction_in[:, 3]).astype(float))
		# for line in auction_in:
		# 	if input_type == "file reader":
		# 		line = line[:len(line) - 1].split(delimiter)
		# 		click = int(line[0])
		# 		price = int(line[1])
		# 		theta = float(line[2])
		# 	else:
		# 		(click, price, theta) = line
		for line in range(np.array(auction_in).shape[0]):

			click = int(auction_in[line, 0])
			price = int(auction_in[line, 1])
			theta = float(auction_in[line, 2])
			if not ifconst_risk:
				risk = float(auction_in[line, 3])
			else:
				risk = risk_avg

			a = self.bid(n, b, theta + risk_tendency[n, b] * risk, max_bid)
			a = min(int(a), min(b, max_bid))


			if a >= price:
				imp += 1
				if click == 1:
					clk += 1
					clk_stat[np.floor((a - 1) / clk_stat_interval).astype(int)] += 1
				b -= price
				cost += price
			n -= 1
			auction += 1

			if n == 0:
				episode += 1
				n = N
				b = B

		return auction, imp, clk, cost, clk_stat

	@staticmethod
	def Dnb_save_points(d_path, out_path, b_bound, n_bound):
		N_bound = 0
		B_bound = 0
		with open(d_path, "r") as fin:
			for line in fin:
				N_bound += 1
				if B_bound == 0:
					line = line[:len(line) - 1].split("\t")
					B_bound = len(line)
		with open(d_path, "r") as fin:
			with open(out_path, "w") as fout:
				fout.write("{}_{}_{}\n".format(N_bound, B_bound, config.vlion_max_market_price))
				n = 0
				for line in fin:
					line = line[:len(line) - 1].split("\t")
					bb = -1
					for b in range(len(line)):
						dnb = float(line[b])
						if abs(dnb) < RLB_DP_I.zero_precision:
							bb = b
							break
					if bb >= 0:
						if n <= n_bound:
							s_ids = bb
						else:
							s_ids = min(bb, b_bound)
						out = "{}_{}_{}\t".format(n, bb, s_ids)
						out += "\t".join(line[:s_ids]) + "\n"
						fout.write(out)
					else:
						if n <= n_bound:
							s_ids = len(line)
						else:
							s_ids = min(b_bound, len(line))
						out = "{}_{}_{}\t".format(n, bb, s_ids)
						out += "\t".join(line[:s_ids]) + "\n"
						fout.write(out)
					n += 1


