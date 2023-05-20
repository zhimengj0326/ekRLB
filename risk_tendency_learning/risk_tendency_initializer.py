from utility import *
import config
import numpy as np

class rt_initializer:
	up_precision = 1e-10
	zero_precision = 1e-12

	def __init__(self, camp_info, gamma, avg_m_price, N, B):
		self.cpm = camp_info['budget'] / camp_info['imp']
		self.theta_avg = camp_info['click'] / camp_info['imp']
		self.ctr_avg = np.mean(camp_info['data'][:, 2])
		self.risk_avg = np.mean(camp_info['data'][:, 3])
		self.gamma = gamma
		self.avg_m_price = avg_m_price

		self.risk_tendency = np.ones([N + 1, B + 1])
		self.V = []
		self.D = []

	def calc_risk_tendency(self, N, B, max_bid, m_pdf, const_risk,  ifrisk, ifconst_risktendency):
		if ifrisk and not ifconst_risktendency:
			e_b_max = 0
			e_b_cul = np.array([0] * (max_bid + 1)).astype(float)
			e_b_bar = self.avg_m_price
			alpha = 1
			sum_pdf = np.sum(m_pdf) / max_bid
			for i in range(1, len(m_pdf)):
				e_b_cul[i] = e_b_cul[i - 1] + float(i) * m_pdf[i]
				e_b_max += i * m_pdf[i]

			risk_tendency = np.zeros([N + 1, B + 1])
			e_b = np.zeros([N+1, B+1])
			for i in range(1, N+1):
				for j in range(1, B+1):
					if float(j) / float(i) > e_b_max:
						e_b[i, j] = max_bid
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
		return self.risk_tendency


	def calc_optimal_value_function_with_approximation_i(self, N, B, max_bid, m_pdf, save_path, const_risk, ifrisk, ifconst_risktendency):
		#print(getTime() + "\tvalue function with approx_i, N={}, B={}, save in {}".format(N, B, save_path))
		V_out = open(save_path, "w")
		V = [0] * (B + 1)
		nV = [0] * (B + 1)
		V_max = 0
		V_inc = 0
		self.calc_risk_tendency(N, B, max_bid, m_pdf, const_risk, ifrisk, ifconst_risktendency)
		a_max = max_bid
		for b in range(0, a_max + 1):
			V_inc += m_pdf[b] * (self.theta_avg)
		for n in range(1, N):
			a = [0] * (B + 1)
			bb = B - 1
			for b in range(B, 0, -1):
				while bb >= 0 and self.gamma * (V[bb] - V[b]) + (self.theta_avg + self.risk_tendency[n, b] * self.risk_avg) >= 0:
					bb -= 1
				if bb < 0:
					a[b] = min(max_bid, b)
				else:
					a[b] = min(max_bid, b - bb - 1)

			for b in range(0, B):
				V_out.write("{}\t".format(V[b]))
			V_out.write("{}\n".format(V[B]))

			V_max = self.gamma * V_max + V_inc
			flag = False
			for b in range(1, B + 1):
				nV[b] = self.gamma * V[b]
				for delta in range(0, a[b] + 1):
					nV[b] += m_pdf[delta] * ((self.theta_avg + self.risk_tendency[n, b] * self.risk_avg) + self.gamma * (V[b - delta] - V[b]) )
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
		for b in range(0, B):
			V_out.write("{0}\t".format(V[b]))
		V_out.write("{0}\n".format(V[B]))
		V_out.flush()
		V_out.close()


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



