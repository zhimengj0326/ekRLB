import pickle
import os
import numpy as np

dataPath = os.path.join(os.getcwd(), '../make-ipinyou-data-master/output/data/')
projectPath = dataPath

ipinyouPath = dataPath
vlionPath = dataPath + "vlion-data/"
yoyiPath = dataPath + "YOYI/data/"

ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
#ipinyou_camps = ["1458", "2259", "2261", "2821"]
vlion_camps = ["20160727"]
yoyi_camps = ["121"]

ipinyou_max_market_price = 300
vlion_max_market_price = 300
yoyi_max_market_price = 1000

info_keys = ["imp_test", "cost_test", "clk_test", "imp_train", "cost_train", "clk_train", "field", "dim", "price_counter_train"]


# info_keys:imp_test   cost_test   clk_test    clk_train   imp_train   field   cost_train  dim  price_counter_train
def get_camp_info(camp, src="ipinyou", ifmix=False):
	if src == "ipinyou":
		filename_train = ipinyouPath + 'iPinYou/' + 'train_' + camp + 'n_MR_meanstd.pickle'
		if ifmix:
			filename_test = ipinyouPath + 'iPinYou/' + 'test_mix_' + camp + 'n_MR_meanstd.pickle'
		else:
			filename_test = ipinyouPath + 'iPinYou/' + 'test_' + camp + 'n_MR_meanstd.pickle'
		# filename_train = ipinyouPath + 'train_' + camp + 'n.multireward_meanstd'
		# filename_test = ipinyouPath + 'test_' + camp + 'n.multireward_meanstd'
	elif src == "vlion":
		info = pickle.load(open(vlionPath + camp + "/info.txt", "rb"))
	elif src == "yoyi":
		filename_train = yoyiPath + 'train_risk.pickle'
		filename_test = yoyiPath + 'test_risk.pickle'

	train_data = pickle.load(open(filename_train, "rb"))
	# riskvalue = np.sum(train_data['data'][:, 3]) / np.sum(train_data['data'][:, 2])
	riskvalue = 1
	train_data['data'][:, 3] = train_data['data'][:, 3] / riskvalue
	eCPC = train_data['budget'] / train_data['click']
	test_data = pickle.load(open(filename_test, "rb"))
	test_data['data'][:, 3] = test_data['data'][:, 3] / riskvalue

	# #data_bid = pickle.load(open(filename_train, "rb"))# click, market_price, CTR, risk
	# data_bid = np.loadtxt(filename_train)
	# data_bid[:, 0] = data_bid[:, 0].astype(int)
	# data_bid[:, 1] = data_bid[:, 1].astype(int)
	# train_imp = data_bid.shape[0]
	# train_budget = np.sum(np.array(data_bid[:, 1]))
	# train_click = np.sum(np.array(data_bid[:, 0]))
	#
	# max_price = 300
	# m_counter = np.zeros([max_price + 1, 1])
	# for i in range(max_price + 1):
	# 	m_counter[i] = list(data_bid[:, 1]).count(i)
	#
	# train_data = {'imp': train_imp, 'budget': train_budget, 'click': train_click, 'mprice_count': m_counter, 'data': data_bid}
	#
	#
	# #data_bid = pickle.load(open(filename_test, "rb"))
	# data_bid = np.loadtxt(filename_test)
	# data_bid[:, 0] = data_bid[:, 0].astype(int)
	# data_bid[:, 1] = data_bid[:, 1].astype(int)
	# test_imp = data_bid.shape[0]
	# test_budget = np.sum(np.array(data_bid[:][1]))
	# test_click = np.sum(np.array(data_bid[:, 0]))
	#
	# m_counter = np.zeros([max_price + 1, 1])
	# for i in range(max_price + 1):
	# 	m_counter[i] = list(data_bid[:, 1]).count(i)
	#
	# test_data = {'imp': test_imp, 'budget': test_budget, 'click': test_click, 'mprice_count': m_counter, 'data': data_bid}
	#
	# filename_train2 = ipinyouPath + 'iPinYou/' + 'train_' + camp + 'n_MR_meanstd.pickle'
	# filename_test2 = ipinyouPath + 'iPinYou/' + 'test_' + camp + 'n_MR_meanstd.pickle'
	#
	# with open(filename_train2, 'wb') as f:
	# 	pickle.dump(train_data, f)
	# with open(filename_test2, 'wb') as f2:
	# 	pickle.dump(test_data, f2)


	return train_data, test_data

def get_mix_camp_info(camp, src="ipinyou"):
	if src == "ipinyou":
		filename_test = ipinyouPath + 'test_' + camp + 'n.multireward_meanstd'
	elif src == "vlion":
		info = pickle.load(open(vlionPath + camp + "/info.txt", "rb"))
	elif src == "yoyi":
		info = pickle.load(open(yoyiPath + camp + "/info.txt", "rb"))

	max_price = 300
	auction_in = open(dataPath + "rlb_data/test_theta_{}.txt".format(camp), "r")

	#data_bid = pickle.load(open(filename_test, "rb"))
	data_bid = np.loadtxt(filename_test)
	data_bid[:, 0] = data_bid[:, 0].astype(int)
	data_bid[:, 1] = data_bid[:, 1].astype(int)
	i = 0
	for line in auction_in:
		line = line[:len(line) - 1].split(" ")
		data_bid[i, 2] = float(line[2])
		i += 1
	test_imp = data_bid.shape[0]
	test_budget = np.sum(np.array(data_bid[:][1]))
	test_click = np.sum(np.array(data_bid[:, 0]))

	m_counter = np.zeros([max_price + 1, 1])
	for i in range(max_price + 1):
		m_counter[i] = list(data_bid[:, 1]).count(i)

	test_data = {'imp': test_imp, 'budget': test_budget, 'click': test_click, 'mprice_count': m_counter, 'data': data_bid}

	filename_test2 = ipinyouPath + 'iPinYou/' + 'test_mix_' + camp + 'n_MR_meanstd.pickle'


	with open(filename_test2, 'wb') as f2:
		pickle.dump(test_data, f2)

