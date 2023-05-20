from deep_q_network import MLP_model
import numpy as np
import os
import pandas as pd
import pickle as pickle
from replay_memory import replay_memory
import matplotlib.pyplot as plt


class RTB_environment:
    """
    This class will construct and manage the environments which the
    agent will interact with. The distinction between training and
    testing environment is primarily in the episode length.
    """

    def __init__(self, camp_info, episode_length, init_budget, camp, learning_rate, norm_state, sess):
        """
        We need to initialize all of the data, which we fetch from the
        campaign-specific dictionary. We also specify the number of possible
        actions, the state, the amount of data which has been trained on,
        and so on.
        :param camp_dict: a dictionary containing data on winning bids, ctr
        estimations, clicks, budget, and so on. We copy the data on bids, ctr
        estimations and clicks; then, we delete the rest of the dictionary.
        :param episode_length: specifies the number of steps in an episode
        :param step_length: specifies the number of auctions per step.
        """

        self.camp = camp
        self.theta_avg = camp_info['click'] / camp_info['imp']
        self.risk_avg = np.mean(camp_info['data'][:, 3])
        self.gamma = 1
        self.up_precision = 1e-10
        self.data_count = 0
        self.total_data = len(camp_info['data'][:, 3])


        self.sess = sess
        self.learning_rate = learning_rate
        # self.batch_size = batch_size
        # self.memory_cap = memory_cap
        # self.replay_memory = replay_memory(self.memory_cap, self.batch_size)

        self.result_dict = {'auctions': 0, 'impressions': 0, 'click': 0, 'cost': 0, 'win-rate': 0, 'eCPC': 0, 'eCPI': 0}

        #self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        self.episode_length = episode_length
        self.bids = []
        self.V = [[0 for i in range(init_budget + 1)] for j in range(episode_length+1)]


        self.time_step = 0
        self.budget = 1
        self.init_budget = 1
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.cost = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0
        self.eCPC = 0
        self.eCPI = 0
        self.termination = True

        self.state = [self.n_regulations, self.budget]
        self.norm_state = norm_state

        # else:
        #     self.rewardnet = q_estimator(len(self.state), len(self.actions), self.learning_rate, 'rewtest')



    def get_camp_data(self):
        """
        This function updates the data variables which are then accessible
        to the step-function. This function also deletes data that has already
        been used and, hence, tries to free up space.
        :return: updated data variables (i.e. bids, ctr estimations and clicks)
        """
        if self.total_data - self.data_count < self.episode_length + 1:   # click, win_price, pctr, ctr_risk
            self.data_count = 0

        ctr_risk = np.array(
            self.camp_dict['data'][self.data_count:self.data_count + self.episode_length, 3])
        ctr_estimations = np.array(
            self.camp_dict['data'][self.data_count:self.data_count + self.episode_length, 2])
        winning_bids = np.array(
            self.camp_dict['data'][self.data_count:self.data_count + self.episode_length, 1])
        winning_bids = winning_bids.astype(int)
        clicks = list(
            self.camp_dict['data'][self.data_count:self.data_count + self.episode_length, 0])

        self.data_count += self.episode_length
        return ctr_risk, ctr_estimations, winning_bids, clicks

    def bid(self, n, b, theta, max_bid):
        a = 0
        if len(self.V) > 0:
            for delta in range(1, min(b, max_bid) + 1):
                bid_try = theta + self.gamma * (self.V[n - 1][b - delta] - self.V[n - 1][b])
                if bid_try >= 0:
                    a = delta
                else:
                    break
        return a

    def calc_optimal_value_function_with_approximation_i(self, N, B, max_bid, m_pdf, risk_tendency):
        # print(getTime() + "\tvalue function with approx_i, N={}, B={}, save in {}".format(N, B, save_path))
        V = [0] * (B + 1)
        nV = [0] * (B + 1)
        V_max = 0
        V_inc = 0

        a_max = max_bid
        for b in range(0, a_max + 1):
            V_inc += m_pdf[b] * (self.theta_avg)
        for n in range(1, N):
            a = [0] * (B + 1)
            bb = B - 1
            for b in range(B, 0, -1):
                while bb >= 0 and self.gamma * (V[bb] - V[b]) + (
                        self.theta_avg + risk_tendency[n, b] * self.risk_avg) >= 0:
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
                    nV[b] += m_pdf[delta] * (
                                (self.theta_avg + risk_tendency[n, b] * self.risk_avg) + self.gamma * (
                                    V[b - delta] - V[b]))
                if abs(nV[b] - V_max) < self.up_precision:
                    for bb in range(b + 1, B + 1):
                        nV[bb] = V_max
                    flag = True
                    break
            V = nV[:]
            self.V[n] = V


    def reset(self, budget):
        """
        This function is called whenever a new episode is initiated
        and resets the budget, the Lambda, the time-step, the termination
        bool, and so on.
        :param budget: the amount of money the bidding agent can spend during
        the period
        :param initial_Lambda: the initial scaling of ctr-estimations to form bids
        :return: initial state, zero reward and a false termination bool
        """
        self.n_regulations = min(self.episode_length, self.data_count)
        self.budget = budget
        self.init_budget = budget * self.n_regulations / self.episode_length

        self.time_step = 0

        self.bids = []
        self.price = []

        budget = self.budget
        self.winning_rate = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0

        # for i in range(min(self.data_count, self.step_length)):
        #     if bids[i] > winning_bids[i] and budget > bids[i]:
        #         budget -= winning_bids[i]
        #         self.impressions += 1
        #         self.cost += winning_bids[i]
        #         self.click += clicks[i]
        #         self.ctr_value += ctr_estimations[i]
        #         self.winning_rate += 1 / min(self.data_count, self.step_length)
        #     else:
        #         continue

        self.state = [self.n_regulations, self.budget]

        self.budget = budget

        self.termination = False

        #return self.state, self.termination

    def episode(self, sess, q_network, max_bid, risk_tendency_table, ifintial=True):
        """
        This function takes an action from the bidding agent (i.e.
        a change in the ctr-estimation scaling, and uses it to compute
        the agent's bids, s.t. it can compare it to the "winning bids".
        If one of the agent's bids exceed a winning bid, it will subtract
        the cost of the impression from the agent's budget, etc, given that
        the budget is not already depleted.
        :param action_index: an index for the list of allowed actions
        :return: a new state, reward and termination bool (if time_step = 96)
        """

        ctr_risk, ctr_estimations, winning_bids, clicks = self.get_camp_data()

        winning_bids = winning_bids.astype(int)

        self.click = 0
        self.cost = 0
        self.ctr_value = 0
        self.winning_rate = 0
        self.impressions = 0

        state = []
        risk_tendency_path = []

        for i in range(len(ctr_estimations)):
            state_instant = [self.n_regulations, self.budget]
            if ifintial:
                rt = risk_tendency_table[self.n_regulations][self.budget]
            else:
                rt = q_network.predict_batch(sess, state_instant)
            bids = self.bid(self.n_regulations, self.budget, ctr_estimations[i] + rt * ctr_risk[i], max_bid)
            bids = min(int(bids), min(self.budget, max_bid))

            self.state = [self.n_regulations, self.budget]
            state.append(self.state)
            risk_tendency_path.append(rt)

            self.n_regulations -= 1
            self.time_step += 1

            if bids > winning_bids[i]:
                self.budget -= winning_bids[i]
                self.impressions += 1
                self.cost += winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / min(self.episode_length, self.data_count)



            if self.time_step == self.episode_length or self.data_count == 0:
                self.termination = True
                break


        return np.array(state), np.array(risk_tendency_path), self.ctr_value

    def run(self, auction_in, risk_tendency, N, init_budget, max_bid, clk_stat_interval, ifconst_risk=False):
        auction = 0
        imp = 0
        clk = 0
        cost = 0
        clk_stat = np.zeros([np.ceil(max_bid / clk_stat_interval).astype(int)])


        B = init_budget

        episode = 1
        n = N
        b = B
        risk_avg = np.mean(np.array(auction_in['data'][:, 3]).astype(float))
        # for line in auction_in:
        # 	if input_type == "file reader":
        # 		line = line[:len(line) - 1].split(delimiter)
        # 		click = int(line[0])
        # 		price = int(line[1])
        # 		theta = float(line[2])
        # 	else:
        # 		(click, price, theta) = line
        for line in range(np.array(auction_in['data']).shape[0]):

            click = int(auction_in['data'][line, 0])
            price = int(auction_in['data'][line, 1])
            theta = float(auction_in['data'][line, 2])
            if not ifconst_risk:
                risk = float(auction_in['data'][line, 3])
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

    def result(self):
        """
        This function returns some statistics from the episode or test
        :return: number of impressions won, number of
        actual clicks, winning rate, effective cost per click,
        and effective cost per impression.
        """
        if self.click == 0:
            self.result_dict['eCPC'] = 0
        else:
            self.eCPC = self.cost / self.click
        if self.impressions == 0:
            self.result_dict['eCPI'] = 0
        else:
            self.eCPI = self.cost / self.impressions

        return self.impressions, self.click, self.cost, \
               self.winning_rate, self.eCPC, self.eCPI

    def load_data(self, camp_dict):
        self.camp_dict = camp_dict
        self.data_count = camp_dict['imp']

        self.Lambda = 1
        self.time_step = 0
        self.n_regulations = 0
        self.result_dict['impressions'] = 0
        self.result_dict['click'] = 0
        self.result_dict['cost'] = 0
        self.result_dict['win-rate'] = 0
        self.result_dict['eCPC'] = 0
        self.result_dict['eCPI'] = 0
        self.termination = True


def get_data(camp_n):
    """
    This function extracts data for certain specified campaigns
    from a folder in the current working directory.
    :param camp_n: a list of campaign names
    :return: two dictionaries, one for training and one for testing,
    with data on budget, bids, number of auctions, etc. The different
    campaigns are stored in the dictionaries with their respective names.
    """
    if type(camp_n) != str:
        train_file_dict = {}
        test_file_dict = {}
        data_path = os.path.join(os.getcwd(), 'iPinYou_data')

        for camp in camp_n:
            test_data = pd.read_csv(data_path + '/' + 'test.theta_' + camp + '.txt',
                                    header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
            train_data = pd.read_csv(data_path + '/' + 'train.theta_' + camp + '.txt',
                                     header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
            camp_info = pickle.load(open(data_path + '/' + 'info_' + camp + '.txt', "rb"))
            test_budget = camp_info['cost_test']
            train_budget = camp_info['cost_train']
            test_imp = camp_info['imp_test']
            train_imp = camp_info['imp_train']

            train = {'imp': train_imp, 'budget': train_budget, 'data': train_data}
            test = {'imp': test_imp, 'budget': test_budget, 'data': test_data}

            train_file_dict[camp] = train
            test_file_dict[camp] = test
    else:
        data_path = os.path.join(os.getcwd(), 'data/ipinyou-data')
        test_data = pd.read_csv(data_path + '/' + camp_n + '/' + 'test.theta.txt',
                                header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        train_data = pd.read_csv(data_path + '/' + camp_n + '/' + 'train.theta.txt',
                                 header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        camp_info = pickle.load(open(data_path + '/' + camp_n + '/' + 'info.txt', "rb"))
        test_budget = camp_info['cost_test']
        train_budget = camp_info['cost_train']
        test_imp = camp_info['imp_test']
        train_imp = camp_info['imp_train']

        train_file_dict = {'imp': train_imp, 'budget': train_budget, 'data': train_data}
        test_file_dict = {'imp': test_imp, 'budget': test_budget, 'data': test_data}

    return train_file_dict, test_file_dict
