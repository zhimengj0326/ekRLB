import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def improve(baseline, compare):
    result= []
    baseline = np.array(baseline)
    for j in range(len(compare)):
        improvement = (np.array(compare[j]) - baseline) / baseline
        result.append(improvement)
    return result

def to_percent(temp, position):
    return '%.0f'%(100*temp) + '%'


c0_vec = [1/32, 1/16, 1/8, 1/4, 1/2]
Lin = [101.2, 107.9, 119.2, 142.8, 185.2]
RLB = [106.0, 120.8, 142.4, 179.3, 241.7]
RRLB = [106.6, 121.5, 143.2, 181.7, 244.2]
CRTRLB = [108.6, 122.4, 144.1, 178.7, 238.9]
CRRLB = [91.6, 103.6, 125.2, 164.6, 242.9]
ssRLB = [86.9, 98.2, 123.7, 157.5, 222.8]

fig1 = plt.figure(1)
plt.figure(figsize=(24, 8.4))
ax1 = plt.subplot(121)
fontsize = 35
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

res1 = improve(Lin, [Lin, RLB, RRLB])
#ax1.figure(figsize=(6.4, 4.8))
plt.plot(c0_vec, res1[0], color="r", linestyle="--", marker="*", linewidth=1, label="Lin")
plt.plot(c0_vec, res1[1], color="g", linestyle="-", marker="^", linewidth=1, label="RLB")
plt.plot(c0_vec, res1[2], color="b", linestyle="-", marker="d", linewidth=1, label="ekRLB")
plt.grid(color="k", linestyle=":")
plt.xlabel("Budget parameter $c_0$", fontsize=fontsize)
plt.ylabel("Clicks Improvement over Lin", fontsize=25)
plt.legend(loc='upper left', fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

ax2 = plt.subplot(122)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
res2 = improve(RRLB, [RRLB, CRTRLB, CRRLB, ssRLB])
plt.plot(c0_vec, res2[0], color="r", linestyle="--", marker="*", linewidth=1, label="ekRLB")
plt.plot(c0_vec, res2[1], color="g", linestyle="-", marker="^", linewidth=1, label="CRTRLB")
plt.plot(c0_vec, res2[2], color="b", linestyle="-", marker="d", linewidth=1, label="CRRLB")
plt.plot(c0_vec, res2[3], color="m", linestyle="--", marker="s", linewidth=1, label="ssRLB")
plt.grid(color="k", linestyle=":")
plt.xlabel("Budget parameter $c_0$", fontsize=fontsize)
plt.ylabel("Clicks Improvement over RRLB", fontsize=25)
plt.legend(loc='lower center', fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.savefig("ipin_click.pdf")
plt.close(fig1)



# ssRTB_results_file = "results/ssRTB_results.pickle"
# ssRTB_results = pickle.load(open(ssRTB_results_file, "rb"))
#
# click_tot = ssRTB_results['click']
#
# ssRTB_avg_results_file = "results/ssRTB_avg_results.pickle"
# ssRTB_avg_results = pickle.load(open(ssRTB_results_file, "rb"))
#
# click_avg = ssRTB_avg_results['click']
# winrate_avg = ssRTB_avg_results['win_rate']
# cpm_avg = ssRTB_avg_results['cpm']


print("click")
