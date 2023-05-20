#!/usr/bin/python
import config
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk_tendency(risk_attendency, path):
	fig = plt.figure(figsize=(10, 7))
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	sns.set(font_scale=1.5)
	ax = sns.heatmap(risk_attendency, cmap='rainbow', xticklabels=1000, yticklabels=500)
	ax.invert_yaxis()
	plt.xlabel("Budget", fontsize=25)
	plt.ylabel("Remaining Time Steps", fontsize=25)
	plt.savefig(path)
	plt.close(fig)

def plot_click(click, winrate, cpm, ecpc, c0_vec, path):
	fig1 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, click, color="r", linestyle="-", marker="^", linewidth=1)
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("Total Clicks")
	plt.legend(loc='lower right')
	plt.savefig(path + "ss_click.png")
	plt.close(fig1)

	fig2 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, winrate, color="r", linestyle="-", marker="^", linewidth=1)
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("Win rate")
	plt.legend(loc='lower right')
	plt.savefig(path + "ss_win_rate.png")
	plt.close(fig2)

	fig3 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, cpm, color="r", linestyle="-", marker="^", linewidth=1)
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("CPM")
	plt.legend(loc='lower right')
	plt.savefig(path + "ss_cpm.png")
	plt.close(fig3)

	fig4 = plt.figure(figsize=(6.4, 4.8))
	plt.plot(c0_vec, ecpc, color="r", linestyle="-", marker="^", linewidth=1)
	plt.grid(color="k", linestyle=":")
	plt.xlabel("Budget parameter $c_0$")
	plt.ylabel("eCPC")
	plt.legend(loc='lower right')
	plt.savefig(path + "ss_eCPC.png")
	plt.close(fig4)