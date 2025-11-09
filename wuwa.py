import random
import math
import numpy
# import rich
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
import matplotlib.pyplot as plt

history = {
	# 角色 : [限定抽取數]
	"忌炎" : [35716, 15037],
	"吟霖": [57576, 18506],
	"今汐" : [83070, 44162],
	"長離" : [93781, 33415],
	"折枝" : [46815, 37159],
	"相里要" : [6226, 5667],
	"守岸人" : [123440, 87622],
	"椿" : [159638, 40611],
	"柯萊塔" : [178388],
	"洛可可" : [43348, 11288],
	"菲比" : [158474],
	"布蘭特" : [59502, 13951],
	"坎特雷拉" : [113522, 20455],
	"贊妮" : [197468],
	"夏空" : [101000, 19542],
	"卡提希婭": [278697],
	"露帕" : [81429, 8012],
	"弗洛洛" : [114966],
	"奧古斯塔" : [107831],
	"尤諾" : [98104],
	"嘉貝莉娜" : [49332],
	"仇遠" : []
}

# starts at the 64th pull; ends at the 78th pull
pulls = numpy.array([2005, 1947, 11400, 19589, 26138, 29894, 30658, 33520, 30576, 23352, 15274, 8176, 3846, 1294, 283])
probabilities = pulls / pulls.sum()
record = numpy.zeros_like(pulls)
record_last = [884337, 859968, 5028855, 8644537, 11532665, 13197587, 13529155, 14790799, 13488083, 10299967, 6744969, 3607573, 1696805, 571933, 124753]

def pull():
	# calculate single pull
	shift = numpy.random.choice(len(pulls), p=probabilities)
	record[shift] += 1
	return shift + 64

def wish():
	# calculate total pulls of C6 character
	guarantee = False
	total_pulls = 0
	up = 0
	while up < 7:
		if guarantee:
			total_pulls += pull()
			up += 1
			guarantee = False
		else:
			is_up = numpy.random.choice([True, False], p = [0.5, 0.5])
			guarantee = not is_up
			total_pulls += pull()
			if is_up:
				up += 1
	return total_pulls

def main():
	batch = 1000  # number of pulls per batch
	batch_size = 10000  # number of batches
	result = numpy.ndarray((batch_size, 2))

	console = Console()
	with Live(console=console, refresh_per_second=4) as live:
		for i in range(batch_size):
			data = numpy.array([wish() for _ in range(batch)])
			std = data.std()
			mean = data.mean()
			result[i] = (mean, std)

			body = f"Mean: {mean:.2f}\nStandard Deviation: {std:.3f}"
			panel = Panel(body, title=f"Batch {i+1}/{batch_size}", expand=False)
			live.update(panel)

	standard_deviation = numpy.mean(result[:, 1])
	mean = numpy.mean(result[:, 0])
	print(f"Mean: {mean:.2f}, Standard Deviation: {standard_deviation:.3f}")
	console.print(record, style="white bold")

	# 抽中五星角色所花費的抽數（64 到 78）
	x = numpy.arange(64, 79)

	plt.figure(figsize=(10, 6))
	bars = plt.bar(x, record, color="#6fa8dc", edgecolor="black")

	plt.title("record", fontsize=16)
	plt.xlabel("pulls", fontsize=12)
	plt.ylabel("times", fontsize=12)

	plt.grid(axis="y", linestyle="--", alpha=0.5)

	plt.tight_layout()
	plt.savefig("record.png")
	plt.show()

if __name__ == "__main__":
	main()
