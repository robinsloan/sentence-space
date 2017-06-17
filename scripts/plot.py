import sys
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

session = sys.argv[1]

if session == None:
  print("You must specify a session!")
  quit()

folder = "session/%s" % session

if not os.path.exists(folder):
  print("There doesn't seem to be a session by that name...")
  quit()

data = pd.read_csv("%s/log.train.csv" % folder, header=None, usecols=[0,1,2,3,4])
data_0 = data[0].iloc[::10]
data_1 = data[1].iloc[::10] # every 100th point
data_2 = data[2].iloc[::10]
data_3 = data[3].iloc[::10].multiply(-1.0)
data_4 = data[4].iloc[::10]

plt.plot(data_0, data_1, 'r-', label='cost')
plt.plot(data_0, data_2, 'c-', label='reconstruction loss')
plt.plot(data_0, data_3, 'm--', label='kld')
plt.plot(data_0, data_4, 'g-', label='aux loss')

plt.legend(loc='lower right')
plt.ylim([0,3000])

plt.savefig("%s/%s_cost_graph.pdf" % (folder, session), format='pdf')
