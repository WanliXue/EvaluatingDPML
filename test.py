# import matplotlib
# matplotlib.use('TkAgg')  #for mac using matplotlib
# # This should be done before `import matplotlib.pyplot`
# # 'Qt4Agg' for PyQt4 or PySide, 'Qt5Agg' for PyQt5
# import matplotlib.pyplot as plt
# import numpy as np
#
# print ('try')
#
# t = np.linspace(0, 20, 500)
# plt.plot(t, np.sin(t))
# plt.show()
import numpy as np

train_y=[1,1,0,2]
print(len(np.unique(train_y)))
