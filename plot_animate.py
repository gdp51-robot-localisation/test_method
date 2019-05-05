import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import csv
import os, os.path


filename = 'pts_SURF_20190429-150831_5diamond_worked_pid.csv'
animation_outpath = "/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01"

file_name_format = filename.split('.')
filename_noext = file_name_format[0]



data = []
with open(filename) as csvfile:
    csvread = csv.reader(csvfile)
    for row in csvread:
        data.extend(row)
    data = np.array(data)
    data = data.astype(np.float)
    data = data.reshape(-1,5)
    time_data = data[:,0]
    x_data = data[:,1]
    y_data = data[:,2]
    print('Animation length (s):', max(time_data))
    print('Number of measurements:',len(x_data))

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2000), ylim=(0, 1500))
plt.xlabel('Width (mm)')
plt.ylabel('Height (mm)')
plt.title('Position')
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = x_data
    y = y_data
    line.set_data(x[:i], y[:i])
    return line,

interval_time = (max(time_data))*1000 / len(x_data)
frame_count = len(x_data)
fps = frame_count / max(time_data)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='GDP51'), bitrate=1800)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
           frames=frame_count, interval=interval_time, blit=True)

# plt.show()
output = os.path.join(animation_outpath, filename_noext + ".mp4")

anim.save(output, writer=writer)




