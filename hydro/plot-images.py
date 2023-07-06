# script to plot images created by v2fromHDF.py
import numpy as np
import matplotlib.pyplot as plt

images = np.load('images.npy')

j = 0
for i in range(3):
    fig, axes = plt.subplots(1,3, figsize=(12,3))#(1,2, figsize=(12,3))
    for iax, ax in enumerate(axes):
        im = ax.imshow(images[j][0], interpolation = 'none') #extent = [-0.8, 0.8, 0, 2*np.pi], 
        plt.colorbar(im, ax=ax)
        plt.xticks(np.linspace(-2,2, 32))
        plt.yticks(np.linspace(0,2*np.pi,32))
        j +=1
		#ax.set(title='{} jet: $p_T=${:.0f} GeV'.format(['QCD','top'][iax], [jetpep0,jetpep][iax][idx][0][0]))
    fig.suptitle("Images for event {}".format(1))
    axes[0].set_title("sample {}".format(j-2))
    axes[1].set_title("sample {}".format(j-1))
    axes[2].set_title("sample {}".format(j))
#plt.show()
    filename = "images-sample{}-to-{}.png".format(j-2, j)
    plt.savefig(filename)
    plt.close(fig)
