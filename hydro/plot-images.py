# script to plot images created by v2fromHDF.py
import numpy as np
import matplotlib.pyplot as plt
import sys




def plot_images(imgName="particles_PbPb_50evt.hdf.npz", outname='images-out/', energy = 'default'):
    energies = {'default' : 5.02, 'JETSCAPE' : 8.521}
    energy  = energies[energy]
    event_n = 1
    sample_n = 1
    j = (event_n-1)*10+sample_n-1 # 10 images per event, so event_n=1 images are at j = 0...9, etc. 
    compressed = np.load(imgName)

    for i in range(2):
        fig, axes = plt.subplots(2,3, figsize=(12,8))
        for (iax, jax), ax in np.ndenumerate(axes):

            im = ax.imshow(np.log(compressed["images"][j][0]), interpolation = 'none')#, extent = (-0.8,0.8,-np.pi,np.pi))
            plt.colorbar(im, ax=ax)
        
            ax.set_ylabel('$\\eta$')
            ax.set_yticks([0,7,15,23,31], np.round(np.linspace(-0.8, 0.8, 5), 2))
            ax.set_xlabel('$\\phi$')
            ax.set_xticks([0,7,15,23,31], np.round(np.linspace(-np.pi, np.pi, 5), 2))
            ax.text(1, -1, 'N_ch = {}'.format(np.sum(compressed['images'][j][0])/np.log(energy)))
            j += 1
    	    #ax.set(title='{} jet: $p_T=${:.0f} GeV'.format(['QCD','top'][iax], [jetpep0,jetpep][iax][idx][0][0]))
        
        plt.subplots_adjust(wspace=0.5,hspace=0.2)
        fig.suptitle("JETSCAPE-noPGun-{}".format(i))
    
        #axes[0].set_title("sample {}".format(sample_n))
        #axes[1].set_title("sample {}".format(sample_n+1))
        #axes[2].set_title("sample {}".format(sample_n+2))
        #plt.show()
        
        filename = outname+"{}.png".format(i)
        plt.savefig(filename)
        plt.close(fig)
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path=sys.argv[1]
        outname = sys.argv[2]
    plot_images(image_path, outname)
