# This script creates particle images and calculates v2 from hdf files using 2-particle correlations from Q vectors
# Based on https://arxiv.org/pdf/1010.0233.pdf
# Images are created for each sample
# Flow coefficients for each event are calculated using all samples
# For simplicity assume there is uniform acceptance in the detector
# TODO: add comments to functions

import h5py
import numpy as np
import matplotlib.pyplot as plt


parts_dtype = [
	('sample', int),
	('ID', int),
	('charge', int),
	('pT', float),
	('ET', float),
	('mT', float),
	('phi', float),
	('y', float),
	('eta', float)
]

def QVector(phis, n):
    return np.sum(np.exp(n*1j*phis))
# by eq (16)


def SingleEvtAvgTwoParticleCorr(Qns, Ms):
    return (np.abs(Qns)*np.abs(Qns) - Ms)


def make_image_sample(sample, energy=5.02):
    """
    makes image for samples

    Args:
    sample (numpy array):a list of particles's data of the sample

    Returns:
    
    numpy array:array of image for sample
    """
    histoE, xedgesE, yedgesE = np.histogram2d(sample['eta'], sample['phi'], weights=np.log(energy)*np.ones(len(sample)), bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histomT, xedgesmT, yedgesmT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['mT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histopT, xedgespT, yedgespT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['pT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    # unweighted
    histo, xedges, yedges = np.histogram2d(sample['eta'], sample['phi'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])

    return np.array([np.array([histoE, histomT, histopT, histo])])


def samples_calculations(particles, N_samples, Q2s,Q3s, Ms, images):
    """
    Calculates and rrturns multiplicities, Q vectors of samples and makes images of samples for given event
    first, for a given event, selects appropriate particles by looping through all samples in it. Then, calculates multiplicity, Q vectors
    and makes image for that sample, and adds those values to corresponding arrays

    args:

    particles (numpy array):a list of particles's datas per event
    N_samples(int): the number of particles in each event (=10) 
    Q2, Q3s (numpy array):an array of elliptic and triangular Q vectors respectively, per sample
    Ms(numpy array): number of particles per sample
    images(numpy array):image array for sample
    
    returns:

    Ms (numpy array)
    Q2s, Q3s (numpy array)
    images (numpy array)

    """
    
    #select suitable particles from samples
    for n in range (N_samples): #loop through samples
      sample=[] 
      for particle in particles: #loop through all particles in a given sample
            if particle['sample']!=n+1 or particle['charge']==0 or np.abs(particle['eta']) >= 0.8: continue; #check the suitability of particle 
            sample.append(particle)
      sample = np.array(sample) 

      #calculate multiplicity and Q vectors of sample, and add values to the end of their arrays
      Ms=np.append(Ms, len(sample)) 
      Q2s = np.append(Q2s, QVector(sample[:]['phi'], 2))
      Q3s = np.append(Q3s, QVector(sample[:]['phi'], 3))

      #make image of given sample
      sample_images=make_image_sample(sample)
      images=np.append(images, sample_images, axis=0)


    return [ Ms, Q2s,  Q3s, images]


 
def calculate_vn_per_event( Ms,Q2s, Q3s,v2s,v3s, N_samples=10):
            """
            calculates and returns flow coefficients for given event

            args:
            Ms (numpy array):numbers of particles per samples of event
            Q2s, Q3s(numpy array):Q vectors of elliptic and triangular flows respectively of samples per event 
            v2s(numpy array): an array of elliptic flow coefficients of previous events
            v3s(numpy array):an array of triangular coefficients of previous events
            N_sample(int): the number of particles in each event (=10)


            returns:
            v2s(numpy array):elliptic flow coefficient of given event
            v3s(numpy array): triangular flow coefficient of given event
            skipped(boolean):  T/F if event is suitable or not
            
            """
            skipped = False 
            weights = Ms*Ms-Ms #multiplicities 
            sum_of_weights = sum(weights)
            single_event_avgs_2 = SingleEvtAvgTwoParticleCorr(Q2s, Ms)
            single_event_avgs_3 = SingleEvtAvgTwoParticleCorr(Q3s, Ms)
           
            if sum(single_event_avgs_3) > 0 and sum(single_event_avgs_2) > 0: # only positive numbers for square root
                v2s = np.append(v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights)*np.ones(N_samples)) 
                v3s = np.append(v3s, np.sqrt(sum(single_event_avgs_3)/sum_of_weights)*np.ones(N_samples))
               # Ms_event = np.append(Ms_event, np.sum(Ms)*np.ones(N_samples)) # charged particle multiplicity per event
            else:
                skipped = True
               
            return [v2s,v3s, skipped]


def  main():

   #define initial parameters
    fn="particles_PbPb_50evt.hdf" 
    energy=5.02
    v2s = np.array([])
    v3s = np.array([])
    images = np.empty((0,4,32,32))
    N_samples=10   


    with h5py.File(fn,"r") as f:

        for evt in f.values(): #loop through all events
           
            Ms=np.array([]) # multiplicities of particles for all samples in this event
            Q2s = np.array([])  #Q vectors for elliptic flow for all samples in this event
            Q3s = np.array([])  #Q vectors for triangular flow for all samples in this event

            particles = np.sort(np.array(evt[:], dtype=parts_dtype), order=['sample','charge'])#sort all particles tuples according to sample number for givent event
    
            Ms, Q2s,  Q3s, images=samples_calculations(particles, N_samples,Q2s,Q3s, Ms,images)#calculate multiplicity of particles, Q vectors and make images of samples

            v2s,v3s, skipped=calculate_vn_per_event(Ms,Q2s,Q3s,v2s,v3s) # use Ms and Q vectors to calculate elliptic and triangular flow coefficients for event
      
            if skipped==True: #check if the event is suitable             
                images = images[:-N_samples] # throw away images of last event

    #create .npz files of images
    Ms_image=np.sum(images,axis=(1,2,3))
    flowdata = np.stack((v2s, v3s, Ms_image), axis=-1)
    #TODO if necessary remove nans and associated images by calling removeNans from removeNans.py 
    #flowdata = removeNans(flowdata)
    np.savez_compressed('{}.npz'.format(fn),images=images,flow_data = flowdata)

 

if __name__=="__main__":
     main()



