# author: pyjorunk
# This script creates particle images and calculates v2 from hdf files using 2-particle correlations from Q vectors
# Calculating with Q-vectors avoids having to consider each particle pair 
# This technique is insensitive to non-flow effects and interference between different harmonics
# Based on https://arxiv.org/pdf/1010.0233.pdf

# Images are created for each sample
# Flow coefficients for each event are calculated using all samples
# For simplicity assume there is uniform acceptance in the detector


import sys
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

# by eq (4)
def QVector(phis, n):
    return sum(np.exp(n*1j*phis))
# by eq (16)
def SingleEvtAvgTwoParticleCorr(Qns, Ms):
    return (np.abs(Qns)*np.abs(Qns) - Ms)#/(Ms*Ms-Ms)

# TODO edit this to make images we want
def make_image_sample(sample): #tree = uproot3.open(fn)['vTree'] #df = tree.pandas.df();
    histo, xedges, yedges = np.histogram2d(sample['eta'], sample['phi'], bins=(32,32))#,weights=myevent['pt'])
    return np.array([histo])#,np.array(flowprop)

v2s = np.array([])
v3s = np.array([])
skipped_v2s = np.array([])
skipped = 0

images = []


print(sys.argv[2:]) # when running, give as arguments the files with data
#for fn in sys.argv[1:]:
fn = "particles_PbPb_50evt.hdf" # for testing just use this one file
for x in [1]:
    with h5py.File(fn,"r") as f:
        event_n = 1
        for evt in f.values():
            Ms = np.array([])
            Q2s = np.array([])
            Q3s = np.array([])
            # make list of particles in event
            particles = np.sort(np.array(evt[:], dtype=parts_dtype), order='sample')
            # loop over samples in each event
            sample_start = 0
            sample_n = particles[0]['sample']
            current_particle = 0
            particle_n = 0
            # count particles in current sample
            for particle in particles:
                if particles[current_particle]['sample'] == sample_n:
                    current_particle += 1
                # compute sample weights and Q-vectors
                else:
                    particle_n = current_particle-sample_start
                    sample = particles[sample_start:current_particle]
                    Ms = np.append(Ms, particle_n)
                    Q2s = np.append(Q2s, QVector(sample['phi'], 2))
                    Q3s = np.append(Q3s, QVector(sample['phi'], 3))

                    images.append(make_image_sample(sample))
                    #print('finished sample {} of event {}'.format(sample_n, event_n))
                    sample_n += 1
                    # move to the start of next sample
                    sample_start += particle_n
                    particle_n = 0

            # calculate v2 v3 per event using all of its samples
            # weights from eq (9) of source paper
            weights = Ms*Ms-Ms
            sum_of_weights = sum(weights)
            single_event_avgs_2 = SingleEvtAvgTwoParticleCorr(Q2s, Ms)
            single_event_avgs_3 = SingleEvtAvgTwoParticleCorr(Q3s, Ms)
            # weighting not needed with skipping the division by Ms*Ms-Ms in the function above
            #weighted_sum_2 = np.dot(weights, single_event_avgs_2)
            #weighted_sum_3 = np.dot(weights, single_event_avgs_3)
            #v2s = np.append(v2s, np.sqrt(weighted_sum_2/sum_of_weights))
            #v3s = np.append(v3s, np.sqrt(weighted_sum_3/sum_of_weights)) # for some events this is sqrt of a negative number

            # only positive flow coefficients usable
            if sum(single_event_avgs_3) > 0:
                v2s = np.append(v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights))
                v3s = np.append(v3s, np.sqrt(sum(single_event_avgs_3)/sum_of_weights))
            else:
                skipped_v2s = np.append(skipped_v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights))
                skipped += 1

            print("finished event {}".format(event_n))
            event_n += 1

# TODO test with toymcflow particles to see if it gives accurate v2 v3
print('skipped {} events with v2s = '.format(skipped), np.round(skipped_v2s,4))
print('v2s =', np.round(v2s, 4))
print('v3s = ', np.round(v3s, 4))

vns = np.stack((v2s, v3s), axis=-1)
np.savez_compressed('images_{}.npz'.format(fn),images=images,flow_coefs = vns)
