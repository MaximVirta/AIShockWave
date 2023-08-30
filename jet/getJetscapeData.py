import numpy as np
import sys
#from particle import Particle


def getJetscapeData(jetscape_hadrons):

    hadrons_in = np.loadtxt(jetscape_hadrons)
    # output as a numpy array first dimension has events, second dimension particles, third dimension coordinates (pT,mass,eta,phi)
    hadrons_out = np.empty((0,9,4))

    # get values as arrays, one line contains 0:PSN 1:PID 2:?? 3:E 4:px 5:py 6:pz 7:eta 8:phi
    #Energy = hadrons_in[:,3]
    px = hadrons_in[:,4]
    py = hadrons_in[:,5]
    #pz = hadrons_in[:,6]
    
    eventNum = getEventNum(hadrons_in) # works

    pT = getPT(px, py)
    #mass = getMass(Energy,px,py,pz, hadrons_in)
    #pdg = Particle.from_pdgid(hadrons_in[:,1])
    #print(pdg)
    eta = hadrons_in[:,7]
    phi = hadrons_in[:,8]
    #print('phi {}'.format(phi))
    #print(np.nonzero(phi<0)[0])
    
    # couldn't get mass
    hadrons_out = np.array([np.column_stack((eventNum, pT, eta, phi))])[0]
    print('hadrons shape {}'.format(hadrons_out.shape))
    np.save('jetscape_particles', hadrons_out)
    makeImage(hadrons_out)



# based on https://stackoverflow.com/questions/34027288/cumulative-counts-in-numpy-without-iteration
# zeroth element in each row is "PSN" and PSN = 0 means start of new event
def getEventNum(hadrons):
    eventNum = (hadrons[:,0] == 0).cumsum()
    return eventNum

# calculate pT from px py
def getPT(px, py):
    return np.sqrt(px*px + py*py)

# calculate mass from E px py pz
# E² = m²c⁴ + p²c² -> m² = E² - p²
def getMass(E, px, py, pz, hadrons_in):
    mass2 = E*E - px*px - py*py - pz*pz 
    if np.any(m2 < 0 for m2 in mass2):
        photons = np.nonzero(hadrons_in[:,1] == 22)[0]
        #print(photons)
        print(mass2[photons])
        sys.exit()
        indices = np.nonzero(mass2 < 0)[0] # indices where mass² < 0
        print('negative mass² count {}'.format(np.shape(indices)))
        print('total # of particles {}'.format(np.shape(hadrons_in)))
        print('negative m²s {}'.format(mass2[indices]))
        print('(hadrons_in[np.where(mass2 < 0)][1] {}'.format(hadrons_in[indices][0]))
        print('most neg mass2 {}'.format(np.min(mass2[indices])))
        negmassind = np.argmin(mass2[indices])
        print(negmassind)
        print(hadrons_in[negmassind])
        print(E[negmassind], px[negmassind], py[negmassind], pz[negmassind])
        print(E[negmassind]*E[negmassind] - px[negmassind]*px[negmassind]- py[negmassind]*py[negmassind] -pz[negmassind]*pz[negmassind])
        sys.exit()
        return E
    else:
        return np.sqrt(mass2)

def makeImage(particles):
    # particles is an array of shape (N_particles, 4), second axis contains n_event, pT, eta, phi
    print(particles)
    print(particles[-1,0])
    N_events = int(particles[-1,0]) #event number of last particle
    # to make numpy concatenate work, create the first event seperately
    sample = particles[np.nonzero(particles[:,0] == 1)[0]]
    print(len(sample[np.nonzero(np.abs(sample[:,2]) <= 0.8)]))
    eta = sample[:,2]
    phi = sample[:,3]
    histoE, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=np.log(5.02)*np.ones(sample.shape[0]))
    histoPt, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=sample[:,1])
    histos = np.array([[histoE, histoPt]])
    for i in range(2, N_events+1):
        sample = particles[np.nonzero(particles[:,0] == i)[0]]
        print(len(sample[np.nonzero(np.abs(sample[:,2]) <= 0.8)]))
        if i > 10:
            sys.exit()
        eta = sample[:,2]
        phi = sample[:,3]
        histoE, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=np.log(5.02)*np.ones(sample.shape[0]))
        histoPt, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=sample[:,1])
        temp_histos = np.array([[histoE, histoPt]])
        histos = np.concatenate((histos, temp_histos), axis = 0)
    np.save('jetscape_images2', histos)


if __name__=='__main__':
    getJetscapeData('hadron_list_1.dat')
