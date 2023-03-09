from __future__ import division

import os
import sys
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import math

from network import Model
from enum import Enum

runType = Enum('runType', ['Full', 'DSManifPert', 'SSManifPert','ControlManifPert'])


# Set default configuration parameters
def get_defaultconfig():
    config = {'rnn_type'    : 'LeakyRNN',  
              'activation'  : 'softplus',    # relu, softplus, tanh, elu
              'tau'         : 100,           # ms
              'tau_noise'   : 2,             # ms
              'dt'          : 1,             # discretization time step
              'sigma_rec'   : 0.05,          # noise scale
              'w_rec_init'  : 'randortho',   # diag, randortho, randgauss
              'l2_h'        : 0.0,           # firing rate / homeostatic regularizer weight
              'l2_wR'        : 0.0,          # recurrent weight regularizer weight
              'l2_wI'        : 0.0,          # input weight regularizer weight
              'l2_wO'        : 0.0,          # output regulaizer weight
              'seed'        : 0,             # Seed for network instance
              'rng'         : None,
              'save_name'   : 'test',
              'init_lr_full': 0.0001,        # Learning Rate
              'batch_size': 10,              # No. of trials to save at a time
              'training_iters' : 10000000,   # Max. number of trials to run
              'SAVE_PARAMS' : True           # Whether to save model parameters after each problem
    }
    config['alpha'] = np.float32(1.0 * config['dt'] / config['tau'])               # Discretization - network
    config['alpha_noise'] = np.float32(1.0 * config['dt'] / config['tau_noise'])   # Discretization - noise

    return config


# save weights and biases
def testAndSaveParams(sess, config, model, images, taskIndex, suff=''):
    if suff != '':
        suff = '_'+ suff

    # Save images
    dat = dict()
    dat['images'] = images

    # Save weights
    wts, wtNames = model.getWeights(sess)
    for wt, nm in zip(wts, wtNames):
        dat['wts_'+nm] = wt

    # Write to file
    sio.savemat(os.path.join('data', 'saved_' + config['save_name'] + '_' + str(taskIndex) +suff+ '.mat'),dat)


# core function for manifold perturbations
def resetOutWeightsWithSameStruct(sess, model, config, k, rType, eigs = None, eigsS = None, states = None):
    outWts = model.getOutWeights(sess)

    if eigs is None:
        states1 = states[:,:,0:50]
        states2 = states[:,:,50:100]

        # Find decision and stimulus subspaces
        states1Mn = np.mean(states1, axis = 2, keepdims = False)
        states2Mn = np.mean(states2, axis = 2, keepdims = False)

        # print(str(states1Mn.shape) + ' ' + str(states2Mn.shape))
        x = np.concatenate((states1Mn, states2Mn), axis=1)
        # print(str(x.shape))
        U,S,Vh = np.linalg.svd(x, full_matrices=False)
        eigs = np.real(U[:,0:k])

        states1D = np.matmul(np.matmul(eigs, eigs.T), np.reshape(states1,(config['num_rnn'],-1)))
        states2D = np.matmul(np.matmul(eigs, eigs.T), np.reshape(states2,(config['num_rnn'],-1)))
        states1S = np.reshape(states1,(config['num_rnn'],-1))-states1D
        states2S = np.reshape(states2,(config['num_rnn'],-1))-states2D
        xS = np.concatenate((states1S, states2S), axis=1)
        US,SS,VhS = np.linalg.svd(xS, full_matrices=False)
        eigsS = np.real(US)
        
    # Swap out first few PCs of source subspace from output weights
    # with first few PCs of target subspace
    newOutWts = np.copy(outWts)
    newOrder = config['rng'].permutation(np.arange(k))
    for i in range(0,k):
        if rType == runType.SSManifPert:
            # S -> S mainfold perturbation
            oldVec = eigsS[:,i]
            newVec = eigsS[:,newOrder[i]]
            ov = np.dot(outWts.T, oldVec)
            newOutWts = newOutWts - np.outer(oldVec, ov) + np.outer(newVec, ov)

        elif rType == runType.DSManifPert:
            # D -> S mainfold perturbation
            oldVec = eigs[:,i]
            newVec = eigsS[:,newOrder[i]]
            ov = np.dot(outWts.T, oldVec)
            newOutWts = newOutWts - np.outer(oldVec, ov) + np.outer(newVec, ov)

        elif rType == runType.ControlManifPert:
            # Control with frozen weights only
            continue
        
        else:
            raise NotImplementedError()

    model.setOutWeights(sess, newOutWts) # update new output weights
    return outWts, eigs, eigsS

# Simulate and save learned trajectories -
# to infer decision and stimulus subspaces,
# and subsequently perform manifold perturbations
def getStates(sess, config, model, images):

    st0 = dict()
    
    for stim in range(config['num_rnn_out']-1):
        _, trials,_ = generateData(config, images, test=True, stim=[stim])
    
        # Generate feed_dict
        feed_dict = {model.x: trials['x'],
                     model.y_rnn: trials['y_rnn'],
                     model.y_rnn_mask: trials['y_rnn_mask']}
    
        # Test model
        st, op, err = sess.run([model.states, model.y_hat_, model.cost_lsq_rnn], feed_dict=feed_dict)

        st0[stim] = st

    return st0
            
# Generate batch of trials
def generateData(config, images = None, test = False, stim = None):
    # Draw new sample images at random and orthonormalize
    if images is None:
        if test:
            raise Exception('No images provided during testing')
        images = config['rng'].normal(size=[config['num_rnn_out']-1] + config['image_shape']).astype(np.float32)
        for stim in range(config['num_rnn_out']-1):
            images[stim,:] = images[stim,:]/np.linalg.norm(images[stim,:])
        proj = np.dot(images[0, :], images[1, :])
        images[1, :] -= proj*images[0, :]
        for stim in range(config['num_rnn_out']-1):
            images[stim,:] = images[stim,:]/np.linalg.norm(images[stim,:])

    # Create input (x) and target (y_rnn) and output temporal mask (y_rnn_mask) matrices
    trials = dict()

    if test:
        stims = np.array(stim)
        datasetSize = 1
    else:
        stims = config['rng'].randint(config['num_rnn_out']-1, size=[config['batch_size']])
        datasetSize = config['batch_size']

    trials['x'] = images[stims,:]
    trials['x'] = np.concatenate((trials['x'], np.float32(config['fixationInput']*np.ones([datasetSize, 1]))), axis=1)

    fixationOffset = int(datasetSize*config['fixationPeriod'][1])
    trials['y_rnn'] = np.zeros((datasetSize*config['tdim'], config['num_rnn_out'])) 
    trials['y_rnn'][0:fixationOffset, config['num_rnn_out']-1] = 1.0 # Fixation
    for stCnt, stim in enumerate(stims):
        trials['y_rnn'][np.arange(fixationOffset+stCnt, datasetSize*config['tdim'], datasetSize), stim] = 1.0
    trials['y_rnn'] = trials['y_rnn'].astype(np.float32)

    tmp = np.ones([datasetSize*config['tdim']])
    tmp[np.arange(fixationOffset, fixationOffset+datasetSize*int(100/config['dt']))] = 0.0
    trials['y_rnn_mask'] = tmp.astype(bool)

    return stims, trials, images

# Generate model and train network
def train(seed          = 0,
          batchSize     = 1,
          l2            = 0.0005,
          l2_wR          = 0.001,
          l2_wI          = 0.0001,
          l2_wO          = 0.1,
          learningRateInit = 0.0001,
          svBnd = 10.0,
          rType = runType.Full,
          **kwargs):

    save_name = '{:d}_{:f}_{:f}_{:f}_{:f}_{:f}_{:f}_{:s}'.format(seed, learningRateInit, l2_wR, l2_wI, l2, l2_wO, svBnd, rType)
                                    
    # Set random seed
    rng = np.random.RandomState(seed)

    # Setup hyper-parameters
    config = get_defaultconfig()
    config['seed']          = seed
    config['batch_size']    = batchSize
    config['rng']           = rng
    config['save_name']     = save_name
    config['l2_h'] = l2
    config['l2_wR'] = l2_wR
    config['svBnd'] = svBnd
    config['l2_wI'] = l2_wI
    config['l2_wO'] = l2_wO

    config['init_lr_full'] = learningRateInit

    # Allow for additional configuration options
    for key, val in kwargs.items():
        config[key] = val

    config['image_shape'] = [10]
    config['num_input'] = np.prod(config['image_shape']) + 1 #Image + fixation stim
    config['num_rnn'] = 100
    config['num_rnn_out'] = 2 + 1 # Saccades + Fixation
    config['fixationInput'] = 1.0/np.sqrt(np.prod(config['image_shape']))


    # Trial duration parameters
    config['tdim'] = int(2000/config['dt'])
    config['stimPeriod']   = np.array([0, int(500/config['dt'])])
    config['fixationPeriod']  = np.array([0, int(1500/config['dt'])])
    config['decisionPeriod'] = np.array([int(1500/config['dt']), int(2000/config['dt'])])

    config['runType'] = rType
    config['max_tasks'] = 1001
    if config['runType'] != runType.Full:
        config['max_tasks'] = 101


    lrFull = config['init_lr_full']

    # Display configuration
    for key, val in config.items():
        print('{:20s} = '.format(key) + str(val))

    t_start = time.time()

    if config['runType'] != runType.Full: # for manifold perturbation only
        saveStates = np.zeros((config['num_rnn'], config['tdim'], 100))

    # Reset tensorflow graphs
    tf.compat.v1.reset_default_graph() 

    # Use customized session that also launches the graph
    with tf.compat.v1.Session() as sess:
        model = Model(config=config) # Generate graph
        model.initialize(sess) # Initialize graph
        model.printTrainable() # List trainable vars
        #sess.graph.finalize() # can't do this if graph is altered during training

        convCnt = []
        perf = []
        trIm = []
        wNR2 = []
        wNR = []
        wNI = []
        wNO = []
        hN = []
        wNormR2 = []
        wNormR = []
        wNormI = []
        wNormO = []
        hNorm = []
        hm = []
        HM = []
        singVals = np.zeros([config['max_tasks'],100])
        images = None
        firstConv = False
        for trial in range(config['training_iters']):
            # Generate a batch of trials
            stims, trials, images = generateData(config, images)
            trIm.extend(stims.tolist())

            # Generate feed_dict
            feed_dict = {model.x: trials['x'],
                         model.y_rnn: trials['y_rnn'],
                         model.y_rnn_mask: trials['y_rnn_mask']}

            # Run forward + backward passes
            _, c_lsq, c_reg, wnR, wnR2, wnI, wnO, hn, hMax, out, maxSingVal, topTenSings = sess.run([model.optimizer_full, model.cost_lsq_rnn, model.cost_reg_rnn, model.wNormR, model.wNormR2, model.wNormI, model.wNormO, model.hNorm, model.hMax, model.y_hat, model.maxSingVal, model.topTenSings], feed_dict=feed_dict)

            # Save trial specific learning stats
            perf.append(c_lsq)
            wNR.append(wnR)
            wNR2.append(wnR2)
            wNI.append(wnI)
            wNO.append(wnO)
            hN.append(hn)
            hm.append(hMax)

            # Print summary stats
            runTime = time.time()-t_start
            if trial%100 == 0:
                print('Trial: ' + str(trial) + ' cost: ' + str(np.mean(perf[-50:])) + ' cost_reg: ' + str(c_reg) + ' cost_lsq: ' + str(c_lsq) + ' Runtime: ' + str(runTime) + ' s')
                sys.stdout.flush()

            # Check for convergence, set converged flagW, save model
            if math.isnan(c_lsq) or len(perf) > 20000:
                taskFailed = True
            else:
                taskFailed = False

            
            # Saved trained model for new problem
            if (len(perf) > 50 and np.mean(perf[-50:]) < 0.005) or taskFailed:
                if taskFailed: # Update problem learning-specific stats when convergence fails
                    convCnt.append(np.nan)
                    wNormR.append(np.nan)
                    wNormR2.append(np.nan)
                    wNormI.append(np.nan)
                    wNormO.append(np.nan)
                    hNorm.append(np.nan)
                    HM.append(np.nan)
                else:
                    convCnt.append(len(perf)-50) # Trials to convergence for new problem

                    # Dump trained model and problem specifics to file after it is learned
                    if len(convCnt) >= 1 and config['SAVE_PARAMS'] == True:
                        testAndSaveParams(sess, config, model, images, len(convCnt))

                    # Save problem learning-specific stat summary
                    wNormR.append(np.mean(wNR[-50:]))
                    wNormR2.append(np.mean(wNR2[-50:]))
                    wNormI.append(np.mean(wNI[-50:]))
                    wNormO.append(np.mean(wNO[-50:]))
                    hNorm.append(np.mean(hN[-50:]))
                    HM.append(max(hm[-50:]))
                    np.savetxt(os.path.join('data', 'trIms_' + config['save_name'] + '_' + str(len(convCnt)) + '.txt'), np.array(trIm), fmt='%f', delimiter=' ')

                    # Set firing rate homeostatic set point after first problem is learned
                    if firstConv == False:
                        firstConv = True
                        model.updateRegularizerTargets(hNorm[-1], wNormR[-1], wNormI[-1], sess)

                    if config['runType'] != runType.Full:  # for manifold perturbation only
                        if len(convCnt) == 1:
                            model.save(len(convCnt))
                    currSingVals = sess.run([model.sings])
                    singVals[len(convCnt)-1,:] = currSingVals[0]

                if config['runType'] != runType.Full:  # for manifold perturbation only
                    st0 = getStates(sess, config, model, images)
                    if len(convCnt) <= 50:
                        X = st0[0]
                        print('Size: ' + str(X.shape))
                        saveStates[:,:,len(convCnt)-1] = X.T
                        X = st0[1]
                        saveStates[:,:,len(convCnt)-1+50] = X.T
                        
                # Summarize and print learning stats for learned problem
                print('Converged in: ' + str(convCnt[-1]) + ' ' + str(len(convCnt)) + ' (' + str(
                    wNormR[-1]) + ' ' +  str(wNormR2[-1]) + ' ('+ str(maxSingVal) +'), ' + str(wNormI[-1]) + ', ' + str(wNormO[-1]) + ') (' + str(hNorm[-1]) + ', ' + str(HM[-1]) + ') ' + str(np.mean(perf[-50:]))+ ' ' + str(topTenSings) )
                print('Sing Dev: ' + str(np.sum(np.abs(singVals[0,:]-currSingVals[0]))))

                if config['runType'] != runType.Full:  # for manifold perturbation only
                    if len(convCnt) == 50:
                        np.save(os.path.join('data', 'states_' + config['save_name']), saveStates)

                        model.restore(1)
                        oldWts, dEigs, sEigs = resetOutWeightsWithSameStruct(sess, model, config, 4, config['runType'], states=saveStates)
                        model.removeTrainable(['out_RNN_weights', 'out_RNN_biases'], config)
                        model.printTrainable()

                    if len(convCnt) >= 50:
                        print(str(convCnt[-1]))
                        model.restore(1)
                        resetOutWeightsWithSameStruct(sess, model, config, 4, config['runType'], eigs=dEigs, eigsS=sEigs)
                        model.printTrainable()
                
                sys.stdout.flush()

                if config['runType'] == runType.Full:
                    # Finalize graph after homeostatic set point is set
                    if len(convCnt) == 1:
                        sess.graph.finalize()

                # Reset problem specific stats for new problem
                perf = []
                wNR = []
                wNR2 = []
                wNI = []
                wNO = []
                hN = []
                hm = []
                trIm = []
                images = None # This initiates sampling of new images for next problem
                
                # Reset adam's internals before onset of leaarning new problem
                model.resetOpt(sess)

            # Done learning all problems?
            if len(convCnt) >= config['max_tasks']:
                break

    print(convCnt)
    # Write training summaries to file
    np.savetxt(os.path.join('data', 'conv_' + config['save_name']  + '.txt'), np.array(convCnt), fmt='%f', delimiter=' ')
    np.savetxt(os.path.join('data', 'wNormR_' + config['save_name']  + '.txt'), np.array(wNormR), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'wNormR2_' + config['save_name']  + '.txt'), np.array(wNormR2), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'wNormI_' + config['save_name']  + '.txt'), np.array(wNormI), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'wNormO_' + config['save_name']  + '.txt'), np.array(wNormO), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'hNorm_' + config['save_name'] + '.txt'), np.array(hNorm), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'HM_' + config['save_name']  + '.txt'), np.array(HM), fmt='%12.9f', delimiter=' ')
    np.savetxt(os.path.join('data', 'SINGS_' + config['save_name']  + '.txt'), singVals, fmt='%12.9f', delimiter=' ')


if __name__ == '__main__':
    train(seed          = 0,
          batchSize     = 1,
          l2            = 0.0005,
          l2_wR          = 0.001,
          l2_wI          = 0.0001,
          l2_wO          = 0.1,
          learningRateInit = 0.0001,
          svBnd = 10.0,
          rType = runType.Full)
