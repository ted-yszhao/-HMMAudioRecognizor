import torch
import numpy as np
import soundfile as sf
import pandas as pd
import scipy as sp
import os
import librosa
import tqdm

class HMM(torch.nn.Module ):
    """
    Hidden Markov Model with discrete transition probability
    and multivariate Gaussian Emission Probabilities.
    """
    def __init__(self, n_states,n_features):
        super(HMM, self).__init__()
        # Number of states
        self.n_states = n_states
        # number of features
        self.n_features = n_features

        # Transition model in log space
        self.transition_model = torch.rand(n_states, n_states)
        
        # Emission model
        self.emission_model = EmissionModel(n_states,n_features)

        # Initial state
        self.initial_state = torch.rand(n_states)

        
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            print('Cuda Available.')
            self.device = torch.device('cuda')
            self.cuda()  


    def encode(self, data ,sr = 16000 , frame_length = 30, hop_length = 10):
        """
            Encode the data using librosa
            params:
                data: list of audio file as a list of numpy arrays, data[i] is the i-th audio file
                        of shape (n_samples,)

                sr: sample rate of the audio files, detault 16000
                frame_length: length of the frame in milliseconds  
                hop_length: hop length in milliseconds

            returns:
                encoded_data: padded Tensor of shape (n_samples, max_length, n_features)
                mask : Tensor of shape (n_samples, max_length)
                tau : length of the frame in samples
        """
        n_fft = int(sr * frame_length / 1000)
        hop_length = int(sr * hop_length / 1000)

        encoded_data = []
        for i in range(len(data)):
            data[i] = data[i].astype(np.float32)
            # normalize the data
            data[i] = data[i] / np.max(np.abs(data[i]))
            encoded_data.append(librosa.feature.mfcc(y=data[i], sr=sr,n_fft=n_fft,hop_length=hop_length,n_mfcc=self.n_features).T) 

        tau = n_fft

        encoded_data , mask = self.pad_data(encoded_data)
        
        return encoded_data , mask , tau

    def pad_data(self,data):
        """
            Pad the data with -Inf to make all the data of the same length
            params:
                data: list of encoded data as a list of T x n_features matrix
            returns:   
                padded_data: Tensor of shape (n_samples, max_length, n_features)
                mask: Tensor of shape (n_samples, max_length)
        """
        n_samples = len(data)
        max_length = max([x.shape[0] for x in data])

        padded_data = torch.full((n_samples,max_length,self.n_features),-np.inf)

        for i in range(n_samples):
            padded_data[i,:data[i].shape[0],:] = torch.tensor(data[i])

        mask = padded_data[:,:,0] != -np.inf
        return padded_data , mask
    
    
    def train(self, data, n_iter = 100):
        """
            Train the HMM model using Expectation Maximization
            i.e. the Baum-Welch algorithm.
            params:
                data: list of audio files as numpy array
                n_iter: number of EM iterations for training
        """

        # Perform MFCC encoding
        data , mask, tau = self.encode(data)

        N , _ , n_features_data = data.shape

        assert n_features_data == self.n_features

        # Send data to GPU if available
        if self.cuda_available:
            data = torch.tensor(data).to(self.device)
            mask = torch.tensor(mask).to(self.device)

        # Initialize the model
        self.initialize_initial_state()

        # Initialize the transition model
        self.initialize_transition_model(tau)

        # Initialize the emission model
        self.initialize_emission_model(data)
        
        
        # TODO: Implement the Baum-Welch algorithm
        for i in tqdm.tqdm(range(n_iter)):
            for x in data:
                pass
    
    def initialize_initial_state(self):
        """
            initialize the initial state of the HMM in log space
            starting from state 0 with probability 1 and 0 otherwise
        """
        self.initial_state[0] = 1
        self.initial_state[1:] = 0
        self.initial_state = torch.log(self.initial_state)
    
    def initialize_transition_model(self,tau):
        """
            Initialize the transition model in log space
            with self transition probability of exp(-1 / (tau - 1))
            params:
                tau: length of the frame in samples
        """

        self.transition_model = torch.zeros(self.n_states,self.n_states)

        ## always exit from state 0
        self.transition_model[0,0] = 0
        self.transition_model[0,1] = 1

        # self-transition probability 
        # strict left to right model
        for i in range(1,self.n_states-1):
            self.transition_model[i,i] = np.exp(-1 / (tau - 1))
            self.transition_model[i,i+1] = 1 - self.transition_model[i,i]
        
        # absorbing terminal state
        self.transition_model[-1,-1] = 1

        # convert to log space
        self.transition_model = torch.log(self.transition_model)
    
    def initialize_emission_model(self,data):
        """
            Initialize the emission model

        """
        global_mu = torch.mean(data,dim=0).to(self.device)
        global_sigma = torch.cov(data.T).to(self.device)

        # Only using the diagonal elements of the covariance matrix
        global_sigma = global_sigma  * torch.eye(self.n_features).to(self.device)

        for i in range(self.n_states):
            self.emission_model.gaussians[i].loc = global_mu
            self.emission_model.gaussians[i].covariance_matrix = global_sigma

    def hmm_forward(self,data):
        """
            Forward pass of the HMM
            params:
                data: T x n_features matrix
            returns:
                alpha: T x n_states matrix
        """
        T , n_features = data.shape

        # adding the start and end state
        alpha = torch.zeros(T + 2,self.n_states).to(self.device)

        # initialize the first state
        alpha[0,:] = self.initial_state 
        
        # recursion forward pass

        # because of the start state, the data for state i is at data[i-1]
        for t in range(1,T + 1):
            for j in range(self.n_states):
                alpha[t,j] = torch.logsumexp(alpha[t-1] + self.transition_model[:, j]) + self.emission_model.log_prob(j,data[t-1])
        return alpha

class EmissionModel(torch.nn.Module):
    """Emmision model for the HMM"""
    def __init__(self,n_states,n_features):
        super(EmissionModel, self).__init__()
        self.n_states = n_states
        self.n_features = n_features

        self.gaussians = [torch.distributions.MultivariateNormal(torch.eye(n_features),torch.eye(n_features,n_features)) for i in range(n_states)]

    def log_prob(self,state,data):
        """
            Compute the log probability of the data given the emission model
            params:
                state: int
                data: T x n_features matrix
            returns:
                log_prob: T x n_states matrix
        """

        # no emission from the first and last state
        if state == 0 or state == self.n_states - 1:
            return -torch.inf
        else:
            return self.gaussians[state].log_prob(data)