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

    The last state is an absorbing terminal state with self transition probability of 1
    and null emission probability.
    """
    def __init__(self, n_states,n_features):
        super(HMM, self).__init__()
        # Number of states with an additional absorbing terminal state
        self.n_states = n_states 
        # number of features
        self.n_features = n_features

        # Transition model in log space , with an absorbing terminal state
        self.transition_model = torch.zeros(n_states, n_states)
        
        # Emission model: list of n_states Multivariate Gaussian distributions
        self.emmision_model = [torch.distributions.MultivariateNormal(torch.eye(n_features),torch.eye(n_features,n_features)) for _ in range(n_states)]

        # Initial state in log space
        self.initial_state = torch.zeros(n_states)

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

        N , max_length , n_features_data = data.shape

        assert n_features_data == self.n_features

        # Send data to GPU if available
        if self.cuda_available:
            data = torch.tensor(data).to(self.device)
            mask = torch.tensor(mask).to(self.device)

        # Initialize the initial state
        self.initialize_initial_state()

        # Initialize the transition model
        self.initialize_transition_model(tau)

        # Initialize the emission model
        self.initialize_emission_model(data)

        
        # TODO: Implement the Baum-Welch algorithm
        for i in tqdm.tqdm(range(n_iter)):
            for n_data in range(N):
                one_data = data[n_data,:,:]
                one_mask = mask[n_data,:]
                l = mask[j].sum().item() # length of the sequence
                alpha = self.forward_pass(one_data,one_mask)
                beta = self.backward_pass(one_data,one_mask)
                # evidence likelihood
                evidence = torch.logsumexp(alpha[l-1,:])
                # occupation likelihood
                gamma = alpha + beta - evidence
                # transition likelihood
                xi = torch.full(size=(max_length,self.n_states,self.n_states),fill_value=-np.inf).to(self.device)
                for t in range(1,l):
                    for i in range(self.n_states):
                        for j in  range(self.n_states):
                            xi[t,i,j] = alpha[t-1,i] + self.transition_model[i,j] + self.emission_model.log_prob(data[t])[j] + beta[t,j] - evidence   

                # viterbi decoding
                delta, back_trace = self.viterbi(one_data,one_mask)
                _ , log_prob = self.viterbi_backtrack(delta,back_trace,mask[j])

                print('Log Probability of the most likely path: ',log_prob)

                # update the parameters
                self.update_parameters(one_data,one_mask,gamma,xi)

    def update_parameters(self,data,mask,gamma,xi):
        """
            Update the parameters of the HMM model
            params:
                data: Tensor of shape (max_length, n_features)
                mask: Tensor of shape (max_length)
                gamma: Tensor of shape (max_length, n_states)
                xi: Tensor of shape (max_length, n_states, n_states)
        """
        l = mask.sum().item()

        # update the transition model
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_model[i,j] = torch.logsumexp(xi[1:l,i,j]) - torch.logsumexp(gamma[:,i])

        normalization = 0
        # update the emission model 
        # TODO: Implement the update of the emission model
        for i in range(self.n_states-1):
            for t in range(l):
                self.emission_model.gaussians[i].loc += torch.exp(gamma[t,i]) * data[t,:]
                self.emission_model.gaussians[i].covariance_matrix += torch.exp(gamma[t,i]) * (data[t,:] - self.emission_model.gaussians[i].loc) @ (data[t,:] - self.emission_model.gaussians[i].loc).T
                normalization += torch.exp(gamma[t,i])
            self.emission_model.gaussians[i].loc /= normalization
            self.emission_model.gaussians[i].covariance_matrix /= normalization
    
    def initialize_initial_state(self):
        """
            initialize the initial state of the HMM in log space
            starting from state 0 with probability 1 and 0 otherwise
            the first state is the start state, due to 
            the strict left to right model
            the first state is just state one. 
        """
        self.initial_state[0] = 1
        self.initial_state[1:] = 0
        self.initial_state = torch.log(self.initial_state)
    
    def initialize_transition_model(self,tau):
        """
            Initialize the transition model in log space
            with self transition probability of exp(-1 / (tau - 1))
            
            also initialize the absorbing terminal state
            params:
                tau: length of the frame in samples
        """

        self.transition_model = torch.zeros(self.n_states,self.n_states).to(self.device)


        # self-transition probability 
        # strict left to right model
        for i in range(0,self.n_states - 1):
            self.transition_model[i,i] = np.exp(-1 / (tau - 1))
            self.transition_model[i,i+1] = 1 - self.transition_model[i,i]
        
        # convert to log space
        self.transition_model = torch.log(self.transition_model).to(self.device)
    
    def initialize_emission_model(self,data):
        """
            Initialize the emission model
            use the first sequence of data to initialize the emission model

        """
        global_mu = torch.mean(data[0,:,:],dim=0).to(self.device)
        global_sigma = torch.cov(data[0,:,:].T).to(self.device)

        # Only using the diagonal elements of the covariance matrix
        global_sigma = global_sigma  * torch.eye(self.n_features).to(self.device)

        for i in range(self.n_states-1):
            self.emission_model.gaussians[i].loc = global_mu
            self.emission_model.gaussians[i].covariance_matrix = global_sigma


    def forward_pass(self,data,mask):
        """
            Forward pass of the HMM for a single sequence of data
            returns the alpha matrix where alpha[t,j] is the probability of being in state j at time t 
            given the observations from time 1 to t
            params:
                data: Tensor of shape (max_length, n_features)
                mask: Tensor of shape (max_length)
            returns:
                alpha: Tensor of shape (max_length, n_states)
        """
        max_length , _ = data.shape

        alpha = torch.full(size=(max_length,self.n_states), fill_value=-np.inf).to(self.device)

        # initialize the first state
        alpha[0,:] = self.initial_state + self.emission_model.log_prob(data[0])

        for t in range(1,max_length):
            if not mask[t]:
                break
            for j in range(self.n_states):
                alpha[t,j] = torch.logsumexp(alpha[t-1,:] + self.transition_model[:,j]) + self.emission_model.log_prob(data[t])[j]
        return alpha
    
    
    def backward_pass(self,data,mask):
        """
            Backward pass of the HMM for a single sequence of data
            returns the beta matrix where beta[t,j] is the probability of being in state j at time t 
            given the observations from time t+1 to T
            params:
                data: Tensor of shape (max_length, n_features)
                mask: Tensor of shape (max_length)
            returns:
                beta: Tensor of shape (max_length, n_states)
        """
        max_length , _ = data.shape

        l = mask.sum().item()

        beta = torch.full(size=(max_length,self.n_states), fill_value=-np.inf).to(self.device)

        # initialize the last state
        beta[l-1,:] = 0

        for t in range(l-2,-1,-1):
            for j in range(self.n_states):
                beta[t,j] = torch.logsumexp(beta[t+1,:] + self.transition_model[j,:] + self.emission_model.log_prob(data[t+1])[j])
        return beta
    
    def viterbi(self,data,mask):
        """
            Viterbi pass of the HMM

            params:
                data: Tensor of shape (max_length, n_features)
                mask: Tensor of shape (max_length)
            returns:
                delta: Tensor of shape (max_length, n_states)
                the delta matrix where delta[t,j] is the log probability of the most likely path
                ending in state j at time t

                back_trace: Tensor of shape (max_length, n_states)
                the backtrace matrix where back_trace[t,j] is the state at time t-1
                in the most likely path ending in state j at time t
           
        """
        max_length , _ = data.shape

        # need beta to calculate the probability of the most likely path
        beta = self.backward_pass(data,mask)

        # adding the start and end state
        delta = torch.full(size=(max_length,self.n_states), fill_value=-np.inf).to(self.device)
        back_trace = torch.full(size=(max_length,self.n_states), fill_value=-1).to(self.device)

        # initialize the first state
        delta[0,:] = self.initial_state + self.emission_model.log_prob(data[0])
        # do not need to initialize back_trace for the first time step

        for t in range(1,max_length):
            if not mask[t]:
                break
            for j in range(self.n_states):
                delta[t,j] = torch.max(delta[t-1,:] + self.transition_model[:,j]) + self.emission_model.log_prob(data[t])[j]
                back_trace[t,j] = torch.argmax(delta[t-1,:] + self.transition_model[:,j])
        
    def viterbi_backtrack(self,delta,back_trace , mask):
        """
            Backtrack the most likely path from the delta and back_trace matrix
            params:
                delta: Tensor of shape (max_length, n_states)
                back_trace: Tensor of shape (max_length, n_states)
                mask: Tensor of shape (max_length)
            returns:
                most_likely_path: Tensor of shape (max_length) ,
                log_prob: log probability of the most likely path
        """
        max_length , _ = delta.shape

        most_likely_path = torch.full(size=(max_length), fill_value=-1).to(self.device)
        log_prob = torch.max(delta[-1,:])

        l = mask.sum().item()

        most_likely_path[-1] = torch.argmax(delta[l-1,:])

        for t in range(l-2,-1,-1):
            most_likely_path[t] = back_trace[t+1,most_likely_path[t+1]]
        return most_likely_path , log_prob
    

class EmmisionModel(torch.nn.Module):
    """
        Multivariate Gaussian Emission Model
    """
    def __init__(self,n_features,n_states):
        super(EmmisionModel, self).__init__()
        self.n_features = n_features
        self.n_states = n_states

        # the last state does not have a gaussian distribution

        self.gaussians = [torch.distributions.MultivariateNormal(torch.eye(n_features),torch.eye(n_features,n_features)) for _ in range(n_states-1)]

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda()
    
    def log_prob(self,obs):
        """
            Log probability of observing the observation given the state
            for the terminal state the emmision log probability is -Inf
            params:
                obs: Tensor of shape (n_features)
            returns:
                log_prob: Tensor of shape (n_states)
        """
        log_prob = torch.full(size=(self.n_states), fill_value=-np.inf).to(self.device)
        for i in range(self.n_states-1):
            log_prob[i] = self.gaussians[i].log_prob(obs)
    