#!/usr/bin/python

import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import os


class TER_sloth:
    def __init__(self, py_model, window_size, class_size, feature_size, rho, tau, c, action_names, action_colors):

        self.window_size = window_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.probabilities_size = window_size
        self.action_names = action_names
        self.action_colors = action_colors

        self.model = py_model
        self.window = np.empty((1,self.window_size,self.feature_size))
        self.window[:] = np.nan

        self.probabilities = np.empty((1,self.probabilities_size,self.class_size))
        self.probabilities[:] = np.nan

        ''' SLOTH parameters '''
        self.rho = rho
        self.tau = tau
        self.c = c

        print("=============================")
        print("SLOTH initialized")
        print("rho: ", self.rho)
        print("tau: ", self.tau)
        print("c: ", self.c)
        print("=============================")

        # print(self.probabilities)
        # print(self.probabilities.shape)
        # print(self.window)
        # print(self.window.shape)

        self.time = 0
        self.peaks = np.zeros((1,self.class_size))

        self.gestures = []

        ''' Stuff for plotting'''
        # plt.ion()
        # self.fig, self.axs = plt.subplots(len(c))
        # self.fig.set_figheight(10)
        # self.fig.set_figwidth(16)
        # self.plot_buffer = np.zeros((len(c), window_size))


    '''plot the classification results in a live plot'''
    def update_plot(self, new_classification, time):
        self.plot_buffer = np.roll(self.plot_buffer, -1, axis=1)
        self.plot_buffer[:,-1] = new_classification

        x = range(time-1, self.plot_buffer.shape[1]+time-1)
        # print([xi + time -1 for xi in x])
        # exit()
        
        for i in range(self.plot_buffer.shape[0]):
            self.axs[i].clear()
            self.axs[i].scatter(x, self.plot_buffer[i], s=0.4, c=self.action_colors[i])
            self.axs[i].set_ylim([-0.1, 1.1])
            self.axs[i].set_title(self.action_names[i])
        self.fig.canvas.flush_events()


    '''prints the classification results in the terminal'''
    def update_terminal_stats(self, new_classification, time):
        print("=====================================")  
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Time: ", time)
        print("Action: ", self.action_names[np.argmax(new_classification)])
        for i in range(new_classification.shape[0]):
            print(f'{self.action_names[i]} : {new_classification[i]:.2f}')
        print("=====================================")


    '''classify input window'''
    def classify(self, input_data):
        # if not np.any(np.isnan(self.window)):
            # with self.graph.as_default():
        self.probabilities = np.roll(self.probabilities,-1,1)
        output = self.model(input_data)
        self.probabilities[0,-1,:] = output.cpu().detach().numpy()
        del output
        # print(self.time, '----', self.action_names[np.argmax(self.probabilities[0,-1,:])])
        self.time += 1
        return self.probabilities[0,-1,:], self.time
        # np.set_printoptions(precision=1)
        # print("probabilities: ", self.probabilities)
        # else:
        #     print("The sliding window is not completely full")


    '''detect gestures'''
    def detect(self):
        delta_prob = (self.probabilities[0,-1,:] - self.probabilities[0,-1-1,:]) 
        possible_peaks = np.where(delta_prob > self.rho)
        # print(delta_prob)
        possible_peaks = possible_peaks[0]

        for ids in possible_peaks:
            if self.peaks[0, ids] == 0:
                self.peaks[0, ids] = self.time
                # print(self.time)
            else:
                time_diff = self.time - self.peaks[0, ids]
                if time_diff >= self.c[ids]:
                    self.peaks[0, ids] = self.time

        active_peaks = np.where(self.peaks[0,:]> 0)
        active_peaks = active_peaks[0]

        for ids in active_peaks:
            time_diff = self.time - self.peaks[0, ids] + 1
            if time_diff >= self.c[ids]:
                start = int(self.probabilities_size-time_diff)
                prob_mean = np.mean(self.probabilities[0,start:,ids])
                if prob_mean > self.tau[ids]:
                    print(self.action_names[ids], '---', self.time-time_diff, '-', self.time)
                    self.peaks[0, ids] = 0
                    self.gestures.append(ids+1)

    def window_update(self, x, y, z):
        self.window = np.roll(self.window,self.window_size-1,1)
        self.window[:,-1,0] = x
        self.window[:,-1,1] = y
        self.window[:,-1,2] = z
        self.time += 1

    def display(self):
        plt.clf()
        plt.figure(1)
        plt.subplot(911)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,0])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(912)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,1])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(913)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,4])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(914)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,5])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(915)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,2])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(916)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,3])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])

        plt.subplot(917)
        plt.plot(range(0,self.window_size),self.window[0,:,0])
        plt.axis([0, self.window_size, -10, 10])
        plt.subplot(918)
        plt.plot(range(0,self.window_size),self.window[0,:,1])
        plt.axis([0, self.window_size, -10, 10])
        plt.subplot(919)
        plt.plot(range(0,self.window_size),self.window[0,:,2])
        plt.axis([0, self.window_size, -10, 10])
        
        plt.ion()
        plt.pause(0.05)

    def get_gesures(self):
        temp = self.gestures
        self.gestures = []
        return temp