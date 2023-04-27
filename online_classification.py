import rospy
from std_msgs.msg import String
import numpy as np
import torch
from models import CNN_1D_multihead
from TER_SLOTH import TER_sloth

in_topic = '/har_packets'

class OnlineClassification:
    def __init__(self, window_size=500, feature_size=24, n_actions=5, model_uri="best_model_idle.pth", do_plot=False):
        self.window_size = window_size
        self.feature_size = feature_size
        self.n_actions = n_actions
        self.do_plot = do_plot

        '''initialize SLOTH parameters'''
        gamma = 1
        theta = 0.8
        Rho = 0.08
        Tau = [gamma * value for value in [0.98804504, 0.9947729, 0.99488586, 0.97964764, 0.9955418]]
        C = [theta * value for value in [742.6095076400679, 1180.1586021505377, 1012.3956989247312, 323.61538461538464, 1106.9691943127962]]

        action_colors   = {0: 'red',            1: 'green',              2: 'blue',          3: 'pink',           4: 'yellow', -1: 'white'}
        action_names   = {0: 'ASSEMBLY',            1: 'BOLT',              2: 'HANDOVER',          3: 'PICKUP',           4: 'SCREW'}


        '''initialize the model'''
        self.model = CNN_1D_multihead(feature_size, n_actions).cuda()
        self.model.load_state_dict(torch.load(model_uri))

        self.sloth = TER_sloth(self.model, window_size=window_size, 
                               class_size=n_actions, feature_size=feature_size, 
                               rho=Rho, tau=Tau, c=C, action_names=action_names, 
                               action_colors=action_colors)
        
        '''initialize latest sample to feed into the window'''
        self.latest_sample = np.empty((1, self.feature_size))
        self.latest_sample[:] = np.nan

        '''initialize the window'''
        self.window = np.zeros((1, self.window_size, self.feature_size))

        # '''initialize the subscriber'''
        # self.sub = rospy.Subscriber(in_topic, String, self.callback)

    def try_unpack_msg(self, msg):
        '''check if the message is valid'''
        arg_list = msg.split(',')
        if len(arg_list) != 7:
            print('Invalid message')
            return None, None           
        return arg_list[:-1], arg_list[-1]


    def full_scale_normalize(data):
        acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
        gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

        # 1g equals 8192. The full range is 2g
        data[:,:,acceleration_idxs] = data[:,:,acceleration_idxs] / 16384.0
        data[:,:,gyroscope_idxs] = data[:,:,gyroscope_idxs] / 1000.0

        return data


    def sensor_switch(self, sensor_id):
        '''['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']'''
        if sensor_id == 0:
            return 0
        elif sensor_id == 1:
            return 6
        elif sensor_id == 2:
            return 12
        elif sensor_id == 3:
            return 18


    def callback(self, data):
        arg_list, sensor_name = self.try_unpack_msg(data.data)
        if arg_list is not None and sensor_name is not None:
            print(f'{sensor_name=}')
            print(f'{arg_list=}')

            id = self.sensor_switch(sensor_name)
            self.latest_sample[id : id + 5] = arg_list[:-1]

            if not np.any(np.isnan(self.window)):
                self.window = np.roll(self.window, -1, 1)
                self.window[:, -1, :] = self.latest_sample

                scaled_window = self.full_scale_normalize(self.window)

                prediction, time = self.sloth.classify(scaled_window)

                if self.do_plot:
                    self.sloth.update_plot(prediction, time)
                self.sloth.update_terminal_stats(prediction, time)
                self.latest_sample[:] = np.nan
            else: 
                return


    def get_classification(self):
        return self.classification


    def listener(self):
        rospy.init_node('online_classification', anonymous=True)
        rospy.Subscriber(in_topic, String, self.callback)



if __name__ == '__main__':
    OC = OnlineClassification(window_size=500, 
                              feature_size=24, 
                              n_actions=5, 
                              model_uri="best_model_idle.pth",
                              do_plot=True)
    OC.listener()