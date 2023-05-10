import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
import numpy as np
import torch
from models import CNN_1D_multihead
from TER_SLOTH import TER_sloth
from scipy.signal import butter, lfilter
from threading import Lock
import matplotlib.pyplot as plt

in_topic = '/har_packets'
FREQ = 25


class OnlineClassification:

    def __init__(self, window_size=500, feature_size=24, n_actions=5, model_uri="best_model_idle.pth", do_plot=False):

        self.window_size = window_size
        self.feature_size = feature_size
        self.n_actions = n_actions
        self.do_plot = do_plot
        self.mutex = Lock()

        '''initialize SLOTH parameters'''
        gamma = 0.9
        theta = 0.8
        Rho = 0.05
        Tau = [gamma * value for value in [0.98804504, 0.9947729, 0.99488586, 0.97964764, 0.9955418]]
        C = [theta * value for value in [742.6095076400679, 1180.1586021505377, 1012.3956989247312, 323.61538461538464, 1106.9691943127962]]
        C = [120, 150, 150, 80, 150]

        action_colors   = {0: 'red',            1: 'green',              2: 'blue',          3: 'pink',           4: 'yellow', -1: 'white'}
        action_names    = {0: 'ASSEMBLY',            1: 'BOLT',              2: 'IDLE',          3: 'PICKUP',           4: 'SCREW'}


        '''initialize the model'''
        self.model = CNN_1D_multihead(feature_size, n_actions).cuda()
        self.model.load_state_dict(torch.load(model_uri))

        self.sloth = TER_sloth(self.model, 
                               window_size=window_size, 
                               class_size=n_actions, 
                               feature_size=feature_size, 
                               rho=Rho, 
                               tau=Tau, 
                               c=C, 
                               action_names=action_names, 
                               action_colors=action_colors)
        
        '''initialize latest sample to feed into the window'''
        self.latest_sample = np.empty((1, self.feature_size))
        # self.latest_sample[:] = np.nan
        self.latest_sample.fill(np.nan)
        '''initialize the window'''
        self.window = np.zeros((1, self.window_size, self.feature_size))
        self.window.fill(np.nan)
        # '''initialize the subscriber'''
        # self.sub = rospy.Subscriber(in_topic, String, self.callback)


    def try_unpack_msg(self, msg):
        '''check if the message is valid'''
        arg_list = msg.split(',')
        if len(arg_list) != 7:
            print('Invalid message')
            return None, None           
        return [float(x) for x in arg_list[:-1]], arg_list[-1]


    def unpack_imu_msg(self, msg):
        data = []
        data.append(msg.linear_acceleration.x)
        data.append(msg.linear_acceleration.y)
        data.append(msg.linear_acceleration.z)
        data.append(msg.angular_velocity.x)
        data.append(msg.angular_velocity.y)
        data.append(msg.angular_velocity.z)
        return data, msg.header.frame_id

    def low_pass(self, sequence, freq):
        '''UNCOMMENT TO PLOT THE ORIGINAL VS THE FILTERED VERSION'''
        # fig = plt.figure()
        # plt.plot(sequence)
        # print(sequence)
        # print(sequence.shape)
        fs = 120
        w = freq / (fs / 2) # Normalize the frequency
        b, a = butter(5, w, btype='low')
        y = lfilter(b,a,sequence)
        # plt.plot(y)
        # plt.show()
        return y

    def frequency_analysis(self, data):
        global FREQ
        newdata = data.copy()

        '''FOR EACH SENSOR FOR EACH FEATURE FILTER THE DATA AND RETURN THE NEW SEQUENCES'''
        for j in range(data.shape[2]):
            newdata[0,:,j] = self.low_pass(newdata[0,:,j], FREQ)

        return newdata


    def full_scale_normalize(self, data):
        tmp = data.copy()
        acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
        gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

        # 1g equals 8192. The full range is 2g
        tmp[:,:,acceleration_idxs] = tmp[:,:,acceleration_idxs] / 16384.0
        tmp[:,:,gyroscope_idxs] = tmp[:,:,gyroscope_idxs] / 100.0
        return tmp


    def sensor_switch(self, sensor_id):
        '''['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']'''
        if sensor_id == 'right_wrist':
            return 0
        elif sensor_id == 'right_back':
            return 6
        elif sensor_id == 'left_wrist':
            return 12
        elif sensor_id == 'left_back':
            return 18


    def update_terminal_stats(self, new_classification, time):
        print("=====================================")  
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("Time: ", time)
        print("Action: ", self.sloth.action_names[np.argmax(new_classification)])
        for i in range(new_classification.shape[0]):
            print(f'{self.sloth.action_names[i]} : {new_classification[i]:.2f}')
        print("=====================================")


    def callback(self, data):
        self.mutex.acquire()

        try:
            arg_list, sensor_name = self.unpack_imu_msg(data)
            if arg_list is not None and sensor_name is not None:
                # print(f'{sensor_name=}')
                # print(f'{arg_list=}')
                # print(not np.isnan(self.latest_sample).any())
                id = self.sensor_switch(sensor_name)
                # print(f'{id=}')
                # print(f'{self.latest_sample.shape=}')
                
                self.latest_sample[0, id : id + 6] = arg_list
                
                if not np.isnan(self.latest_sample).any():# and not np.isnan(self.window).any():
                    self.window = np.roll(self.window, -1, axis=1)
                    # print(f'{self.latest_sample=}')
                    self.window[0, -1, :] = self.latest_sample.copy()

                    if not np.isnan(self.window).any():
                        # scaled_window = self.frequency_analysis(self.window)
                        scaled_window = self.full_scale_normalize(self.window)
                        padded_window = np.zeros((1, 500, 24))
                        padded_window[0, 500-scaled_window.shape[1]:, :] = scaled_window[0, :, :]
                        scaled_tensor_window = torch.from_numpy(padded_window).float().cuda()
                        prediction, time = self.sloth.classify(scaled_tensor_window)
                        # print("=====================================")
                        # print([f'{s:.2f}' for s in scaled_tensor_window[0, -1, :]][:6])
                        # print([f'{s:.2f}' for s in scaled_tensor_window[0, -1, :]][6:12])
                        # print([f'{s:.2f}' for s in scaled_tensor_window[0, -1, :]][12:18])
                        # print([f'{s:.2f}' for s in scaled_tensor_window[0, -1, :]][18:24])
                        # print(scaled_tensor_window[0,:,2])
                        # print(self.window.shape)
                        # sys.exit()
                        if self.do_plot:
                            self.sloth.update_plot(prediction, time)
                        # self.update_terminal_stats(prediction, time)
                        self.sloth.detect()
                    self.latest_sample.fill(np.nan)
        finally:
            self.mutex.release()
        return


    def get_classification(self):
        return self.classification


    def listener(self):
        rospy.init_node('online_classification', anonymous=True)
        rospy.Subscriber(in_topic, Imu, self.callback)
        rospy.spin()


if __name__ == '__main__':
    OC = OnlineClassification(window_size=300, 
                              feature_size=24, 
                              n_actions=5, 
                              model_uri="best_model908.pth",
                              do_plot=False)
    OC.listener()

    # space = [x for x in range(100)]
    # data = np.zeros((1,100,12))
    # for i in range(12):
    #     data[:, :, i] = space
    # print(data)
    # data = np.roll(data, -1, axis=1)
    # data[0, -1, :] = 100
    # print(data)
    # data = np.roll(data, -1, axis=1)
    # data[0, -1, :] = 101
    # print(data)
    # data = np.roll(data, -1, axis=1)
    # data[0, -1, :] = 102
    # print(data)