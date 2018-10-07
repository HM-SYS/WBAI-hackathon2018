import cv2
import numpy as np

import brica
from .utils import load_image

"""
This is a sample implemention of PFC (Prefrontal cortex) module.
You can change this as you like.
"""

class Phase(object):
    INIT = -1  # Initial phase
    START = 0  # Start phase while finding red cross cursor
    TARGET = 1 # Target finding phsae


class CursorFindAccumulator(object):
    def __init__(self, decay_rate=0.9):
        # Accumulated likelilood
        self.decay_rate = decay_rate
        self.likelihood = 0.0

        self.cursor_template = load_image("data/debug_cursor_template_w.png")

    def accumulate(self, value):
        self.likelihood += value
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)

    def reset(self):
        self.likelihood = 0.0

    def process(self, retina_image):
        match = cv2.matchTemplate(retina_image, self.cursor_template,
                                  cv2.TM_CCOEFF_NORMED)
        match_rate = np.max(match)
        self.accumulate(match_rate * 0.1)

    def post_process(self):
        # Decay likelihood
        self.likelihood *= self.decay_rate



class PFC(object):
    def __init__(self):
        self.timing = brica.Timing(3, 1, 0)

        self.cursor_find_accmulator = CursorFindAccumulator()

        self.phase = Phase.INIT
        self.buffer_size = 10
        self.pixel_size = 21
        self.value_size = 1
        self.pre_reward = 0
        self.buffer_index = 0
        self.episode_buffer_List_index = 0
        self.map_image = np.zeros((128, 128, 3), dtype=np.uint8)
        self.image_dim = 128
        self.feature_threshold = 0.2
        self.episode_index = 0
        self.pre_reward = 0
        self.BefferStock_decay_rate = 0.9
        self.episode = None

        self.potentialMap_8shape = np.ones((8 , 8))

        self.visualBuffer = [np.zeros((8, 8),dtype = np.float32)]

        self.feature_list = [np.zeros((self.image_dim*self.image_dim),dtype = np.float32),
                             np.zeros((self.image_dim*self.image_dim),dtype=np.uint8),
                             np.zeros((self.image_dim*self.image_dim),dtype=np.uint8)]

        self. feature = [np.zeros((self.pixel_size, self.pixel_size), dtype = np.float32)]

        self.episode_buffer = [np.zeros((self.buffer_size, self.pixel_size * self.pixel_size), dtype=np.float32),
                               np.zeros((self.buffer_size, self.value_size), dtype = np.float32)]

        self.episode_buffer_stock = [np.zeros((self.buffer_size, self.pixel_size * self.pixel_size), dtype=np.float32),
                               np.zeros((self.buffer_size, self.value_size), dtype = np.float32)]

        self.potentialMap = [np.zeros((1, self.image_dim//2), dtype = np.float32)]

    def __call__(self, inputs):
        if 'from_vc' not in inputs:
            raise Exception('PFC did not recieve from VC')
        if 'from_fef' not in inputs:
            raise Exception('PFC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('PFC did not recieve from BG')
        if 'from_hp' not in inputs:
            raise Exception('PFC did not recieve from HP')

        # Image from Visual cortex module.
        retina_image = inputs['from_vc']
        # Allocentrix map image from hippocampal formatin module.
        if inputs['from_hp'] is not None:
            map_image, angle = inputs['from_hp']
            self.potentialMap_8shape = self.create_potentialMap(angle)
            self.potentialMap = np.reshape(self.potentialMap_8shape,(1,64))

        if inputs['from_fef'] is not None:
            fef_data, saliency_map = inputs['from_fef']
            self.feature = self.Feature_Find(saliency_map)
            self.feature = self.feature.reshape(1, self.pixel_size*self.pixel_size)
            self.episode_buffer_stock = self.BefferStock(self.feature, self.pre_reward)
            if self.pre_reward != 0:
                episode_flag = True
                self.episode = self.episode_buffer_stock
            else:
                self.episode = None

        if inputs['from_bg'] is not None:
            reward = inputs['from_bg']
            if reward != 0:
                self.potentialMap_8shape = np.ones((8 , 8))
            self.pre_reward = reward

        # This is a very sample implementation of phase detection.
        # You should change here as you like.
        self.cursor_find_accmulator.process(retina_image)
        self.cursor_find_accmulator.post_process()

      if self.phase == Phase.INIT:
            if self.cursor_find_accmulator.likelihood > 0.5:
                self.phase = Phase.START

        elif self.phase == Phase.TARGET:
            if self.cursor_find_accmulator.likelihood > 0.4 and reward != 0:
                self.phase = Phase.START

        elif self.phase == Phase.START:
            if self.cursor_find_accmulator.likelihood > 0.2:
                self.phase = Phase.TARGET

        if self.phase == Phase.INIT or self.phase == Phase.START:
            # TODO: 領野をまたいだ共通phaseをどう定義するか？
            fef_message = 0
        else:
            fef_message = 1

        return dict(to_fef=[fef_message, self.potentialMap, map_image],
                    to_bg=self.potentialMap,
                    to_hp=self.episode)

    def BefferStock(self, feature, reward):
        self.buffer_index = self.buffer_index % self.buffer_size
        self.episode_buffer[0][self.buffer_index] = feature
        self.episode_buffer[1][self.buffer_index] = reward
        self.buffer_index += 1
        if reward != 0:
            for i in range(self.buffer_index):
                self.episode_buffer[1][self.buffer_index - i]  = reward*(self.BefferStock_decay_rate**i)

        if self.buffer_index == self.buffer_size:
            self.buffer_index = 0

        return self.episode_buffer

    def Feature_Find(self, saliency_map):
        count = 0
        Max_feature = 0
        self.feature_lists = []
        self.episode_feature = np.array([[0.0]*self.pixel_size for i in range(self.pixel_size)])

        for i in range(self.image_dim):
            for j in range(self.image_dim):
                if saliency_map[i][j] < self.feature_threshold:
                    self.feature_list[0][count] = 0.0
                    self.feature_list[1][count] = i
                    self.feature_list[2][count] = j
                    self.feature_lists.append(self.feature_list[0][count])
                    count += 1

                else:
                    self.feature_list[0][count] = saliency_map[i][j]
                    self.feature_list[1][count] = i
                    self.feature_list[2][count] = j
                    self.feature_lists.append(self.feature_list[0][count])
                    count += 1

        Max_feature = np.argmax(self.feature_lists)

        for x in range(self.pixel_size):
            Central_visual_field_x = (self.feature_list[1][Max_feature] - ((self.pixel_size - 1) // 2)) + x
            for y in range(self.pixel_size):
                Central_visual_field_y = (self.feature_list[2][Max_feature] - ((self.pixel_size - 1) // 2)) + y
                if (Central_visual_field_x < 0) or (self.image_dim <= Central_visual_field_x):
                    self.episode_feature[x][y] = 0.0
                elif (Central_visual_field_y < 0) or (self.image_dim <= Central_visual_field_y):
                    self.episode_feature[x][y] = 0.0
                else:
                    self.episode_feature[x][y] = saliency_map[Central_visual_field_x][Central_visual_field_y]
                    if self.episode_feature[x][y] < self.feature_threshold:
                        self.episode_feature[x][y] = 0.0

        return self.episode_feature

    def create_potentialMap(self, angle) :
        val = -0.4
        angle_h = -angle[0]
        angle_v = -angle[1]
        param = 0.1
        attenuationValue = 0.95
        for i in range(8):
           if(val < angle_h and angle_h <= val + param):
               xPosi = i
           if(val < angle_v and angle_v <= val + param):
               yPosi = i
           val += param
        self.potentialMap_8shape[xPosi][yPosi] = self.potentialMap_8shape[xPosi][yPosi] * attenuationValue

        return self.potentialMap_8shape
