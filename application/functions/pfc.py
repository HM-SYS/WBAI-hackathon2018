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
        print(self.likelihood)
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
        #minPotentialValue = 0
        #minPotentialValue > 0.5

        #if minPotentialValue in self.potentialMap_8shape :
        self.likelihood *= self.decay_rate
        #else :
            #self.potentialMap_8shape = np.ones((8 , 8))

class PFC(object):
    def __init__(self):
        self.timing = brica.Timing(3, 1, 0)

        self.cursor_find_accmulator = CursorFindAccumulator()

        self.phase = Phase.INIT

        self.potentialMap_8shape = np.ones((8 , 8))


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


        """"""

        if inputs['from_hp'] is not None:
            map_image, angle = inputs['from_hp']
            self.potentialMap_8shape = self.create_potentialMap(angle)
            self.potentialMap = np.reshape(self.potentialMap_8shape,(1,64))
            #print("angle_h : " + str(angle[0]))
            #print("angle_v : " + str(angle[1]))
            #self.potentialMap = self.create_potentialMap()
            #self.potentialMap = self.create_potentialMap(map_image)
            # Allocentrix map image from hippocampal formatin module.

            # Using potentail_map
        """"""
        """"""

        if inputs['from_fef'] is not None:
            fef_data = inputs['from_fef']
            self.visualBuffer = self.create_visualBuffer(fef_data)

        """"""
        if inputs['from_bg'] is not None:
            reward = inputs['from_bg']
            if reward != 0:
                self.potentialMap_8shape = np.ones((8 , 8))


        # This is a very sample implementation of phase detection.
        # You should change here as you like.
        self.cursor_find_accmulator.process(retina_image)
        self.cursor_find_accmulator.post_process()

        if self.phase == Phase.INIT:
            if self.cursor_find_accmulator.likelihood > 0.6:
                self.phase = Phase.START
        elif self.phase == Phase.START:
            if self.cursor_find_accmulator.likelihood < 0.4:
                self.phase = Phase.TARGET
        else:
            if self.cursor_find_accmulator.likelihood > 0.2:
                self.phase = Phase.START

        if self.phase == Phase.INIT or self.phase == Phase.START:
            # TODO: 領野をまたいだ共通phaseをどう定義するか？
            fef_message = 0
        else:
            fef_message = 1

        return dict(to_fef=[fef_message, self.potentialMap, map_image],
                    to_bg=self.potentialMap)

        """"""

    def create_visualBuffer(self, fef_data) :
        saliencyAcc = []
        cursorAcc = []
        visualBuffer = []
        data_len = len(fef_data) // 2
        for i in range(data_len):
            saliencyAcc.append(fef_data[i][0])
            cursorAcc.append(fef_data[i + data_len][0])
            visualBuffer.append((fef_data[i][0]) + (fef_data[i + data_len][0]))

        return visualBuffer

        """"""

    def create_potentialMap(self, angle) :
        val = -0.4
        angle_h = -angle[0]
        angle_v = -angle[1]
        #print(angle_h)
        #print(angle_v)
        param = 0.1
        attenuationValue = 0.95

        for i in range(8):
            if(val < angle_h and angle_h <= val + param):
                xPosi = i
            if(val < angle_v and angle_v <= val + param):
                yPosi = i
            val += param

        self.potentialMap_8shape[xPosi][yPosi] = self.potentialMap_8shape[xPosi][yPosi] * attenuationValue
        #print(self.potentialMap_8shape)
        #if(self.potentialMap <= 0.5):
            #self.likelihood *= 0.5


        #self.potentialMap_8shape = self.potentialMap_8shape + attenuationValue * 10000000000000
        #self.likelihood = self.potentialMap_8shape * ( i / 1.5)
        #print(self.likelihood)

        #print(str(self.potentialMap))

        return self.potentialMap_8shape
