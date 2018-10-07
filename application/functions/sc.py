import numpy as np
import brica

class SC(object):
    """
    SC (superior colliculus) module.
    SC outputs action for saccade eye movement.
    """
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

        self.last_fef_data = None
        self.data_size = 128
        self.likelihood_over_threshold = [np.zeros(self.data_size),
                                    np.zeros(self.data_size),
                                    np.zeros(self.data_size)]

        self.likelihood_under_threshold = [np.zeros(self.data_size),
                                     np.zeros(self.data_size),
                                     np.zeros(self.data_size)]

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef']
        # Likelihood thresolds from BG module.
        
        likelihood_thresholds, potentialMap = inputs['from_bg']

        action = self._decide_action(fef_data, likelihood_thresholds)

        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data

        return dict(to_environment=action)

    def _decide_action(self, fef_data, likelihood_thresholds):
        Max = 0.0
        self.likelihoods_over_threshold = []
        self.likelihoods_under_threshold = []

        assert(len(fef_data) == len(likelihood_thresholds))

        count_over = 0
        count_under = 0
        # Calculate average eye ex, ey with has likelihoods over
        # the thresholds from BG.
        for i,data in enumerate(fef_data):
            likelihood = data[0]
            ex = data[1]
            ey = data[2]
            likelihood_threshold = likelihood_thresholds[i]

            if likelihood > likelihood_threshold:
                self.likelihood_over_threshold[0][count_over] = likelihood
                self.likelihood_over_threshold[1][count_over] = ex
                self.likelihood_over_threshold[2][count_over] = ey
                self.likelihoods_over_threshold.append(self.likelihood_over_threshold[0][count_over])

                count_over += 1

            else:
                self.likelihood_under_threshold[0][count_under] = likelihood
                self.likelihood_under_threshold[1][count_under] = ex
                self.likelihood_under_threshold[2][count_under] = ey
                self.likelihoods_under_threshold.append(self.likelihood_under_threshold[0][count_under])
                count_under += 1

        # Action values should be within range [-1.0~1.0]
        if count_over != 0:
            Max = np.argmax(self.likelihoods_over_threshold)
            action = [self.likelihood_over_threshold[1][Max],self.likelihood_over_threshold[2][Max]]
        else:
            Max = np.argmax(self.likelihoods_under_threshold)
            action = [self.likelihood_under_threshold[1][Max],self.likelihood_under_threshold[2][Max]]

        return np.array(action, dtype=np.float32)
