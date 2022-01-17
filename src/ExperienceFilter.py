import numpy as np

class ExperienceFilter:
   
    def __init__(self, data=dict()):
        self.datapoints = list(data.keys())
        self.datavalues = list(data.values())
        self._dim = len(self.datapoints)

    def apply_filter(self, new_point):
        weights = self._get_datapoint_weights(new_point)
        new_value = np.average(self.datavalues, axis=0, weights=weights)
        return new_value

    def _get_datapoint_weights(self, new_point):   # TODO: Also implement a Gaussian kernel version of this        
        distances = np.array(self.datapoints) - np.tile(new_point, (self._dim, 1))
        weights = np.linalg.norm(distances, axis=1)
        return weights