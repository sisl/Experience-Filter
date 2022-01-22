import numpy as np

class ExperienceFilter:
   
    def __init__(self, data=dict(), axeslabels=tuple()):
        self.axeslabels = axeslabels
        self.datapoints = list(data.keys())
        self.datavalues = list(data.values())

    def apply_filter(self, new_point):
        weights = self._get_datapoint_weights(new_point)
        new_value = np.average(self.datavalues, axis=0, weights=weights)
        return new_value

    def _get_datapoint_weights(self, new_point):   # TODO: Also implement a Gaussian kernel version of this        
        distances = np.array(self.datapoints) - np.tile(new_point, (len(self.datapoints), 1))
        weights = np.reciprocal(np.linalg.norm(distances, axis=1))    # weights are inverse of distance
        weights[weights == np.inf] = 1.0e10
        return weights