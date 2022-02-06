import numpy as np
import os
from tqdm import tqdm
from glob import glob
from helper_funcs import *

class ExperienceFilter:
    def __init__(self, data=dict(), axeslabels=tuple(), normalize_axes=True):
        self.axeslabels = axeslabels
        self.normalize_axes = normalize_axes
        self.datapoints = list()
        self.datapoints_normalized = list()
        self.datavalues = list()
        self.add_datapoints(new_data=data)

    def remove_datapoint(self, point):
        if point not in self.datapoints:
            print(f"## INFO: Data point {point} was not in EF.")
            return 
        else:
            idx = self.datapoints.index(point)
            del self.datapoints[idx]
            del self.datavalues[idx]
            if self.normalize_axes: self._normalize()
            return

    def remove_datapoints(self, list_of_points):
        for point in list_of_points:
            self.remove_datapoint(self, point)
        if self.normalize_axes: self._normalize()
        return

    def add_datapoints(self, new_data=dict()):
        self.datapoints.extend(new_data.keys())
        self.datavalues.extend(new_data.values())
        if self.normalize_axes: self._normalize()

    def apply_filter(self, new_point, weighting="inv_distance"):
        weights = self._get_datapoint_weights(new_point, weighting)
        print(weights)
        new_value = np.average(self.datavalues, axis=0, weights=weights)
        return new_value

    def _normalize(self):
        """Normalize all axes of datalabels between 0 and 1."""
        dp = np.vstack(self.datapoints)
        min_dp = np.min(dp, axis=0)
        dp -= min_dp 
        max_dp = np.max(dp, axis=0)
        max_dp[max_dp==0] = 1    # max_dp==0 would only occur if there was a single element along that axis
        self.datapoints_normalized = self._to_tuples(((self.datapoints - min_dp) / max_dp).tolist())
        return

    def _to_tuples(self, L):
        return [tuple(x) for x in L]

    def _get_nearest_neighbor(self, dtpts, point):
        distances = np.array(dtpts) - np.tile(point, (len(dtpts), 1))
        idx = np.argmin(np.linalg.norm(distances, axis=1))
        val = dtpts[idx]
        return idx, val

    def _get_datapoint_weights(self, new_point, weighting):
        if self.normalize_axes:
            dtpts = self.datapoints_normalized
        else:
            dtpts = self.datapoints

        if weighting == "nearest":
            idx, _ = self._get_nearest_neighbor(dtpts, new_point)
            weights = np.zeros(len(dtpts))
            weights[idx] = 1.0
            return weights

        elif weighting == "inv_distance":
            distances = np.array(dtpts) - np.tile(new_point, (len(dtpts), 1))
            weights = np.reciprocal(np.linalg.norm(distances, axis=1))    # weights are inverse of distance
            weights[weights == np.inf] = 1.0e10
            return weights

        elif weighting == "gaussian":
            # TODO: Implement a Gaussian kernel version of this  
            pass

        else:
            print(f"## INFO: Incorrect `weighting` argument.")
            return 