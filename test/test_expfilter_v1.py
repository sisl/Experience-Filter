import numpy as np
import sys
sys.path.append('../src/')
from ExperienceFilter import ExperienceFilter

data = {(1,1,2): np.random.rand(100, 3, 100),
        (1,2,4): np.random.rand(100, 3, 100),
        (2,5,2): np.random.rand(100, 3, 100)}

new_point = (4,5,5)

# With normalization of datapoints, weighting is `inv_distance`
EF = ExperienceFilter(data=data)
new_value = EF.apply_filter(new_point)
print(f"Normalized Datapoints of EF: {EF.datapoints_normalized}")
print(f"New datapoint: {new_point}")
print(f"Value of new datapoint: {new_value}")
print(f"Shape: {new_value.shape}\n")

# With normalization of datapoints, weighting is `gaussian`
EF = ExperienceFilter(data=data)
new_value2 = EF.apply_filter(new_point, weighting='gaussian')
print(f"Normalized Datapoints of EF: {EF.datapoints_normalized}")
print(f"New datapoint: {new_point}")
print(f"Value of new datapoint: {new_value2}")
print(f"Shape: {new_value.shape}\n")

diff = np.linalg.norm(new_value - new_value2)
print(f"Norm of value differences: {diff}")