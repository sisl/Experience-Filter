import numpy as np
import sys
sys.path.append('../src/')
from ExperienceFilter import ExperienceFilter

data = {(1,1,2): np.random.rand(100, 3, 100), (2,3,2): np.random.rand(100, 3, 100)}
f = ExperienceFilter(data=data)

new_point = (4,5,5)
new_value = f.apply_filter(new_point)
print(new_value)
print(f"Shape: {new_value.shape}")