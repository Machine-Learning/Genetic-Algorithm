import numpy
import re
import json
import random
from client import *

answer = [-2.04721003e-06, -1.06491014e-12, -1.23642519e-13, 3.30260180e-11,
  -2.09684947e-10, -1.98266818e-15,  8.68207755e-16,  2.79621592e-05,
  -1.97774034e-06, -1.99121904e-08,  1.00440398e-09]

answer2 = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]

print(submit(SECRET_KEY,answer2))