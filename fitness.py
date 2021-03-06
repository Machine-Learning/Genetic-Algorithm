import numpy
import re
import json
import random
from client import *
god = [[3.416817e+11, 2.526358e+11],
      [1.657821e+11, 1.188904e+11],
      [1.901893e+11, 3.025361e+11],
      [7.676473e+11, 3.194196e+12],
      [1.351072e+10, 3.682966e+11],
      [1.345709e+10, 3.583668e+11]]

god_value = [[-1.44909562e-06, -1.29718099e-12, -1.20265933e-13,  7.11858431e-11, -2.07917956e-10, -2.69263868e-15,  5.54055927e-16,  1.89746224e-05, -2.11758873e-06, -9.34359105e-09,  9.29939975e-10],
  [-1.23494754e-06, -1.00039028e-12, -1.99559719e-13,  4.34046048e-11, -1.51763117e-10, -1.74364097e-15,  5.27404571e-16,  2.34191499e-05, -1.88620672e-06, -1.52038712e-08,  8.89588136e-10],
  [-1.61301154e-06, -6.92844074e-13, -1.59316731e-13,  6.00231194e-11, -2.94835027e-10, -2.19483348e-15,  5.66619558e-16,  1.81657469e-05, -2.11758873e-06, -1.29200926e-08,  9.96282176e-10],
  [-1.59092239e-06, -1.29718099e-12, -1.20265933e-13,  7.75103188e-11, -2.96896098e-10, -2.69263868e-15,  5.54055927e-16,  1.77515239e-05, -2.18407471e-06, -9.46900838e-09,  1.02830306e-09],
  [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10],
  [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04821003e-06, -1.59792834e-08,  9.98214034e-10]]
imp_position = [8,9,11,10]
not_imp = [6]
answer = [-2.04721003e-06, -1.06491014e-12, -1.23642519e-13, 3.30260180e-11,
  -2.09684947e-10, -1.98266818e-15,  8.68207755e-16,  2.79621592e-05,
  -1.97774034e-06, -1.99121904e-08,  1.00440398e-09]

# answer2 = [-1.23494754e-06, -1.00039028e-12, -1.99559719e-13,  4.34046048e-11,
#  -1.51763117e-10, -1.74364097e-15,  5.27404571e-16,  2.34191499e-05,
#  -1.88620672e-06, -1.52038712e-08,  8.89588136e-10]
# answer3=   [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11,
#  -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05,
#  -2.04721003e-06, -1.59792834e-08,  9.98214034e-10],


# answer3 = [-1.23494754e-06, -1.00039028e-12, -1.99559719e-13,  4.34046048e-11,
#  -1.51763117e-10, -1.74364097e-15,  5.27404571e-16,  2.34191499e-05,
#  -1.88620672e-06, -1.52038712e-08,  9.00588136e-10]
answer4 = [-1.44909562e-06, -1.09792665e-12, -1.20265933e-13,  7.11858431e-11,
  -2.44158918e-10, -2.69263868e-15,  5.72924390e-16,  1.81657469e-05,
  -2.11758873e-06, -9.34359105e-09,  9.12519370e-10]
answer3 = [-1.79185355e-06, -1.77031102e-12, -1.90596398e-13,  2.38789784e-11,
  -1.27818453e-10, -6.21554843e-16,  4.50059546e-16,  2.22397548e-05,
  -2.09239354e-06, -1.62563606e-08,  1.04100895e-09]
# print(submit(SECRET_KEY,god_value[0]))
# print(answer4)
print(submit(SECRET_KEY,answer4))
# value = get_errors(SECRET_KEY,answer3)
# value[0] = "{:e}".format(value[0])
# value[1] = "{:e}".format(value[1])
# print(value)

'''
9 -> increase - train (dec) and validation (inc)
'''
'''
[-1.44909562e-06, -1.05307640e-12, -1.21066407e-13,  8.46521378e-11, -2.68610422e-10, -2.63156538e-15,  6.62909197e-16,  2.57772837e-05, -2.04404637e-06, -9.63743488e-09,  1.02830306e-09]  => [1.266389e+13, 3.052603e+13]  (rank = 33)
[-1.44909562e-06, -1.09792665e-12, -1.20265933e-13,  7.11858431e-11, -2.44158918e-10, -2.69263868e-15,  5.72924390e-16,  1.81657469e-05, -2.11758873e-06, -9.34359105e-09,  9.12519370e-10]  => [1.011379e+12, 7.044934e+11]  (rank = 19)
'''