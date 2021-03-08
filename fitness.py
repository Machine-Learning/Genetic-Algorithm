import numpy
import re
import json
import random
from client import *


sending_array = [
  [-1.44909562e-06, -1.09792665e-12, -1.20265933e-13,  7.11858431e-11, -2.44158918e-10, -2.69263868e-15,  5.72924390e-16,  1.81657469e-05, -2.11758873e-06, -9.34359105e-09,  9.12519370e-10],
  [-1.44909562e-06, -1.29718099e-12, -1.20265933e-13,  7.11858431e-11, -2.07917956e-10, -2.69263868e-15,  5.54055927e-16,  1.89746224e-05, -2.11758873e-06, -9.34359105e-09,  9.29939975e-10],
  [-1.53066798e-07, -6.66539037e-12, -1.22231174e-13,  2.42278958e-11, -6.71291154e-11, -4.59547123e-16,  5.97030105e-16,  3.19633693e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],
  [-4.28272514e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.91473763e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],
  [-4.28272514e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.91473763e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],
  [-4.30835708e-07, -4.31672726e-12, -1.05712276e-13,  2.21445186e-11, -6.94460036e-11, -5.80324468e-16,  2.69146865e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],
  [-4.30835708e-07, -4.31672726e-12, -1.05712276e-13,  2.21445186e-11, -6.94460036e-11, -5.80324468e-16,  2.69146865e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],
  [-4.20082896e-07, -3.57384929e-12, -5.65686908e-14,  3.71742278e-11, -8.01852925e-11, -5.52776565e-16,  9.63391924e-16,  3.31382492e-05, -1.76584722e-06, -1.96914593e-08,  8.28733117e-10],
  [-1.79602844e-07, -5.03996481e-12, -9.06687943e-14,  2.30412271e-11, -5.80339280e-11, -6.51454762e-16,  5.97030105e-16,  3.19633693e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10], 
  [-5.01812078e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.49095609e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10]
]  
#   [-7.88046100e-07, -2.56002806e-12, -1.07145150e-13,  4.24744352e-11, -9.89586050e-11, -8.57703420e-16,  9.86446514e-16,  2.87486269e-05, -1.76584722e-06, -1.75757736e-08,  8.28733117e-10],    # ['4.908185e+10', '4.904843e+10'] => 9.8e+10
#   [-1.91954431e-07, -5.11904401e-12, -9.00635140e-14,  2.30412271e-11, -6.03386842e-11, -5.47514207e-16,  5.97030105e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10]     # ['3.914766e+10', '5.488546e+10'] => 9.3e+10
#   [-5.01812078e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.49095609e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['3.926844e+10', '5.461136e+10'] => 9.3e+10
#   [-5.50750049e-07, -3.55332909e-12, -1.10204759e-13,  2.25058822e-11, -5.58925938e-11, -5.21480167e-16,  3.75129661e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['3.742290e+10', '5.966605e+10'] => 9.6e+10
#   [-4.36517350e-07, -4.30042531e-12, -1.10204759e-13,  2.01604763e-11, -5.17146791e-11, -7.04020070e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['5.999000e+10', '3.680471e+10'] => 9.6e+10
#   [-1.79602844e-07, -5.03996481e-12, -9.06687943e-14,  2.30412271e-11, -5.80339280e-11, -6.51454762e-16,  5.97030105e-16,  3.19633693e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10]     # ['6.606371e+10', '2.760565e+10'] => 9.3e+10
#   [-5.50750049e-07, -3.55332909e-12, -1.10204759e-13,  2.25058822e-11, -5.58925938e-11, -5.21480167e-16,  3.75129661e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['3.742290e+10', '5.966605e+10'] => 9.6e+10
#   [-4.36517350e-07, -4.30042531e-12, -1.10204759e-13,  2.01604763e-11, -5.17146791e-11, -7.04020070e-16,  4.54820632e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['5.999000e+10', '3.680471e+10'] => 9.6e+10
#   [-1.90351039e-07, -5.03996481e-12, -1.22231174e-13,  2.94448402e-11, -4.72895170e-11, -6.51454762e-16,  5.97030105e-16,  3.19633693e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10]     # ['6.606371e+10', '2.760565e+10'] => 9.3e+10
  
#   [-7.05440452e-07, -2.43609962e-12, -1.07145150e-13,  3.60676591e-11, -1.23588465e-10, -8.71615134e-16,  9.86446514e-16,  2.87486269e-05, -1.76584722e-06, -1.75757736e-08,  8.28733117e-10],    # ['5.139011e+10', '4.768931e+10'] => 9.8e+10
#   [-5.71293396e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.08357451e-16,  3.75129661e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['3.674190e+10', '6.226746e+10'] => 9.0e+10
#   [-4.28272514e-07, -5.10802204e-12, -1.10204759e-13,  2.01604763e-11, -5.20432045e-11, -7.09993928e-16,  5.31668553e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['6.114141e+10', '3.647669e+10'] => 9.0e+10
#   [-3.40962826e-07, -2.65009789e-12, -1.13214765e-13,  3.37107560e-11, -8.47863075e-11, -9.20359636e-16,  4.21884421e-16,  3.28292451e-05, -1.76584722e-06, -1.91699830e-08,  8.28733117e-10],    # ['7.428342e+10', '2.714243e+10'] => 9.0e+10
#   [-5.71293396e-07, -4.30042531e-12, -1.10204759e-13,  1.94095691e-11, -5.17146791e-11, -5.08357451e-16,  3.75129661e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['3.674190e+10', '6.226746e+10'] => 9.0e+10
#   [-4.28272514e-07, -5.10802204e-12, -1.10204759e-13,  2.01604763e-11, -5.20432045e-11, -7.09993928e-16,  5.31668553e-16,  3.22586960e-05, -1.66491747e-06, -1.96914593e-08,  7.94753961e-10],    # ['6.114141e+10', '3.647669e+10'] => 9.0e+10
  
#   [-3.98738924e-07, -3.97499715e-12, -9.85506563e-14,  3.56416909e-11, -8.47863075e-11, -6.25756575e-16,  3.46358207e-16,  3.28292451e-05, -1.76584722e-06, -1.96914593e-08,  8.28733117e-10],    # ['7.428342e+10', '2.714243e+10'] => 9.0e+10
#   [-3.40962826e-07, -2.65009789e-12, -1.13214765e-13,  3.37107560e-11, -8.47863075e-11, -9.20359636e-16,  4.21884421e-16,  3.28292451e-05, -1.76584722e-06, -1.91699830e-08,  8.28733117e-10],    # ['7.428342e+10', '2.714243e+10'] => 9.0e+10
#   [-4.47008243e-07, -3.65407161e-12, -9.41501134e-14,  3.83539894e-11, -8.42547092e-11, -7.27904372e-16,  9.22674479e-16,  3.31382492e-05, -1.76584722e-06, -1.96914593e-08,  8.28733117e-10],    # ['7.685392e+10', '2.834176e+10'] => 11
#   [-4.67355314e-07, -3.65407161e-12 ,-1.02786647e-13,  2.45478307e-11, -5.29467127e-11, -7.16198475e-16 , 7.79212146e-16,  3.39434178e-05, -1.82435127e-06, -1.96914593e-08 , 8.20059169e-10],
# ]

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
answer = [-1.44909562e-06, -1.09792665e-12, -1.20265933e-13,  7.11858431e-11, -2.44158918e-10, -2.69263868e-15,  5.72924390e-16,  1.81657469e-05, -2.11758873e-06, -9.34359105e-09,  9.12519370e-10]

print(submit(SECRET_KEY,answer))
# value = get_errors(SECRET_KEY,answer3)
# value[0] = "{:e}".format(value[0])
# value[1] = "{:e}".format(value[1])
# print(value)

'''
9 -> increase - train (dec) and validation (inc)
'''
'''
[-1.44909562e-06, -1.09792665e-12, -1.20265933e-13,  7.11858431e-11, -2.44158918e-10, -2.69263868e-15,  5.72924390e-16,  1.81657469e-05, -2.11758873e-06, -9.34359105e-09,  9.12519370e-10]  => [1.011379e+12, 7.044934e+11]  (error = 4.94763848e+12)
[-1.44909562e-06, -1.29718099e-12, -1.20265933e-13,  7.11858431e-11, -2.07917956e-10, -2.69263868e-15,  5.54055927e-16,  1.89746224e-05, -2.11758873e-06, -9.34359105e-09,  9.29939975e-10]  => [3.416817e+11, 2.526358e+11]  (error = 5.76766108e+12)
[-1.44909562e-06, -1.05307640e-12, -1.21066407e-13,  8.46521378e-11, -2.68610422e-10, -2.63156538e-15,  6.62909197e-16,  2.57772837e-05, -2.04404637e-06, -9.63743488e-09,  1.02830306e-09]  => [1.266389e+13, 3.052603e+13]  (error = 1.40596959e+13)
[-1.59092239e-06, -1.29718099e-12, -1.20265933e-13,  7.75103188e-11, -2.96896098e-10, -2.69263868e-15,  5.54055927e-16,  1.77515239e-05, -2.18407471e-06, -9.46900838e-09,  1.02830306e-09]  => [7.676473e+11, 3.194196e+12]  (error = 1.52697537e+13)
[-4.67355314e-07, -3.65407161e-12 ,-1.02786647e-13,  2.45478307e-11, -5.29467127e-11, -7.16198475e-16 , 7.79212146e-16,  3.39434178e-05, -1.82435127e-06, -1.96914593e-08 , 8.20059169e-10]  => [6.256665e+11, 5.805383e+11]  (error = 1.75510553e+13)
[-3.40962826e-07, -2.65009789e-12, -1.13214765e-13,  3.37107560e-11, -8.47863075e-11, -9.20359636e-16,  4.21884421e-16,  3.28292451e-05, -1.76584722e-06, -1.91699830e-08,  8.28733117e-10]  => [6.413612e+10, 6.029538e+10]  (error = 2.32626055e+13)
[-1.23494754e-06, -1.00039028e-12, -1.99559719e-13,  4.34046048e-11, -1.51763117e-10, -1.74364097e-15,  5.27404571e-16,  2.34191499e-05, -1.88620672e-06, -1.52038712e-08,  8.89588136e-10]  => [1.657821e+11, 1.188904e+11]  (error = 2.54521972e+13)
'''