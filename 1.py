from testpy import *
arr = []
for i in range(0,len(val)):
    for j in range(0,len(val[0])):
        temp_val1 = val[i][j][0] + val[i][j][1]
        temp_val2 = (i+1,j)
        temp_val3 = ("{:e}".format(val[i][j][0]), "{:e}".format(val[i][j][1]))
        temp_val4 = "line : " +  str(i*33 +j*3 + 2)
        if(val[i][j][0] <= 1.0e11 and val[i][j][1] <= 1.0e11):
            arr.append((temp_val1,temp_val2,temp_val3,temp_val4))
        
arr.sort()

for i in arr :
    print(i)
