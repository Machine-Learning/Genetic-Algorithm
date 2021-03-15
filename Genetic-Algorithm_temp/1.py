from testpy import *
from abcdef import *
from client import *
import logging


logging.basicConfig(filename='app.log', filemode='w', format='%(message)s',level=logging.DEBUG)
logging.debug('This will get logged to a file')

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
# print("> What you  want to Do : ")
# print("> if you want to print values markdown format file press 0 ")
# print("> if you want to print value in a txt format press 1")
# print("> If you want to submit value online press 2 (in increasing order of Total Error")
val = 1            # change this value as required 
if(val == 0):
    print("### Some Lowest Values :")   
    j = 1
    for i in arr :
        print("> " + "" + str(j) + ".")
        print("> " + str(i))
        temp_val = []
        j+=1
        if(int(i[1][0])>=0):
            print("```\n" + str(generations[i[1][0]-1][i[1][1]]) + "\n```\n")
        else:
            print("Find Generation in 1.out as Generation < 0 ")
elif(val == 1):
    for i in arr:
        print("> " + str(i))
        if(int(i[1][0])>=0):
            print(generations[i[1][0]-1][i[1][1]])
        else:
            print("Find Generation in 1.out as Generation < 0 ")
elif(val == 2):
    print("Entering submition Zone!")
    print("Enter 1 to submit next value")
    i=0
    while i < len(arr):
        print("-----------------------------------------------------------------------------------------------")
        print(arr[i])
        if(int(arr[i][1][0])>=0):
            print("population vector : " + str(generations[arr[i][1][0]-1][arr[i][1][1]]))
        else:
            print("Find Generation in 1.out as Generation < 0 ")
            break
        # temp = input("Wanna Submit this value : ")
        temp = 1
        # print("temp : ",temp)
        if(int(temp) == 1):
            print("This value is submitted")
            print(submit(SECRET_KEY,generations[arr[i][1][0]-1][arr[i][1][1]]))
            rank = input("Enter the rank obtained : ")
            logging.debug("-----------------------------------------------------------------------------------------------")
            logging.debug("rank : " + str(rank))
            logging.debug("vector : " + str(generations[arr[i][1][0]-1][arr[i][1][1]]))
            logging.debug("Errors : " + str(arr[i][2]))
        else:
            print("Wrong Input")
            i=i-1
        i+=1
else:
    print("Enter correct value")