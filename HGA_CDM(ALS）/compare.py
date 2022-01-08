import  numpy as np
import  pandas as pd


# TestA = pd.read_csv('student-sk.csv').values
# RA = pd.read_csv('M1/A_200_10_5.csv', skiprows=50).values

TestA = pd.read_csv('student-sk.csv').values
RA = pd.read_csv('M2/A_120_8.csv', skiprows=60).values
a,b=np.shape(RA)

ccb = 0
for i in range(a):
    tem=1
    for j in range(b):
        tem=tem * int(TestA[i][j]==RA[i][j])
    if tem!=0:
        ccb=ccb+1
    else:
        ccb=ccb+0

print(ccb/a)

count=0
for i in range(a):
    for j in range(b):
        count=count+int(TestA[i][j]==RA[i][j])

print(count/(a*b) )