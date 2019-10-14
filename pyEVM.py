import numpy as np
import cv2
import butter
from matplotlib import pyplot as plt

# 1. Read Video file

# generate video object
filename = "./input/original_video_SR30.mp4"
cap = cv2.VideoCapture(filename)

# 2. Gray Scale로 변환

# 첫번째 프레임 읽기
ret, frame = cap.read()

# 영상의 사이즈 확인
frame_size = frame.shape
print("Shape of image is : ",frame.shape)

# Gray Sacle 이미지로 변환
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 4. Pyramid
# 한 레벨에 대해 먼저 해보기

Gau = [gray]
Down = cv2.pyrDown(gray)
Gau.append(Down)

for i in range (0, 5):
    Down = cv2.pyrDown(Down)
    Gau.append(Down)

Gau_copy = []
Gau_copy[:] = Gau
Gau_copy.reverse()

# 이미지를 sequential하게 저장
Gau_seq = [[],[],[],[],[],[]]
for i in range (0,6):
    Gau_seq[i].append(Gau_copy.pop())
    
Lap = []
for i in range (0, 6):
    Up = cv2.pyrUp(Gau.pop())
    Lap.append(Up - Gau[len(Gau)-1])
    
#Lap.reverse()

Lap_seq = [[],[],[],[],[],[]]
for i in range (0,6):
    Lap_seq[i].append(Lap.pop())

# 모든 프레임에 대해 적용

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #hist = cv2.equalizeHist(gray)
    
    Gau = [gray]
    Down = cv2.pyrDown(gray)
    Gau.append(Down)

    for i in range (0, 5):
        Down = cv2.pyrDown(Down)
        Gau.append(Down)
    
    Gau_copy = []
    Gau_copy[:] = Gau
    Gau_copy.reverse()
    
    for i in range (0,6):
        Gau_seq[i].append(Gau_copy.pop())

    Lap = []
    for i in range (0, 6):
        Up = cv2.pyrUp(Gau.pop())
        Lap.append(Up - Gau[len(Gau)-1])

    #Lap.reverse()
    
    for i in range (0,6):
        Lap_seq[i].append(Lap.pop())
    ret, frame = cap.read()

# 5. 필터링 및 각 진동 확대
# Parameter Initializing 

SR = 2200 # SR is Sampling Rate
Fn = SR/2
T = 1/SR # T is Period
Low_freq = 350
High_freq = 370
alpha = 10.0

# 연산을 위해 부동소수점으로 변환

for i in range(0, 6):
    for j in range(0, 101):
        Lap_seq[i][j] = Lap_seq[i][j].astype('float64')

# 원하는 주파수 선택적으로 확대 
import time
start = time.time()

# level 0
for i in range (0, frame_size[0]):
    for j in range (0, frame_size[1]):
        time_stack = []
        
        for k in range (0, 101):
            time_stack.append(Lap_seq[0][k][i][j])
            
        time_stack = butter.butter_bandpass_filter(time_stack, Low_freq, High_freq, SR, order=5)
        
        for k in range (0, 101):
            Lap_seq[0][k][i][j] += alpha * time_stack[k]
            
end = time.time()
delay = end - start
print("1st level : ", delay)

# level 1 
start = time.time()
for i in range (0, Lap_seq[1][0].shape[0]):
    for j in range (0, Lap_seq[1][0].shape[1]):
        time_stack = []
        
        for k in range (0, 101):
            time_stack.append(Lap_seq[1][k][i][j])
            
        time_stack = butter.butter_bandpass_filter(time_stack, Low_freq, High_freq, SR, order=5)
        
        for k in range (0, 101):
            Lap_seq[1][k][i][j] += alpha * time_stack[k]
            
end = time.time()
delay = end - start
print("2nd level : ", delay)

# level 2
start = time.time()

for i in range (0, Lap_seq[2][0].shape[0]):
    for j in range (0, Lap_seq[2][0].shape[1]):
        time_stack = []
        
        for k in range (0, 101):
            time_stack.append(Lap_seq[2][k][i][j])
            
        time_stack = butter.butter_bandpass_filter(time_stack, Low_freq, High_freq, SR, order=5)
        
        for k in range (0, 101):
            Lap_seq[2][k][i][j] += time_stack[k]
            
end = time.time()
delay = end - start
print("3nd level : ", delay)

# 변환했던 부동소수점 데이터를 다시 정수로 변환

for i in range(0, 6):
    for j in range(0, 101):
        Lap_seq[i][j] = Lap_seq[i][j].astype('uint8')

# 5. 동영상으로 저장하기
# 영상 스택 쌓기

Rec2= []
for i in range (0, 101):
    Rec2.append( Lap_seq[2][i] + Gau_seq[2][i])

for i in range (0, 101):
    Rec2[i] = cv2.pyrUp(Rec2[i])

Rec1 = []
for i in range (0, 101):
    Rec1.append( Lap_seq[1][i] + Rec2[i] )

for i in range (0, 101):
    Rec1[i] = cv2.pyrUp(Rec1[i])

Rec0 = []
for i in range (0, 101):
    Rec0.append( Lap_seq[0][i] + Rec1[i] )

# Output 파일 생성 - 문제있음
# 코덱문제인지? 동영상이 재생이 안됨

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./output/magnified.avi', fourcc, 30 , (256,640))

for i in range (0, 101):
    out.write(Rec0[i])

out.release()