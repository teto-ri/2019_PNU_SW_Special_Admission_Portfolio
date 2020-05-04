import numpy as np

new_data=[0,0,0,0]

new_data[0] = int(input('야자여부:'))

new_data[1]= int(input('잠시간:'))

new_data[2]= int(input('게임시간:'))

new_data[3]= int(input('1주일 총 학원시간:'))


new_x = np.array([new_data]).reshape(1, 4)

print(new_x)
