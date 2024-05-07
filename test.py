import time
for i in range(200): 
    # wait for 1 seconde: 
    # time.sleep(5)
    print(i)


with open('results.txt', 'a') as f:
    f.write('test finished\n')