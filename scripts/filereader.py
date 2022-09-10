from fileinput import filename
from filepaths import ACCEL_DIR, GYRO_DIR

accel_files = list(ACCEL_DIR.glob('*'))
gyro_files = list(GYRO_DIR.glob('*'))

"""

file = open('<filename>')
lines = file.readlines()

processedList = []

for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        print('Error at line number: ', i)
"""

