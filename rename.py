import os

folder = '1100831CUT'
count = 1

for file_name in os.listdir(folder):
    #old file_name
    source = file_name

    #new file_name
    destination = "fall" + str(count) + ".mp4"

    #os.rename
    #在Linux或者OS X中，使用绝对路径打开文件的时候应该使用斜杠/，在Windows中的时候，应该使用反斜杠\
    os.rename(f'{folder}\{source}', f'{folder}\{destination}')
    count += 1

print('All file has been rename')
#verify new file name
print('new name are:')
res = os.listdir(folder)
print(res)