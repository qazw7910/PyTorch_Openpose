import os

folder = 'video/fall'
count = 1

for file_name in os.listdir(folder):
    #old file_name
    source = file_name

    #new file_name
    destination = "fall" + str(count) + ".mp4"

    #os.rename
    #In Linux or OS X，use absolutly path to open the file should use slash: '/' . In Windows,should use reverse slash: '\'
    os.rename(f'{folder}\{source}', f'{folder}\{destination}')
    count += 1

print('All file has been rename')
#verify new file name
print('new name are:')
res = os.listdir(folder)
print(res)