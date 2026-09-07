import os
 
# 定义文件夹路径
folder1 = r'D:\CTData\lung\Dataset\jpg'
folder2 = r'D:\CTData\lung\Dataset\txt'
import os
 
def compare_folders(folder1, folder2):
    # 获取两个文件夹中所有文件的路径和文件名（不含扩展名）
    files1 = {os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}
 
    # 找出只存在于一个文件夹中的文件名（不含扩展名）
    only_in_folder1 = files1 - files2
    only_in_folder2 = files2 - files1
 
    # 返回结果
    return only_in_folder1, only_in_folder2
 
only_in_folder1, only_in_folder2 = compare_folders(folder1, folder2)

for i in only_in_folder1:
    os.remove(f'{folder1}\{i}.jpg')
    f'{folder1}\{i}.jpg'
    
for i in only_in_folder2:
    os.remove(f'{folder2}\{i}.txt')
    print(f'{folder2}\{i}.txt')