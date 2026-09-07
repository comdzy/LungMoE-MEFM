import os

# 获取文件夹内所有文件和文件夹的列表
def rename(folder_path, prefix_to_remove, extension):
    for filename in os.listdir(folder_path):
        # 检查文件名是否以指定前缀开始
        if filename.startswith(prefix_to_remove):
            # 构建原文件路径和新的文件名（去掉前缀）
            old_file_path = os.path.join(folder_path, filename)
            new_filename = filename[len(prefix_to_remove):]  # 去掉前缀后的新文件名 
            # 获取文件名前10位（不包括扩展名）
            new_filename = new_filename[:10] + extension
            # 构建新的文件路径
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}"')

# 定义要删除的前缀
prefix_to_remove = '1.3.6.1.4.1.14519.5.2.1.6279.6001.'
 
# 指定文件夹路径
folder_path_jpg = r'D:\CTData\lung\Dataset\jpg'
folder_path_txt = r'D:\CTData\lung\Dataset\txt'
rename(folder_path_jpg, prefix_to_remove, '.jpg')
rename(folder_path_txt, prefix_to_remove, '.txt')