import os
import re
from tqdm import tqdm

def remove_zero_malignancy(root_dir):
    for mode in ['train', 'test']:
        label_dir = os.path.join(root_dir, 'labels', mode)
        image_dir = os.path.join(root_dir, 'images', mode)
        
        # 获取所有标签文件
        txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        for txt_file in tqdm(txt_files, desc=f"Processing {mode} set"):
            txt_path = os.path.join(label_dir, txt_file)
            img_path = os.path.join(image_dir, txt_file.replace('.txt', '.jpg'))
            
            # 读取并处理标签文件
            with open(txt_path, 'r') as f:
                content = f.read()
            
            # 分割ROI块
            roi_blocks = re.split(r'(ROI \d+:)', content)[1:]
            valid_blocks = []
            
            # 过滤包含Malignancy: 0的块
            for i in range(0, len(roi_blocks)-1, 2):
                header = roi_blocks[i]
                data = roi_blocks[i+1]
                if "Malignancy: 0" not in data:
                    valid_blocks.append(header + data)
            
            # 处理更新内容
            if valid_blocks:
                new_content = '\n\n'.join(valid_blocks)
                with open(txt_path, 'w') as f:
                    f.write(new_content)
            else:
                # 删除空文件
                os.remove(txt_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
                print(f"Removed empty case: {txt_file}")

if __name__ == "__main__":
    remove_zero_malignancy("./dataset")
    print("清理操作完成！")