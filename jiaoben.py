import os
import shutil

def move_images(source_folder, destination_folder, start_index, end_index):
    """
    将指定范围的 .jpg 图像从源文件夹移动到目标文件夹。
    
    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    :param start_index: 起始索引
    :param end_index: 结束索引
    """
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    
    for i in range(start_index, end_index + 1):
        filename = f"{i}.jpg"  # 构造文件名
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # 检查文件是否存在
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")
        else:
            print(f"File not found: {source_path}")

if __name__ == "__main__":
    # 定义源文件夹和目标文件夹路径
    source_folder = "/Users/wangzhou/leaves/data/images"  # 替换为实际路径
    destination_folder = "/Users/wangzhou/leaves/data/test"  # 替换为实际路径
    
    # 定义范围
    start_index = 18353
    end_index = 27152
    
    # 调用函数
    move_images(source_folder, destination_folder, start_index, end_index)