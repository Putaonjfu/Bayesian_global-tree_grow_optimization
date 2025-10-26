import os
import csv
import glob

# --- 配置 ---
# 1. 输入文件夹：存放源 txt 文件的地方
INPUT_DIR = r"E:\PHD\Forests\data\large tropical trees with terrestrial LiDAR\2_异速方程数据"

# 2. 输出文件夹：存放转换后的 csv 文件的地方
OUTPUT_DIR = r"E:\PHD\Forests\data\large tropical trees with terrestrial LiDAR\2_异速方程数据"

# 3. 源文件后缀：可以是 '.txt', '.trees', 或任何你需要转换的文件后缀
#    使用 '*' 可以转换文件夹内所有文件
SOURCE_FILE_EXTENSION = ".*"


# --- 配置结束 ---


def batch_convert_to_csv(input_directory, output_directory, extension):
    """
    批量将指定目录下的文本文件转换为CSV格式。
    假定源文件使用一个或多个空格作为分隔符。
    """
    # 步骤 1: 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_directory, exist_ok=True)
    print(f"输出目录 '{output_directory}' 已准备就绪。")

    # 步骤 2: 查找所有符合条件的源文件
    # 使用 glob 模块可以方便地匹配文件
    search_path = os.path.join(input_directory, f"*{extension}")
    source_files = glob.glob(search_path)

    if not source_files:
        print(f"在目录 '{input_directory}' 中未找到任何后缀为 '{extension}' 的文件。")
        return

    print(f"找到 {len(source_files)} 个文件，准备开始转换...")

    # 步骤 3: 遍历每个文件并进行转换
    for input_filepath in source_files:
        # 从完整路径中获取文件名 (例如 'GUY.h.trees')
        filename = os.path.basename(input_filepath)

        # 构建输出文件的路径，并将后缀改为 .csv
        # os.path.splitext 会将 'GUY.h.trees' 分割成 ('GUY.h', '.trees')
        base_name, _ = os.path.splitext(filename)
        output_filepath = os.path.join(output_directory, f"{base_name}.csv")

        print(f"正在转换: '{filename}' -> '{os.path.basename(output_filepath)}'")

        try:
            with open(input_filepath, 'r', encoding='utf-8') as infile, \
                    open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:

                # 创建一个 CSV writer 对象
                csv_writer = csv.writer(outfile)

                # 逐行读取源文件
                for line in infile:
                    # 去除行首尾的空白字符，然后用 split() 按任意数量的空白符分割
                    columns = line.strip().split()

                    # 将分割后的列写入 CSV 文件
                    csv_writer.writerow(columns)

        except Exception as e:
            print(f"处理文件 '{filename}' 时发生错误: {e}")

    print("\n所有文件转换完成！")


if __name__ == "__main__":
    # 确保输入目录存在
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：输入目录 '{INPUT_DIR}' 不存在。请创建该目录并放入您的文本文件。")
    else:
        batch_convert_to_csv(INPUT_DIR, OUTPUT_DIR, SOURCE_FILE_EXTENSION)