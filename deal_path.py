import json
import os

# ==================== 配置区域 ====================
# 1. 需要处理的 JSON 文件路径列表
JSON_FILES = [
    # "data/Cog-RailSem19_Joint_fine-tuning_Type_II_train.json",
    # "data/Cog-RailSem19_RailMove_Type_I_train.json",
    # "data/Cog-RailSem19_RailPos_Type_I_train.json",
    # "data/Cog-RailSem19_RailThreat_Type_I_train.json",
    # "data/Cog-RailSem19_RailThreat_Type_II_test.json"
    # "/data/Projects/SongZH/Cog-MRSI/data_after_deal/Type_II/test/QAs/Cog-MRSI_Joint_fine-tuning_Type_II_test.json",
    "/data/Projects/SongZH/Cog-MRSI/data_after_deal/Type_I/test/QAs/Cog-MRSI_RailThreat_Type_I_test.json",
    # 在这里添加更多 JSON 文件路径
]

# 2. 路径替换策略（提供两种常用方式，你可以根据需要选择一种）
# 方式 A: 直接替换掉老路径的某个前缀 (比如把旧服务器路径换成新服务器路径)
# OLD_PREFIX = "/home/tsinghua/TianYL/ZhangQY/railway_track_image_and_label/500_offline2/"
# NEW_PREFIX = "/data/Projects/ZhangQY/Harmonization/output_data/test/Type_I/"
# OLD_PREFIX = "/data/Projects/ZhangQY/Harmonization/output_data/train/Type_II/"
# NEW_PREFIX = "/data/Projects/ZhangQY/Harmonization/output_data_inr/train/Type_I/"
OLD_PREFIX = "/home/tsinghua/TianYL/ZhangQY/railway_track_image_and_label/new_label_offline5/"
NEW_PREFIX = "/data/Projects/SongZH/Cog-MRSI/data_after_deal/Type_I/test/image/"

# 方式 B: 如果你想直接抛弃旧目录，只保留文件名，并拼接到全新的目录下，可以将此开关设为 True
FORCE_NEW_DIR = False 
NEW_DIR_PATH = "/new/absolute/path/to/images/"
# ==================================================

def modify_image_paths():
    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"❌ 未找到文件: {json_path}，跳过该文件。")
            continue
        
        print(f"Processing {json_path}...")
        
        # 读取 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ 文件 {json_path} 格式有误，无法解析为 JSON。")
                continue

        # 确保数据是列表格式（LLaVA 数据集通常是一个大列表）
        if not isinstance(data, list):
            # 如果单条数据直接是一个字典，包装成列表处理
            data = [data]
            is_single_dict = True
        else:
            is_single_dict = False

        modified_count = 0
        
        # 遍历每条数据
        for item in data:
            if "images" in item and isinstance(item["images"], list):
                new_images_list = []
                for old_path in item["images"]:
                    if FORCE_NEW_DIR:
                        # 方式 B: 提取纯文件名，拼接新目录
                        file_name = os.path.basename(old_path)
                        new_path = os.path.join(NEW_DIR_PATH, file_name)
                    else:
                        # 方式 A: 字符串替换前缀
                        if old_path.startswith(OLD_PREFIX):
                            new_path = old_path.replace(OLD_PREFIX, NEW_PREFIX, 1)
                        else:
                            new_path = old_path # 如果不匹配前缀则保持原样
                    
                    new_images_list.append(new_path)
                
                item["images"] = new_images_list
                modified_count += 1

        # 写回文件（这里默认覆盖原文件，若想另存为，可以修改这里的路径，例如 json_path + ".modified"）
        output_path = json_path 
        with open(output_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文和特殊字符不被转义，indent=2 保持好看的缩进格式
            json.dump(data[0] if is_single_dict else data, f, ensure_ascii=False, indent=2)
        
        print(f"✨ 成功更新 {json_path}！共修改了 {modified_count} 条数据的图片路径。")

if __name__ == "__main__":
    modify_image_paths()