"""
从 raw_data 目录读取每个类别文件，为每个类别编号，
取每个文件前50条数据，生成 CSV 训练文件 (text, label)。
"""
import csv
from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent / "data/raw"
OUTPUT_CSV = Path(__file__).parent.parent / "output/train_data.csv"
OUTPUT_LABEL_MAP = Path(__file__).parent.parent / "output/label_map.csv"
MAX_SAMPLES_PER_CLASS = 100


def main():
    # 收集所有类别文件并排序，确保编号稳定
    category_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    print(RAW_DATA_DIR)

    # 构建类别名 -> 编号的映射
    label_map = {}
    for idx, fpath in enumerate(category_files):
        category_name = fpath.stem  # 文件名去掉 .txt
        label_map[category_name] = idx

    # 打印类别编号映射
    print("=" * 40)
    print("类别编号映射表")
    print("=" * 40)
    for name, label_id in label_map.items():
        print(f"  {label_id:>2d} : {name}")
    print("=" * 40)
    print(f"共 {len(label_map)} 个类别\n")

    # 读取数据并写入 CSV
    total_samples = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for fpath in category_files:
            category_name = fpath.stem
            label_id = label_map[category_name]
            count = 0

            with open(fpath, "r", encoding="utf-8") as rf:
                for line in rf:
                    text = line.strip()
                    if text.endswith("。"):
                        text = text[:-1]
                    if not text:
                        continue
                    writer.writerow([text, label_id])
                    count += 1
                    total_samples += 1
                    if category_name == "other":
                        if count >= 200:
                            break
                    elif count >= MAX_SAMPLES_PER_CLASS:
                        break

            print(f"  类别 [{category_name}] (label={label_id}): 取 {count} 条")

    print(f"\n✅ 训练数据已保存到: {OUTPUT_CSV}")
    print(f"   总样本数: {total_samples}")

    # 同时保存 label_map 到文件，方便后续使用
    with open(OUTPUT_LABEL_MAP, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "category"])
        for name, label_id in label_map.items():
            writer.writerow([label_id, name])
    print(f"   类别映射已保存到: {OUTPUT_LABEL_MAP}")


def unique_sort(file_path: str):
    """读取文件，对语料去重并按长度排序，结果写回原文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去重（保留首次出现顺序）并过滤空行
    unique_lines = list(dict.fromkeys(line.strip() for line in lines))
    unique_lines = [line for line in unique_lines if line]

    # 按长度排序
    unique_lines.sort(key=len)

    # 写回原文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_lines) + "\n")

    print(f"✅ {file_path}: 原始 {len(lines)} 行 → 去重排序后 {len(unique_lines)} 行")


def uniform_raw():
    category_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    print(RAW_DATA_DIR)

    print("=" * 40)
    print("执行语料去重与排序")
    print("=" * 40)
    for fpath in category_files:
        unique_sort(str(fpath))


if __name__ == "__main__":
    # uniform_raw()
    main()
