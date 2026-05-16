"""
将 output/processed 下的文件按 7:1 比例分割成训练集和测试集
- finetune.jsonl -> data/train_qa.jsonl, data/test_qa.jsonl
- pretrain.txt -> data/train.txt, data/test.txt
"""

import os
import random

def split_file(input_path, train_output, test_output, ratio=7, is_jsonl=False):
    """按 ratio:1 分割文件"""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    total = len(lines)
    test_size = total // (ratio + 1)
    train_size = total - test_size

    train_lines = lines[:train_size]
    test_lines = lines[train_size:]

    os.makedirs(os.path.dirname(train_output), exist_ok=True)

    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(test_output, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print(f"{input_path}:")
    print(f"  Total: {total}")
    print(f"  Train: {train_size} -> {train_output}")
    print(f"  Test: {test_size} -> {test_output}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'output', 'processed')
    data_dir = os.path.join(base_dir, 'data')

    # 处理 finetune.jsonl (用于 ft_500.sh)
    finetune_input = os.path.join(processed_dir, 'finetune.jsonl')
    if os.path.exists(finetune_input):
        split_file(
            finetune_input,
            os.path.join(data_dir, 'train_qa.jsonl'),
            os.path.join(data_dir, 'test_qa.jsonl'),
            ratio=7,
            is_jsonl=True
        )
    else:
        print(f"Warning: {finetune_input} not found")

    # 处理 pretrain.txt (用于 pre_500.sh)
    pretrain_input = os.path.join(processed_dir, 'pretrain.txt')
    if os.path.exists(pretrain_input):
        split_file(
            pretrain_input,
            os.path.join(data_dir, 'train.txt'),
            os.path.join(data_dir, 'test.txt'),
            ratio=7,
            is_jsonl=False
        )
    else:
        print(f"Warning: {pretrain_input} not found")


if __name__ == '__main__':
    main()