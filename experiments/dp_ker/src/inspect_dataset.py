from __future__ import annotations

from collections import Counter

from data_loader import RAW_DATA_DIR, load_raw_samples


def main() -> None:
    samples = load_raw_samples()
    category_counter = Counter(sample["filename_category"] for sample in samples)
    print(f"原始数据目录: {RAW_DATA_DIR}")
    print(f"样本数: {len(samples)}")
    print("类别统计:")
    for key, value in sorted(category_counter.items()):
        print(f"  {key}: {value}")
    if samples:
        sample = samples[0]["payload"]
        print("顶层字段:", list(sample.keys()))
        print("input字段:", list(sample["input"].keys()))
        print("output字段:", list(sample["output"].keys()))
        print("details字段:", list(sample["details"].keys()))


if __name__ == "__main__":
    main()

