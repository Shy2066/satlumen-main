"""Convert traces, retaining sample identity and bounded working memory."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_dataset(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("dataset 必须是单个有效目录名")
    return value


def selected_files(directory, manifest=None):
    if manifest is None:
        return [(p.name, None) for p in sorted(directory.iterdir())
                if p.is_file() and re.fullmatch(r"\d+-\d+", p.name)]
    records, seen = [], set()
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "label", "origin_group"} <= set(reader.fieldnames or []):
            raise ValueError("候选清单缺少 filename/label/origin_group")
        for row in reader:
            name = row["filename"]
            if not re.fullmatch(r"\d+-\d+", name) or name in seen:
                raise ValueError(f"非法或重复文件名: {name}")
            if int(name.split("-")[0]) != int(row["label"]) or not row["origin_group"]:
                raise ValueError(f"清单标签或同源组无效: {name}")
            if row.get("needs_timestamp_review", "").lower() == "true" or row.get("duplicate_of"):
                raise ValueError(f"清单包含未排除的异常或重复: {name}")
            seen.add(name)
            records.append((name, row["origin_group"]))
    return sorted(records)


def convert_dataset(dataset, root=ROOT, manifest=None, refresh=False, strict_timestamps=False,
                    remap_labels=False):
    safe_dataset(dataset)
    directory = Path(root) / "dataset" / dataset
    records = selected_files(directory, manifest)
    if not records:
        raise ValueError("没有可转换的轨迹")
    source_classes = sorted({int(name.split("-")[0]) for name, _ in records})
    if source_classes != list(range(max(source_classes) + 1)) and not remap_labels:
        raise ValueError(
            f"原始类别编号不连续：共 {len(source_classes)} 类，编号为 {source_classes}。"
            "若该目录本应包含完整数据，请先补齐缺失类别；"
            "若要独立训练当前子集，可显式使用 --remap-labels，"
            "并将实验配置 num_classes 改为实际类别数。重编号后不能直接复用原分类头。")
    output = Path(root) / "npz_dataset" / dataset
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "data.npz"
    if destination.exists() and not refresh:
        raise FileExistsError(f"拒绝覆盖 {destination}；使用 --refresh")
    ids, groups, labels, hashes, skipped = [], [], [], [], []
    with TemporaryDirectory(prefix=".convert-", dir=output) as temporary:
        samples = np.lib.format.open_memmap(
            Path(temporary) / "X.npy", mode="w+", dtype=np.float32,
            shape=(len(records), 10000, 2))
        try:
            for name, group in tqdm(records, desc=dataset):
                raw = (directory / name).read_bytes()
                if not raw.strip():
                    skipped.append({"sample_id": name, "reason": "empty"})
                    continue
                trace = np.loadtxt(io.BytesIO(raw), delimiter="\t", ndmin=2)
                if trace.shape[1] != 2 or not np.isfinite(trace).all():
                    raise ValueError(f"非有限值或非两列轨迹: {name}")
                if strict_timestamps and (np.any(trace[:, 0] < 0) or
                                          np.any(np.diff(trace[:, 0]) < 0) or
                                          np.any(trace[:, 0] > 1e6)):
                    skipped.append({"sample_id": name, "reason": "timestamp_anomaly"})
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                index = len(ids)
                samples[index] = 0
                trace = trace[:10000].astype(np.float32)
                if not np.isfinite(trace).all():
                    raise ValueError(f"float32 溢出: {name}")
                times = np.trim_zeros(trace[:, 0], "b").copy()
                times[times == 0] = 1e-6
                samples[index, :len(times), 0] = times
                samples[index, :len(trace), 1] = trace[:, 1]
                ids.append(name)
                groups.append(group or "sha256_" + digest)
                hashes.append(digest)
                labels.append(int(name.split("-")[0]))
            if not ids:
                raise ValueError("过滤后没有轨迹")
            accepted_classes = sorted(set(labels))
            mapping = {label: index for index, label in enumerate(accepted_classes)}
            if not remap_labels and accepted_classes != list(range(max(labels) + 1)):
                raise ValueError("过滤后类别编号不连续；请核查缺失类别，或显式使用 --remap-labels")
            original_y = np.asarray(labels, dtype=np.int64)
            encoded_y = np.asarray([mapping[label] for label in labels], dtype=np.int64) if remap_labels else original_y
            staged = Path(temporary) / "data.npz"
            np.savez_compressed(staged, X=samples[:len(ids)], y=encoded_y,
                                **({"original_y": original_y} if remap_labels else {}),
                                sample_ids=np.asarray(ids), origin_groups=np.asarray(groups),
                                trace_sha256=np.asarray(hashes))
            report = {"dataset": dataset, "selected": len(records), "accepted": len(ids),
                      "classes": len(set(labels)), "skipped": skipped,
                      "candidate_manifest": str(Path(manifest).resolve()) if manifest else None,
                      "candidate_manifest_sha256": sha256_file(manifest) if manifest else None,
                      "strict_timestamps": strict_timestamps,
                      "remap_labels": remap_labels,
                      "original_to_model_label": {str(label): int(mapping[label] if remap_labels else label)
                                                  for label in accepted_classes},
                      "data_sha256": sha256_file(staged)}
            staged.replace(destination)
            (output / "conversion_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            samples._mmap.close()
    print(f"{dataset}: accepted={len(ids)} skipped={len(skipped)} classes={len(set(labels))}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tor_fiber")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--strict-timestamps", action="store_true")
    parser.add_argument("--remap-labels", action="store_true",
                        help="按原编号升序重映射为 0..K-1，保留原标签；仅用于独立子集实验")
    args = parser.parse_args()
    convert_dataset(args.dataset, manifest=args.manifest, refresh=args.refresh,
                    strict_timestamps=args.strict_timestamps, remap_labels=args.remap_labels)


if __name__ == "__main__":
    main()
