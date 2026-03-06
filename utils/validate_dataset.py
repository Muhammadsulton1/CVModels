import ast
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import yaml
except ImportError:
    yaml = None


ClassId = int
ClassMap = Dict[ClassId, str]
CountMap = Dict[Union[int, str], int]

YOLO_METADATA_FILES = ("data.yaml", "dataset.yaml", "obj.names", "classes.txt", "labels/classes.txt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class DuplicatePair:
    original_path: str
    duplicate_path: str
    hamming_distance: int


@dataclass
class DuplicateSearchResult:
    image_folder: str
    threshold: int
    total_images: int
    processed_images: int
    invalid_images: List[str] = field(default_factory=list)
    pairs: List[DuplicatePair] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class ImbalanceResult:
    dataset_format: str
    path: str
    class_names: Dict[int, str] = field(default_factory=dict)
    class_counts: CountMap = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def render_table(self) -> str:
        if not self.class_counts:
            return "Нет данных. Сначала запустите расчёт дисбаланса."

        total = sum(self.class_counts.values())
        sorted_items = sorted(self.class_counts.items(), key=lambda item: item[1], reverse=True)
        max_name_len = max((len(self._display_name(class_id)) for class_id, _ in sorted_items), default=15)
        max_count = sorted_items[0][1] if sorted_items else 0

        lines = [
            f"{'Class':<{max_name_len}}  {'Count':>7}  {'%':>6}  Distribution",
            "-" * (max_name_len + 40),
        ]

        for class_id, count in sorted_items:
            name = self._display_name(class_id)
            pct = count / total * 100 if total > 0 else 0
            bar_len = int(count / max_count * 20) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"{name:<{max_name_len}}  {count:>7}  {pct:>5.1f}%  {bar}")

        lines.append("-" * (max_name_len + 40))
        lines.append(f"{'TOTAL':<{max_name_len}}  {total:>7}")
        return "\n".join(lines)

    def _display_name(self, class_id: Union[int, str]) -> str:
        if isinstance(class_id, int):
            return self.class_names.get(class_id, f"class_{class_id}")
        return str(class_id)

    def __str__(self) -> str:
        table = self.render_table()
        if not self.warnings:
            return table
        return f"{table}\nПредупреждения:\n- " + "\n- ".join(self.warnings)


@dataclass
class SplitLeakageResult:
    root_path: str
    split_names: List[str]
    summaries: List[str] = field(default_factory=list)
    sample_examples: Dict[str, List[str]] = field(default_factory=dict)
    item_splits: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RemapPlan:
    reference_map: ClassMap
    candidate_map: ClassMap
    label: str
    remap: Dict[int, int] = field(default_factory=dict)
    unchanged_indices: List[int] = field(default_factory=list)
    reordered_matches: List[str] = field(default_factory=list)
    same_index_name_mismatches: List[str] = field(default_factory=list)
    missing_reference: List[str] = field(default_factory=list)
    extra_candidate: List[str] = field(default_factory=list)
    ambiguous_reference_names: List[str] = field(default_factory=list)
    ambiguous_candidate_names: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    safe_to_apply: bool = False
    needs_changes: bool = False


@dataclass
class ValidationReport:
    label: str
    fmt: str
    path: str
    candidate_map: ClassMap
    remap_plan: RemapPlan
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    downstream_warnings: List[str] = field(default_factory=list)
    ok: bool = False


@dataclass
class FixResult:
    fmt: str
    path: str
    changed_files: List[str] = field(default_factory=list)
    backup_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)


@dataclass
class DuplicateActionResult:
    action: str
    pair_index: int
    original_path: str
    duplicate_path: str
    label_path: Optional[str] = None
    moved_to: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    backup_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    reference_format: str
    reference_path: str
    reports: List[ValidationReport] = field(default_factory=list)
    imbalance_results: List[ImbalanceResult] = field(default_factory=list)
    split_results: List[SplitLeakageResult] = field(default_factory=list)
    duplicate_results: List[DuplicateSearchResult] = field(default_factory=list)
    fix_results: List[FixResult] = field(default_factory=list)


def compute_phash(image_path: str) -> Tuple[Optional[int], str]:
    """Вычисляет perceptual hash изображения."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV (cv2) and numpy are required for duplicate search.")
    try:
        img = _read_image(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, image_path

        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(img))
        dct_low_freq = dct[0:8, 0:8]
        median = np.median(dct_low_freq)
        binary_matrix = dct_low_freq > median
        hash_int = 0
        for bit in binary_matrix.flatten():
            hash_int = (hash_int << 1) | int(bit)

        return hash_int, image_path
    except Exception as exc:
        print(f"Error processing {image_path}: {exc}")
        return None, image_path


def find_duplicates(image_folder: str, threshold: int = 5) -> DuplicateSearchResult:
    """Находит дубликаты и возвращает структурированный результат."""
    image_folder = str(_resolve_input_path(image_folder))
    if np is None:
        raise ImportError("numpy is required for duplicate search.")
    paths = [
        str(path)
        for path in Path(image_folder).rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    result = DuplicateSearchResult(
        image_folder=image_folder,
        threshold=threshold,
        total_images=len(paths),
        processed_images=0,
    )

    if not paths:
        return result

    start_time = time.time()
    with Pool(cpu_count()) as pool:
        phash_results = pool.map(compute_phash, paths)

    valid_results = [item for item in phash_results if item[0] is not None]
    result.invalid_images = [item[1] for item in phash_results if item[0] is None]
    result.processed_images = len(valid_results)

    if not valid_results:
        result.elapsed_seconds = time.time() - start_time
        return result

    hashes = np.array([item[0] for item in valid_results], dtype=np.uint64)
    files = np.array([item[1] for item in valid_results])
    num_images = len(hashes)
    hashes_bytes = hashes.view(np.uint8).reshape(num_images, 8)
    hashes_bits = np.unpackbits(hashes_bytes, axis=1)
    duplicates_to_remove = set()

    for idx in range(num_images):
        if idx in duplicates_to_remove:
            continue

        current_bits = hashes_bits[idx]
        slice_bits = hashes_bits[idx + 1:]
        if slice_bits.shape[0] == 0:
            break

        distances = np.bitwise_xor(slice_bits, current_bits).sum(axis=1)
        local_indices = np.where(distances <= threshold)[0]
        global_indices = local_indices + (idx + 1)

        for dup_idx, distance in zip(global_indices, distances[local_indices]):
            if dup_idx in duplicates_to_remove:
                continue
            duplicates_to_remove.add(dup_idx)
            result.pairs.append(
                DuplicatePair(
                    original_path=str(files[idx]),
                    duplicate_path=str(files[dup_idx]),
                    hamming_distance=int(distance),
                )
            )

    result.elapsed_seconds = time.time() - start_time
    return result


def find_duplicates_pairs(image_folder: str, threshold: int = 5) -> List[Tuple[str, str, int]]:
    """Backwards-compatible helper with the old return type."""
    result = find_duplicates(image_folder=image_folder, threshold=threshold)
    return [(pair.original_path, pair.duplicate_path, pair.hamming_distance) for pair in result.pairs]


def print_duplicate_report(result: DuplicateSearchResult) -> None:
    print(f"Найдено изображений: {result.total_images}")
    print(f"Обработано изображений: {result.processed_images}")
    if result.invalid_images:
        print(f"Не удалось прочитать изображений: {len(result.invalid_images)}")
    print(f"Найдено пар дубликатов: {len(result.pairs)}")
    print(f"Время: {result.elapsed_seconds:.2f} сек.")


def print_duplicate_pair_summary(pair: DuplicatePair, pair_index: int) -> None:
    print(f"\n[Пара дубликатов #{pair_index}]")
    print(f"Оставляем: {pair.original_path}")
    print(f"Дубликат:  {pair.duplicate_path}")
    print(f"Hamming distance: {pair.hamming_distance}")


def visualize_duplicates(pairs: Sequence[Union[DuplicatePair, Tuple[str, str, int]]], num_to_show: int = 5) -> None:
    """Отрисовывает пары изображений side-by-side. Импортирует matplotlib только при вызове."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for duplicate visualization.")
    if not pairs:
        print("Нет дубликатов для отображения.")
        return

    import matplotlib.pyplot as plt

    count = min(len(pairs), num_to_show)
    print(f"Отображение первых {count} пар...")

    for index in range(count):
        pair = pairs[index]
        if isinstance(pair, DuplicatePair):
            orig_path, dup_path, dist = pair.original_path, pair.duplicate_path, pair.hamming_distance
        else:
            orig_path, dup_path, dist = pair

        img1 = _read_image(orig_path, cv2.IMREAD_COLOR)
        img2 = _read_image(dup_path, cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            continue

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Hamming Distance: {dist} (Pair #{index + 1})", fontsize=16)

        axes[0].imshow(img1)
        axes[0].set_title("Оригинал")
        axes[0].axis("off")
        axes[0].text(0, -10, os.path.basename(orig_path), fontsize=8, color="blue")

        axes[1].imshow(img2)
        axes[1].set_title("Дубликат")
        axes[1].axis("off")
        axes[1].text(0, -10, os.path.basename(dup_path), fontsize=8, color="red")
        plt.tight_layout()
        plt.show()


def apply_duplicate_action(
    pair: DuplicatePair,
    pair_index: int,
    action: str,
    move_root: Optional[Path] = None,
    create_backup: bool = True,
) -> DuplicateActionResult:
    duplicate_path = _resolve_input_path(pair.duplicate_path)
    label_path = _find_paired_label_file(duplicate_path)
    result = DuplicateActionResult(
        action=action,
        pair_index=pair_index,
        original_path=pair.original_path,
        duplicate_path=str(duplicate_path),
        label_path=str(label_path) if label_path is not None else None,
    )

    if action == "skip":
        return result

    if action == "delete":
        files_to_delete = [duplicate_path]
        if label_path is not None:
            files_to_delete.append(label_path)
        for file_path in files_to_delete:
            if not _path_exists(file_path):
                result.warnings.append(f"Файл уже отсутствует: {file_path}")
                continue
            if create_backup:
                result.backup_files.append(str(_backup_file(file_path)))
            _delete_file(file_path)
            result.deleted_files.append(str(file_path))
        return result

    if action == "move":
        if move_root is None:
            raise ValueError("Для действия 'move' требуется move_root")

        moved_paths = [duplicate_path]
        if label_path is not None:
            moved_paths.append(label_path)
        for file_path in moved_paths:
            if not _path_exists(file_path):
                result.warnings.append(f"Файл уже отсутствует: {file_path}")
                continue
            destination = _build_duplicate_move_destination(file_path, move_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(_normalize_windows_long_path(file_path), _normalize_windows_long_path(destination))
            result.moved_to.append(str(destination))
        return result

    raise ValueError(f"Неизвестное действие для дубликата: {action}")


def run_duplicate_autofix(
    duplicate_result: DuplicateSearchResult,
    dataset_format: str = "image_folder",
    annotation_path: Optional[str] = None,
    create_backup: bool = True,
) -> FixResult:
    fix_result = FixResult(fmt=f"duplicates_{dataset_format}", path=duplicate_result.image_folder)
    if not duplicate_result.pairs:
        return fix_result

    print("\n[Автообработка дубликатов]")
    show_visual = _ask_yes_no("Показать несколько пар дубликатов перед исправлением?", default=False)
    sample_pairs = duplicate_result.pairs[: min(3, len(duplicate_result.pairs))]
    if show_visual:
        for pair_index, pair in enumerate(sample_pairs, start=1):
            print_duplicate_pair_summary(pair, pair_index)
        try:
            visualize_duplicates(sample_pairs, num_to_show=len(sample_pairs))
        except ImportError as exc:
            print(f"Не удалось показать пары визуально: {exc}")

    action = _ask_duplicate_dataset_action()
    move_root: Optional[Path] = None
    if action == "skip":
        fix_result.skipped_files.extend(pair.duplicate_path for pair in duplicate_result.pairs)
        return fix_result

    if action == "move":
        default_move_root = _resolve_input_path(duplicate_result.image_folder).parent / "duplicates_review"
        move_root = _resolve_input_path(_ask_text("Куда переносить дубликаты", default=str(default_move_root)))

    processed_duplicate_paths: List[Path] = []
    for pair_index, pair in enumerate(duplicate_result.pairs, start=1):
        action_result = apply_duplicate_action(
            pair=pair,
            pair_index=pair_index,
            action=action,
            move_root=move_root,
            create_backup=create_backup,
        )

        if action_result.deleted_files:
            fix_result.changed_files.extend(action_result.deleted_files)
        if action_result.backup_files:
            fix_result.backup_files.extend(action_result.backup_files)
        if action_result.moved_to:
            fix_result.changed_files.extend(action_result.moved_to)
        if action_result.warnings:
            fix_result.warnings.extend(action_result.warnings)
        processed_duplicate_paths.append(_resolve_input_path(pair.duplicate_path))

    if dataset_format == "coco" and annotation_path is not None and action in {"delete", "move"}:
        coco_fix = _apply_coco_duplicate_annotations_fix(
            annotation_path=annotation_path,
            duplicate_paths=processed_duplicate_paths,
            create_backup=create_backup,
        )
        fix_result.changed_files.extend(coco_fix.changed_files)
        fix_result.backup_files.extend(coco_fix.backup_files)
        fix_result.warnings.extend(coco_fix.warnings)
        fix_result.skipped_files.extend(coco_fix.skipped_files)

    return fix_result


def calculate_imbalance(path: str, dataset_format: str) -> ImbalanceResult:
    path = str(_resolve_input_path(path))
    dataset_format = dataset_format.lower()
    if dataset_format not in ("coco", "yolo", "image_folder"):
        raise ValueError(f"unknown dataset format {dataset_format}")

    result = ImbalanceResult(dataset_format=dataset_format, path=path)
    processors = {
        "coco": _calc_imbalance_coco,
        "yolo": _calc_imbalance_yolo,
        "image_folder": _calc_imbalance_image_folder,
    }
    processors[dataset_format](result)
    total = sum(result.class_counts.values())
    if total > 0:
        for class_id, count in result.class_counts.items():
            pct = count / total * 100
            if pct < 10:
                name = result.class_names.get(class_id, str(class_id))
                result.warnings.append(f"У класса {name} сильный дисбаланс: {pct:.2f}%")
    return result


def _calc_imbalance_image_folder(result: ImbalanceResult) -> None:
    for child in sorted(Path(result.path).iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            result.class_counts[child.name] = len([item for item in child.iterdir() if item.is_file()])


def _calc_imbalance_yolo(result: ImbalanceResult) -> None:
    root = Path(result.path)
    metadata = _discover_yolo_metadata(root)
    result.class_names = metadata["class_map"]

    for txt_path in _iter_yolo_label_files(root, metadata["metadata_files"]):
        try:
            lines = _read_text_file(txt_path).splitlines()
        except FileNotFoundError:
            result.warnings.append(f"Пропущен отсутствующий label-файл: {txt_path}")
            continue
        except UnicodeDecodeError:
            result.warnings.append(f"Пропущен не-текстовый файл: {txt_path}")
            continue
        except OSError as exc:
            result.warnings.append(f"Пропущен недоступный файл {txt_path}: {exc}")
            continue

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                class_idx = int(parts[0])
            except (ValueError, IndexError):
                result.warnings.append(f"Некорректная строка в {txt_path}: {stripped}")
                continue
            result.class_counts[class_idx] = result.class_counts.get(class_idx, 0) + 1


def _calc_imbalance_coco(result: ImbalanceResult) -> None:
    with open(result.path, encoding="utf-8") as file:
        data = json.load(file)

    for category in data.get("categories", []):
        category_id = int(category["id"])
        result.class_names[category_id] = category["name"]
        result.class_counts[category_id] = 0

    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id not in result.class_counts:
            raise ValueError(f"unknown category {category_id}")
        result.class_counts[category_id] += 1


class CalculateImbalance:
    """Совместимость со старым API + новый структурированный результат."""

    def __init__(self, path: str):
        self.path = path
        self.result: Optional[ImbalanceResult] = None

    def __call__(self, name: str) -> "CalculateImbalance":
        self.result = calculate_imbalance(self.path, name)
        return self

    @property
    def cls_names(self) -> Dict[int, str]:
        return {} if self.result is None else self.result.class_names

    @property
    def cls_imbalance(self) -> CountMap:
        return {} if self.result is None else self.result.class_counts

    def __str__(self) -> str:
        if self.result is None:
            return "No data. Call calc_imbalance() first."
        return str(self.result)


def analyze_split_leakage(path: str) -> SplitLeakageResult:
    root = _resolve_input_path(path)
    split_dirs = sorted([item for item in root.iterdir() if item.is_dir()])

    if not 2 <= len(split_dirs) <= 3:
        raise ValueError(f"Ожидается 2 или 3 сплита, найдено: {len(split_dirs)}")

    split_files = {item.name: _collect_split_leakage_items(item) for item in split_dirs}
    result = SplitLeakageResult(root_path=path, split_names=[item.name for item in split_dirs])

    item_splits: Dict[str, List[str]] = {}
    for split_name, files in split_files.items():
        for relative_path in files:
            item_splits.setdefault(relative_path, []).append(split_name)
    result.item_splits = {
        relative_path: sorted(split_names)
        for relative_path, split_names in item_splits.items()
        if len(split_names) > 1
    }

    for (name_a, files_a), (name_b, files_b) in combinations(split_files.items(), 2):
        intersection = sorted(files_a & files_b)
        if not intersection:
            continue
        key = f"{name_a} & {name_b}"
        result.summaries.append(f"{key}: {len(intersection)} файлов")
        result.sample_examples[key] = intersection[:3]

    return result


def _collect_split_leakage_items(split_dir: Path) -> set[str]:
    images_dir = split_dir / "images"
    if images_dir.exists() and images_dir.is_dir():
        return {
            file_path.relative_to(images_dir).as_posix()
            for file_path in images_dir.rglob("*")
            if file_path.is_file()
        }

    return {
        file_path.relative_to(split_dir).as_posix()
        for file_path in split_dir.rglob("*")
        if file_path.is_file()
    }


def _build_split_image_path(root: Path, split_name: str, relative_path: str) -> Path:
    split_dir = root / split_name
    images_dir = split_dir / "images"
    if images_dir.exists():
        return images_dir / relative_path
    return split_dir / relative_path


def _build_split_label_path(root: Path, split_name: str, relative_path: str) -> Optional[Path]:
    image_path = _build_split_image_path(root, split_name, relative_path)
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() != "images":
            continue
        label_parts = parts.copy()
        label_parts[index] = "labels"
        candidate = Path(*label_parts).with_suffix(".txt")
        if _path_exists(candidate):
            return candidate
        return None

    candidate = image_path.with_suffix(".txt")
    if _path_exists(candidate):
        return candidate
    return None


def _build_split_leakage_move_destination(root: Path, split_name: str, source_path: Path, move_root: Path) -> Path:
    root = _resolve_input_path(root)
    source_path = _resolve_input_path(source_path)
    move_root = _resolve_input_path(move_root)
    relative = source_path.relative_to(root / split_name)
    return move_root / split_name / relative


def check_split_leakage(path: str) -> List[str]:
    """Backwards-compatible helper with the old return type."""
    result = analyze_split_leakage(path)
    for key, examples in result.sample_examples.items():
        print(f"вот такие данные попали в разные сплиты: [{key}] Примеры: {examples}")
    return result.summaries


def print_split_leakage_report(result: SplitLeakageResult) -> None:
    if not result.summaries:
        print("Leakage между сплитами не найден.")
        return
    for summary in result.summaries:
        print(summary)
    for key, examples in result.sample_examples.items():
        print(f"Примеры для {key}: {examples}")


def run_split_leakage_autofix(
    split_result: SplitLeakageResult,
    create_backup: bool = True,
) -> FixResult:
    fix_result = FixResult(fmt="split_leakage", path=split_result.root_path)
    if not split_result.item_splits:
        return fix_result

    print("\n[Автообработка пересечений сплитов]")
    print("Нужно один раз выбрать, в каком сплите оставлять конфликтные файлы.")
    print("Во всех остальных сплитах копии будут обработаны одинаково.")

    move_root: Optional[Path] = None
    root = _resolve_input_path(split_result.root_path)
    keep_split, action = _ask_split_leakage_batch_action(split_result.split_names)

    if action == "skip" or keep_split is None:
        fix_result.skipped_files.extend(sorted(split_result.item_splits))
        return fix_result

    if action == "move":
        default_move_root = root / "leakage_review"
        move_root = _resolve_input_path(_ask_text("Куда переносить конфликтные файлы", default=str(default_move_root)))

    for item_index, (relative_path, split_names) in enumerate(sorted(split_result.item_splits.items()), start=1):
        print(f"\n[Конфликт #{item_index}] {relative_path}")
        for idx, split_name in enumerate(split_names, start=1):
            image_path = _build_split_image_path(root, split_name, relative_path)
            label_path = _build_split_label_path(root, split_name, relative_path)
            print(f"  {idx}. {split_name}")
            print(f"     image: {image_path}")
            if label_path is not None:
                print(f"     label: {label_path}")

        for split_name in split_names:
            if split_name == keep_split:
                continue
            image_path = _build_split_image_path(root, split_name, relative_path)
            label_path = _build_split_label_path(root, split_name, relative_path)
            affected_paths = [image_path]
            if label_path is not None:
                affected_paths.append(label_path)

            for file_path in affected_paths:
                if not _path_exists(file_path):
                    fix_result.warnings.append(f"Файл уже отсутствует: {file_path}")
                    continue
                if action == "delete":
                    if create_backup:
                        fix_result.backup_files.append(str(_backup_file(file_path)))
                    _delete_file(file_path)
                    fix_result.changed_files.append(str(file_path))
                elif action == "move":
                    destination = _build_split_leakage_move_destination(root, split_name, file_path, move_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(_normalize_windows_long_path(file_path), _normalize_windows_long_path(destination))
                    fix_result.changed_files.append(str(destination))

    return fix_result


class DatasetClassValidator:
    """
    Validates class index consistency across dataset versions.
    Supported formats: COCO JSON, YOLO (data.yaml / classes.txt), ImageFolder.
    """

    SUPPORTED_FORMATS = ("coco", "yolo", "image_folder")

    def __init__(self, reference_format: str, reference_path: str):
        self._check_format(reference_format)
        self.reference_format = reference_format
        self.reference_path = reference_path
        self.reference_map = self._parse(reference_format, reference_path)

    def validate(self, fmt: str, path: str, label: Optional[str] = None) -> ValidationReport:
        self._check_format(fmt)
        label = label or path
        candidate_map = self._parse(fmt, path)
        remap_plan = self._compare(self.reference_map, candidate_map, label)

        errors = list(remap_plan.errors)
        warnings = list(remap_plan.warnings)
        warnings.extend(self._format_alignment_warnings(fmt, remap_plan))
        downstream_warnings = self._collect_downstream_warnings()

        report = ValidationReport(
            label=label,
            fmt=fmt,
            path=path,
            candidate_map=candidate_map,
            remap_plan=remap_plan,
            errors=errors,
            warnings=warnings,
            downstream_warnings=downstream_warnings,
            ok=len(errors) == 0,
        )
        self._print_report(report)
        return report

    def validate_many(self, candidates: List[dict]) -> List[ValidationReport]:
        return [
            self.validate(fmt=item["format"], path=item["path"], label=item.get("label"))
            for item in candidates
        ]

    def apply_fix(self, fmt: str, path: str, report: ValidationReport, create_backup: bool = True) -> FixResult:
        self._check_format(fmt)
        if not report.remap_plan.safe_to_apply:
            raise ValueError(f"Автоисправление невозможно для {report.label}: mapping небезопасен")
        if not report.remap_plan.needs_changes:
            return FixResult(fmt=fmt, path=path, warnings=["Изменения не требуются."])

        if fmt == "yolo":
            return self._apply_yolo_fix(path=path, remap_plan=report.remap_plan, create_backup=create_backup)
        if fmt == "coco":
            return self._apply_coco_fix(path=path, remap_plan=report.remap_plan, create_backup=create_backup)
        raise ValueError(f"Автоисправление пока не поддерживается для формата {fmt}")

    def _parse(self, fmt: str, path: str) -> ClassMap:
        parsers = {
            "coco": self._parse_coco_json,
            "yolo": self._parse_yolo,
            "image_folder": self._parse_image_folder,
        }
        return parsers[fmt](path)

    @staticmethod
    def _parse_coco_json(path: str) -> ClassMap:
        path = str(_resolve_input_path(path))
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}

    @staticmethod
    def _parse_yolo(dataset_root: str) -> ClassMap:
        dataset_root = str(_resolve_input_path(dataset_root))
        if yaml is None:
            txt_metadata = _discover_yolo_text_metadata(Path(dataset_root))
            if txt_metadata:
                return txt_metadata
        metadata = _discover_yolo_metadata(Path(dataset_root))
        if metadata["class_map"]:
            return metadata["class_map"]
        raise FileNotFoundError(_build_yolo_metadata_error(dataset_root))

    @staticmethod
    def _parse_image_folder(dataset_root: str) -> ClassMap:
        root = _resolve_input_path(dataset_root)
        idx_file = root / "class_to_idx.json"
        if idx_file.exists():
            with open(idx_file, encoding="utf-8") as file:
                mapping = json.load(file)
            return {int(value): key for key, value in mapping.items()}
        classes = sorted(
            item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
        )
        return {index: name for index, name in enumerate(classes)}

    @staticmethod
    def _compare(reference: ClassMap, candidate: ClassMap, label: str) -> RemapPlan:
        plan = RemapPlan(reference_map=dict(reference), candidate_map=dict(candidate), label=label)

        reference_name_to_ids = _invert_class_map(reference)
        candidate_name_to_ids = _invert_class_map(candidate)

        for name, ids in sorted(reference_name_to_ids.items()):
            if len(ids) > 1:
                plan.ambiguous_reference_names.append(
                    f"Название '{name}' встречается несколько раз в reference: {ids}"
                )
        for name, ids in sorted(candidate_name_to_ids.items()):
            if len(ids) > 1:
                plan.ambiguous_candidate_names.append(
                    f"Название '{name}' встречается несколько раз в {label}: {ids}"
                )

        plan.errors.extend(plan.ambiguous_reference_names)
        plan.errors.extend(plan.ambiguous_candidate_names)

        reference_names = set(reference_name_to_ids)
        candidate_names = set(candidate_name_to_ids)

        for missing_name in sorted(reference_names - candidate_names):
            ref_idx = reference_name_to_ids[missing_name][0]
            plan.missing_reference.append(
                f"  [MISSING] class '{missing_name}' (reference index {ref_idx}) absent in {label}"
            )

        for extra_name in sorted(candidate_names - reference_names):
            candidate_idx = candidate_name_to_ids[extra_name][0]
            plan.extra_candidate.append(
                f"  [NEW] class '{extra_name}' (candidate index {candidate_idx}) exists only in {label}"
            )

        plan.errors.extend(plan.missing_reference)
        plan.errors.extend(plan.extra_candidate)

        shared_names = sorted(reference_names & candidate_names)
        for name in shared_names:
            ref_idx = reference_name_to_ids[name][0]
            cand_idx = candidate_name_to_ids[name][0]
            plan.remap[cand_idx] = ref_idx
            if cand_idx == ref_idx:
                plan.unchanged_indices.append(cand_idx)
            else:
                plan.reordered_matches.append(
                    f"  [REMAP] '{name}' should move {cand_idx} -> {ref_idx} in {label}"
                )

        for idx in sorted(set(reference) & set(candidate)):
            ref_name = reference[idx]
            cand_name = candidate[idx]
            if ref_name == cand_name:
                continue
            if ref_name in candidate_name_to_ids and cand_name in reference_name_to_ids:
                plan.same_index_name_mismatches.append(
                    f"  [SWAP] index {idx}: reference='{ref_name}' vs {label}='{cand_name}'"
                )
            else:
                plan.same_index_name_mismatches.append(
                    f"  [MISMATCH] index {idx}: reference='{ref_name}' vs {label}='{cand_name}'"
                )

        plan.warnings.extend(plan.same_index_name_mismatches)
        plan.warnings.extend(plan.reordered_matches)

        plan.needs_changes = any(old_idx != new_idx for old_idx, new_idx in plan.remap.items())
        plan.safe_to_apply = not plan.errors and len(plan.remap) == len(candidate)
        if not plan.safe_to_apply and not plan.errors:
            plan.errors.append(
                f"  [UNSAFE] cannot build full remap for {label}: partial mapping {plan.remap}"
            )
        return plan

    @staticmethod
    def _print_report(report: ValidationReport) -> None:
        status = "OK" if report.ok else "FAILED"
        print(f"\n{'=' * 55}")
        print(f"  {report.label} ({report.fmt}) -> {status}")
        print(f"{'=' * 55}")
        print(f"Эталонная карта классов: {report.remap_plan.reference_map}")
        print(f"Карта классов датасета: {report.candidate_map}")
        for message in report.errors + report.warnings + report.downstream_warnings:
            print(message)
        if report.ok and not report.warnings and not report.downstream_warnings:
            print("  Все индексы полностью совпадают с эталоном.")
        if report.remap_plan.safe_to_apply and report.remap_plan.needs_changes:
            print("  Доступно автоисправление: remap можно применить безопасно.")

    def _format_alignment_warnings(self, fmt: str, remap_plan: RemapPlan) -> List[str]:
        warnings: List[str] = []
        target_ids = sorted(remap_plan.reference_map)
        if fmt == "yolo" and target_ids:
            expected_ids = list(range(min(target_ids), min(target_ids) + len(target_ids)))
            if target_ids != expected_ids or min(target_ids) != 0:
                warnings.append(
                    "  [WARN] target YOLO ids are not contiguous from 0; labels will still be rewritten,"
                    " but some YOLO tools expect 0..N-1."
                )
        return warnings

    def _collect_downstream_warnings(self) -> List[str]:
        repo_root = Path(__file__).resolve().parents[1]
        warnings: List[str] = []
        warnings.extend(_inspect_train_config(repo_root, self.reference_map))
        warnings.extend(_inspect_inference_mapping(repo_root, self.reference_map))
        return warnings

    def _apply_yolo_fix(self, path: str, remap_plan: RemapPlan, create_backup: bool = True) -> FixResult:
        root = _resolve_input_path(path)
        metadata = _discover_yolo_metadata(root)
        result = FixResult(fmt="yolo", path=path)

        for label_path in _iter_yolo_label_files(root, metadata["metadata_files"]):
            updated_lines, should_write, file_warnings = _rewrite_yolo_label_lines(
                label_path=label_path,
                remap=remap_plan.remap,
            )
            result.warnings.extend(file_warnings)
            if not should_write:
                if file_warnings:
                    result.skipped_files.append(str(label_path))
                continue
            if create_backup:
                result.backup_files.append(str(_backup_file(label_path)))
            label_path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
            result.changed_files.append(str(label_path))

        metadata_result = _rewrite_yolo_metadata_files(
            root=root,
            metadata=metadata,
            target_map=remap_plan.reference_map,
            create_backup=create_backup,
        )
        result.changed_files.extend(metadata_result.changed_files)
        result.backup_files.extend(metadata_result.backup_files)
        result.warnings.extend(metadata_result.warnings)
        result.skipped_files.extend(metadata_result.skipped_files)
        return result

    def _apply_coco_fix(self, path: str, remap_plan: RemapPlan, create_backup: bool = True) -> FixResult:
        coco_path = _resolve_input_path(path)
        result = FixResult(fmt="coco", path=path)

        with open(coco_path, encoding="utf-8") as file:
            data = json.load(file)

        category_by_old_id = {
            int(category["id"]): dict(category)
            for category in data.get("categories", [])
        }
        new_categories = []
        for new_id, name in sorted(remap_plan.reference_map.items()):
            old_ids = [old_id for old_id, target_id in remap_plan.remap.items() if target_id == new_id]
            category_payload = category_by_old_id.get(old_ids[0], {}) if old_ids else {}
            category_payload["id"] = new_id
            category_payload["name"] = name
            new_categories.append(category_payload)

        missing_annotations = []
        for annotation in data.get("annotations", []):
            old_id = int(annotation["category_id"])
            if old_id not in remap_plan.remap:
                missing_annotations.append(old_id)
                continue
            annotation["category_id"] = remap_plan.remap[old_id]

        if missing_annotations:
            unique_missing = sorted(set(missing_annotations))
            raise ValueError(f"COCO fix aborted: annotations contain unmapped category ids {unique_missing}")

        data["categories"] = new_categories
        if create_backup:
            result.backup_files.append(str(_backup_file(coco_path)))
        coco_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        result.changed_files.append(str(coco_path))
        return result

    def _check_format(self, fmt: str) -> None:
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unknown format '{fmt}'. Choose from: {self.SUPPORTED_FORMATS}")


def run_validation_pipeline(
    reference_format: Optional[str] = None,
    reference_path: Optional[str] = None,
    candidates: Optional[List[Dict[str, str]]] = None,
    run_imbalance_checks: bool = False,
    imbalance_targets: Optional[List[Dict[str, str]]] = None,
    run_split_checks: bool = False,
    split_targets: Optional[List[str]] = None,
    run_duplicate_checks: bool = False,
    duplicate_targets: Optional[List[Dict[str, Any]]] = None,
    interactive: bool = True,
    create_backup: bool = True,
) -> PipelineResult:
    """
    Единый интерактивный pipeline для валидации и автоисправления датасетов.
    """
    if reference_format is None:
        if not interactive:
            raise ValueError("reference_format is required when interactive=False")
        reference_format = _ask_format("Reference format")
    if reference_path is None:
        if not interactive:
            raise ValueError("reference_path is required when interactive=False")
        reference_path = _ask_text("Reference path")
    validator, reference_format, reference_path = _create_validator_with_retry(
        reference_format=reference_format,
        reference_path=reference_path,
        interactive=interactive,
    )
    result = PipelineResult(reference_format=reference_format, reference_path=reference_path)

    candidate_items = candidates or []
    if interactive and not candidate_items:
        candidate_items = _collect_candidates_from_input()

    for candidate in candidate_items:
        try:
            report = validator.validate(
                fmt=candidate["format"],
                path=candidate["path"],
                label=candidate.get("label"),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n[Ошибка валидации] {candidate.get('label', candidate['path'])}: {exc}")
            if interactive:
                continue
            raise
        result.reports.append(report)

        if interactive:
            _print_remap_plan(report.remap_plan)

        if candidate["format"] in {"yolo", "coco"} and report.remap_plan.safe_to_apply and report.remap_plan.needs_changes:
            should_apply = bool(candidate.get("apply_fix", False))
            if interactive:
                should_apply = _ask_yes_no(
                    f"Apply auto-fix for {report.label} ({candidate['format']})?", default=False
                )
            if should_apply:
                fix_result = validator.apply_fix(
                    fmt=candidate["format"],
                    path=candidate["path"],
                    report=report,
                    create_backup=create_backup,
                )
                result.fix_results.append(fix_result)
                print_fix_report(fix_result)

    imbalance_items = imbalance_targets or []
    if interactive and run_imbalance_checks and not imbalance_items:
        imbalance_items = _collect_dataset_targets("imbalance")
    for item in imbalance_items:
        imbalance_result = calculate_imbalance(path=item["path"], dataset_format=item["format"])
        result.imbalance_results.append(imbalance_result)
        print("\n[Дисбаланс]")
        print(imbalance_result)

    split_items = split_targets or []
    if interactive and run_split_checks and not split_items:
        split_items = _collect_plain_paths("split leakage root")
    for split_path in split_items:
        split_result = analyze_split_leakage(split_path)
        result.split_results.append(split_result)
        print("\n[Пересечение сплитов]")
        print_split_leakage_report(split_result)
        if interactive and split_result.item_splits:
            should_fix = _ask_yes_no("Запустить автообработку пересечений сплитов?", default=False)
            if should_fix:
                fix_result = run_split_leakage_autofix(
                    split_result=split_result,
                    create_backup=create_backup,
                )
                result.fix_results.append(fix_result)
                print_fix_report(fix_result)

    duplicate_items = duplicate_targets or []
    if interactive and run_duplicate_checks and not duplicate_items:
        duplicate_items = _collect_duplicate_targets()
    for item in duplicate_items:
        duplicate_result = find_duplicates(
            image_folder=item["path"],
            threshold=int(item.get("threshold", 5)),
        )
        result.duplicate_results.append(duplicate_result)
        print("\n[Дубликаты]")
        print_duplicate_report(duplicate_result)
        if interactive and duplicate_result.pairs:
            should_fix = _ask_yes_no("Запустить автообработку найденных дубликатов?", default=False)
            if should_fix:
                fix_result = run_duplicate_autofix(
                    duplicate_result=duplicate_result,
                    dataset_format=item.get("format", "image_folder"),
                    annotation_path=item.get("annotation_path"),
                    create_backup=create_backup,
                )
                result.fix_results.append(fix_result)
                print_fix_report(fix_result)

    return result


def _discover_yolo_metadata(root: Path) -> Dict[str, Any]:
    metadata_files = set()
    class_map: ClassMap = {}
    yaml_path_used: Optional[Path] = None
    yaml_names_kind: Optional[str] = None

    if yaml is not None:
        for yaml_name in ("data.yaml", "dataset.yaml"):
            yaml_path = root / yaml_name
            if not yaml_path.exists():
                continue
            metadata_files.add(yaml_path)
            with open(yaml_path, encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            names = config.get("names", [])
            if isinstance(names, list):
                class_map = {index: str(name) for index, name in enumerate(names)}
                yaml_names_kind = "list"
            elif isinstance(names, dict):
                class_map = {int(key): str(value) for key, value in names.items()}
                yaml_names_kind = "dict"
            yaml_path_used = yaml_path
            if class_map:
                break

    txt_path_used: Optional[Path] = None
    if not class_map:
        text_metadata = _discover_yolo_text_metadata(root)
        if text_metadata:
            class_map = text_metadata
            for txt_name in ("obj.names", "classes.txt", "labels/classes.txt"):
                txt_path = root / txt_name
                if not txt_path.exists():
                    continue
                metadata_files.add(txt_path)
                txt_path_used = txt_path
                break

    for txt_name in ("obj.names", "classes.txt", "labels/classes.txt"):
        txt_path = root / txt_name
        if txt_path.exists():
            metadata_files.add(txt_path)

    return {
        "class_map": class_map,
        "metadata_files": metadata_files,
        "yaml_path": yaml_path_used,
        "yaml_names_kind": yaml_names_kind,
        "txt_path": txt_path_used,
    }


def _discover_yolo_text_metadata(root: Path) -> ClassMap:
    for txt_name in ("obj.names", "classes.txt", "labels/classes.txt"):
        txt_path = root / txt_name
        if not txt_path.exists():
            continue
        lines = [line.strip() for line in _read_text_file(txt_path).splitlines() if line.strip()]
        return {index: line for index, line in enumerate(lines)}
    return {}


def _iter_yolo_label_files(root: Path, metadata_files: Sequence[Path]) -> List[Path]:
    metadata_set = {path.resolve() for path in metadata_files}

    label_files = [
        path for path in sorted(root.rglob("*.txt"))
        if _is_yolo_label_file(path, metadata_set)
    ]

    labels_dir_files = [path for path in label_files if "labels" in {part.lower() for part in path.parts}]
    if labels_dir_files:
        return labels_dir_files
    return label_files


def _is_yolo_label_file(path: Path, metadata_files: set[Path]) -> bool:
    resolved = path.resolve()
    if resolved in metadata_files:
        return False
    if ".bak." in path.name:
        return False
    if path.name.lower() in {"train.txt", "val.txt", "test.txt"}:
        return False
    return True


def _find_paired_label_file(image_path: Path) -> Optional[Path]:
    image_path = _resolve_input_path(image_path)
    parts = list(image_path.parts)

    for index, part in enumerate(parts):
        if part.lower() != "images":
            continue
        label_parts = parts.copy()
        label_parts[index] = "labels"
        candidate = Path(*label_parts).with_suffix(".txt")
        if _path_exists(candidate):
            return candidate

    sibling_candidate = image_path.with_suffix(".txt")
    if _path_exists(sibling_candidate):
        return sibling_candidate
    return None


def _build_duplicate_move_destination(source_path: Path, move_root: Path) -> Path:
    source_path = _resolve_input_path(source_path)
    move_root = _resolve_input_path(move_root)
    anchor_name = "dataset_split"
    lower_parts = [part.lower() for part in source_path.parts]
    if anchor_name in lower_parts:
        anchor_index = lower_parts.index(anchor_name)
        relative = Path(*source_path.parts[anchor_index + 1:])
    else:
        relative = Path(source_path.name)
    return move_root / relative


def _path_exists(path: Union[str, Path]) -> bool:
    normalized_path = _normalize_windows_long_path(path)
    return os.path.exists(normalized_path)


def _delete_file(path: Union[str, Path]) -> None:
    normalized_path = _normalize_windows_long_path(path)
    os.remove(normalized_path)


def _build_yolo_metadata_error(dataset_root: str) -> str:
    return (
        f"Не найдены метаданные классов YOLO в '{dataset_root}'. "
        "Ожидался один из файлов: data.yaml, dataset.yaml, obj.names, classes.txt, labels/classes.txt. "
        "Если это ImageFolder-датасет или корень со сплитами, выберите формат reference = 'image_folder'."
    )


def _invert_class_map(mapping: ClassMap) -> Dict[str, List[int]]:
    inverted: Dict[str, List[int]] = {}
    for idx, name in mapping.items():
        inverted.setdefault(name, []).append(idx)
    for ids in inverted.values():
        ids.sort()
    return inverted


def _is_contiguous_zero_based(ids: Sequence[int]) -> bool:
    sorted_ids = sorted(ids)
    return sorted_ids == list(range(len(sorted_ids)))


def _rewrite_yolo_label_lines(label_path: Path, remap: Dict[int, int]) -> Tuple[List[str], bool, List[str]]:
    warnings: List[str] = []
    try:
        original_lines = _read_text_file(label_path).splitlines()
    except FileNotFoundError:
        return [], False, [f"Пропущен отсутствующий label-файл: {label_path}"]
    except UnicodeDecodeError:
        return [], False, [f"Пропущен не-текстовый файл: {label_path}"]

    updated_lines: List[str] = []
    changed = False
    annotation_lines_seen = 0

    for line_number, line in enumerate(original_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated_lines.append(line)
            continue

        parts = stripped.split()
        try:
            old_id = int(parts[0])
            float_values = [float(value) for value in parts[1:]]
            _ = float_values
        except ValueError:
            warnings.append(f"Пропущена некорректная строка в {label_path}:{line_number}: {stripped}")
            updated_lines.append(line)
            continue

        annotation_lines_seen += 1
        if old_id not in remap:
            warnings.append(f"В {label_path}:{line_number} найден unmapped class id {old_id}")
            updated_lines.append(line)
            continue

        new_id = remap[old_id]
        if new_id != old_id:
            changed = True
        if len(parts) == 1:
            updated_lines.append(str(new_id))
        else:
            updated_lines.append(f"{new_id} {' '.join(parts[1:])}")

    should_write = changed and annotation_lines_seen > 0
    return updated_lines, should_write, warnings


def _rewrite_yolo_metadata_files(
    root: Path,
    metadata: Dict[str, Any],
    target_map: ClassMap,
    create_backup: bool,
) -> FixResult:
    result = FixResult(fmt="yolo_metadata", path=str(root))
    sorted_target_items = sorted(target_map.items())
    contiguous_ids = _is_contiguous_zero_based(list(target_map.keys()))

    yaml_path = metadata.get("yaml_path")
    if yaml_path and yaml is not None:
        original_text = _read_text_file(yaml_path)
        with open(yaml_path, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        names_kind = metadata.get("yaml_names_kind")
        if names_kind == "list" and contiguous_ids:
            config["names"] = [name for _, name in sorted_target_items]
        else:
            config["names"] = {int(class_id): name for class_id, name in sorted_target_items}

        updated_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if updated_text != original_text:
            if create_backup:
                result.backup_files.append(str(_backup_file(yaml_path)))
            _write_text_file(yaml_path, updated_text)
            result.changed_files.append(str(yaml_path))

    for txt_name in ("obj.names", "classes.txt", "labels/classes.txt"):
        txt_path = root / txt_name
        if not txt_path.exists():
            continue
        if not contiguous_ids:
            result.skipped_files.append(str(txt_path))
            result.warnings.append(
                f"Текстовый список классов {txt_path} не обновлён: target ids не являются contiguous 0..N-1."
            )
            continue
        updated_text = "\n".join(name for _, name in sorted_target_items) + "\n"
        original_text = _read_text_file(txt_path)
        if updated_text != original_text:
            if create_backup:
                result.backup_files.append(str(_backup_file(txt_path)))
            _write_text_file(txt_path, updated_text)
            result.changed_files.append(str(txt_path))

    return result


def _backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def _resolve_input_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _read_image(path: Union[str, Path], flags: int) -> Optional[Any]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV (cv2) and numpy are required for image reading.")

    normalized_path = _normalize_windows_long_path(path)
    try:
        if os.name == "nt":
            file_bytes = np.fromfile(normalized_path, dtype=np.uint8)
            if file_bytes.size == 0:
                return None
            return cv2.imdecode(file_bytes, flags)
        return cv2.imread(str(normalized_path), flags)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _normalize_windows_long_path(path: Union[str, Path]) -> str:
    path_str = str(path)
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str[2:]
    if len(path_str) >= 240:
        return "\\\\?\\" + path_str
    return path_str


def _read_text_file(path: Union[str, Path], encoding: str = "utf-8") -> str:
    normalized_path = _normalize_windows_long_path(path)
    with open(normalized_path, "r", encoding=encoding) as file:
        return file.read()


def _write_text_file(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    normalized_path = _normalize_windows_long_path(path)
    with open(normalized_path, "w", encoding=encoding) as file:
        file.write(text)


def _apply_coco_duplicate_annotations_fix(
    annotation_path: str,
    duplicate_paths: Sequence[Path],
    create_backup: bool,
) -> FixResult:
    coco_path = _resolve_input_path(annotation_path)
    result = FixResult(fmt="coco_duplicates", path=str(coco_path))

    with open(_normalize_windows_long_path(coco_path), encoding="utf-8") as file:
        data = json.load(file)

    duplicate_names = {path.name.lower() for path in duplicate_paths}
    matched_images = []
    skipped_names = []

    for image in data.get("images", []):
        file_name = str(image.get("file_name", ""))
        if Path(file_name).name.lower() in duplicate_names:
            matched_images.append(image)

    matched_ids = {int(image["id"]) for image in matched_images}
    matched_names = {Path(str(image.get("file_name", ""))).name.lower() for image in matched_images}
    skipped_names = sorted(duplicate_names - matched_names)
    for file_name in skipped_names:
        result.warnings.append(f"В COCO не найдено image entry для дубликата: {file_name}")

    if not matched_ids:
        return result

    data["images"] = [image for image in data.get("images", []) if int(image["id"]) not in matched_ids]
    data["annotations"] = [
        annotation
        for annotation in data.get("annotations", [])
        if int(annotation.get("image_id", -1)) not in matched_ids
    ]

    if create_backup:
        result.backup_files.append(str(_backup_file(coco_path)))
    _write_text_file(coco_path, json.dumps(data, ensure_ascii=False, indent=2))
    result.changed_files.append(str(coco_path))
    return result


def _create_validator_with_retry(
    reference_format: str,
    reference_path: str,
    interactive: bool,
) -> Tuple[DatasetClassValidator, str, str]:
    while True:
        try:
            validator = DatasetClassValidator(
                reference_format=reference_format,
                reference_path=reference_path,
            )
            return validator, reference_format, reference_path
        except (FileNotFoundError, ValueError) as exc:
            if not interactive:
                raise
            print(f"\n[Ошибка reference] {exc}")
            if reference_format == "yolo":
                print("Подсказка: если это папка с подпапками классов или train/val/test-сплитами, используйте 'image_folder'.")
            retry = _ask_yes_no("Повторно ввести format/path для reference?", default=True)
            if not retry:
                raise
            reference_format = _ask_format("Формат reference", default=reference_format)
            reference_path = _ask_text("Путь к reference", default=reference_path)


def _inspect_train_config(repo_root: Path, reference_map: ClassMap) -> List[str]:
    config_path = repo_root / "config" / "train_conf.yaml"
    if not config_path.exists():
        return []
    if yaml is None:
        return [f"  [DOWNSTREAM] Cannot inspect {config_path}: PyYAML is not installed."]

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    metric_names = (((config.get("METRICS") or {}).get("class_names")) or [])
    if not metric_names:
        return []

    expected_map = {idx: name for idx, name in enumerate(metric_names)}
    if expected_map == reference_map:
        return []
    return [
        f"  [DOWNSTREAM] {config_path} METRICS.class_names={metric_names} does not match reference map {reference_map}"
    ]


def _inspect_inference_mapping(repo_root: Path, reference_map: ClassMap) -> List[str]:
    inference_path = repo_root / "trainer" / "inference.py"
    if not inference_path.exists():
        return []

    source = inference_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"  [DOWNSTREAM] Failed to parse {inference_path} to inspect cls_names mapping."]

    literal_mapping = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "cls_names":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    value = None
                if isinstance(value, dict):
                    literal_mapping = {str(name): int(idx) for name, idx in value.items()}
                    break
        if literal_mapping is not None:
            break

    if literal_mapping is None:
        return []

    reference_name_to_idx = {name: idx for idx, name in reference_map.items()}
    if literal_mapping == reference_name_to_idx:
        return []
    return [
        f"  [DOWNSTREAM] {inference_path} cls_names={literal_mapping} does not match reference map {reference_map}"
    ]


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "1", "да", "д"}


def _ask_duplicate_action() -> str:
    prompt = "Действие для дубликата: [d] удалить, [m] перенести, [s] пропустить"
    while True:
        answer = input(f"{prompt} [s]: ").strip().lower()
        if not answer:
            return "skip"
        if answer in {"d", "delete", "удалить", "у"}:
            return "delete"
        if answer in {"m", "move", "перенести", "п"}:
            return "move"
        if answer in {"s", "skip", "пропустить"}:
            return "skip"
        print("Неизвестное действие. Выберите d / m / s.")


def _ask_duplicate_dataset_action() -> str:
    prompt = "Как обработать все найденные дубликаты в этом датасете: [d] удалить, [m] перенести, [s] пропустить"
    while True:
        answer = input(f"{prompt} [s]: ").strip().lower()
        if not answer:
            return "skip"
        if answer in {"d", "delete", "удалить", "у"}:
            return "delete"
        if answer in {"m", "move", "перенести", "п"}:
            return "move"
        if answer in {"s", "skip", "пропустить"}:
            return "skip"
        print("Неизвестное действие. Выберите d / m / s.")


def _ask_split_leakage_batch_action(split_names: Sequence[str]) -> Tuple[Optional[str], str]:
    while True:
        keep_raw = input(
            "Номер сплита, в котором нужно оставить все конфликтные файлы "
            f"(1-{len(split_names)}) или [s] пропустить: "
        ).strip().lower()
        if keep_raw in {"s", "skip", "пропустить"}:
            return None, "skip"
        if keep_raw.isdigit():
            keep_index = int(keep_raw) - 1
            if 0 <= keep_index < len(split_names):
                keep_split = split_names[keep_index]
                break
        print("Некорректный выбор сплита.")

    while True:
        action_raw = input(
            f"Как обработать все копии из остальных сплитов относительно '{keep_split}'? "
            "[d] удалить, [m] перенести, [s] пропустить [m]: "
        ).strip().lower()
        if not action_raw:
            return keep_split, "move"
        if action_raw in {"d", "delete", "удалить", "у"}:
            return keep_split, "delete"
        if action_raw in {"m", "move", "перенести", "п"}:
            return keep_split, "move"
        if action_raw in {"s", "skip", "пропустить"}:
            return keep_split, "skip"
        print("Неизвестное действие. Выберите d / m / s.")


def _ask_text(prompt: str, default: Optional[str] = None) -> str:
    if default is not None:
        answer = input(f"{prompt} [{default}]: ").strip()
        return answer or default
    answer = input(f"{prompt}: ").strip()
    while not answer:
        answer = input(f"{prompt}: ").strip()
    return answer


def _ask_format(prompt: str, default: str = "coco") -> str:
    supported = "/".join(DatasetClassValidator.SUPPORTED_FORMATS)
    answer = input(f"{prompt} ({supported}) [{default}]: ").strip().lower()
    answer = answer or default
    while answer not in DatasetClassValidator.SUPPORTED_FORMATS:
        answer = input(f"Неизвестный формат. Выберите {supported}: ").strip().lower()
    return answer


def _collect_candidates_from_input() -> List[Dict[str, str]]:
    print("\n[Candidates]")
    print("Candidates — это необязательные дополнительные датасеты для сравнения с reference-датасетом.")
    print("Используйте их, если хотите сравнить один или несколько датасетов/версий с одним эталоном.")
    print("Если у вас сейчас только один датасет, candidates можно пропустить.")

    candidates: List[Dict[str, str]] = []
    add_first = _ask_yes_no("Добавить candidate-датасет для сравнения?", default=False)
    if not add_first:
        return candidates

    while True:
        fmt = _ask_format("Формат candidate")
        path = _ask_text("Путь к candidate")
        label = _ask_text(
            "Label (необязательное имя в отчёте, например 'train split' или 'dataset v2')",
            default=path,
        )
        candidates.append({"format": fmt, "path": path, "label": label})
        if not _ask_yes_no("Добавить ещё один candidate?", default=False):
            break
    return candidates


def _collect_dataset_targets(title: str) -> List[Dict[str, str]]:
    print(f"\nСбор целей для этапа: {title}")
    print("Здесь можно указать один датасет или несколько датасетов.")
    items: List[Dict[str, str]] = []
    while True:
        fmt = _ask_format(f"Формат для {title}")
        path = _ask_text(f"Путь для {title}")
        items.append({"format": fmt, "path": path})
        if not _ask_yes_no(f"Добавить ещё одну цель для {title}?", default=False):
            break
    return items


def _collect_plain_paths(title: str) -> List[str]:
    print(f"\nСбор путей для этапа: {title}")
    items: List[str] = []
    while True:
        items.append(_ask_text(title))
        if not _ask_yes_no(f"Добавить ещё один путь для {title}?", default=False):
            break
    return items


def _collect_duplicate_targets() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    while True:
        path = _ask_text("Папка для поиска дубликатов")
        dataset_format = _ask_format("Формат датасета для дубликатов", default="image_folder")
        annotation_path = None
        if dataset_format == "coco":
            annotation_path = _ask_text("Путь к COCO annotations.json")
        threshold_raw = _ask_text("Порог pHash", default="5")
        item = {"path": path, "threshold": int(threshold_raw), "format": dataset_format}
        if annotation_path is not None:
            item["annotation_path"] = annotation_path
        items.append(item)
        if not _ask_yes_no("Добавить ещё одну папку для поиска дубликатов?", default=False):
            break
    return items


def _print_remap_plan(plan: RemapPlan) -> None:
    print("\n[План remap]")
    print(f"Reference: {plan.reference_map}")
    print(f"Candidate: {plan.candidate_map}")
    if plan.remap:
        print("Предлагаемый remap:")
        for old_idx, new_idx in sorted(plan.remap.items()):
            class_name = plan.candidate_map.get(old_idx, plan.reference_map.get(new_idx, "unknown"))
            print(f"  {old_idx} -> {new_idx} ({class_name})")
    if plan.errors:
        print("Блокирующие проблемы:")
        for item in plan.errors:
            print(item)
    if plan.warnings:
        print("Предупреждения:")
        for item in plan.warnings:
            print(item)
    if plan.safe_to_apply and plan.needs_changes:
        print("Автоисправление можно применить безопасно.")
    elif not plan.needs_changes:
        print("Remap не требуется.")


def print_fix_report(result: FixResult) -> None:
    print(f"\n[Результат исправления] {result.fmt} -> {result.path}")
    if result.changed_files:
        print("Изменённые файлы:")
        for file_path in result.changed_files:
            print(f"  - {file_path}")
    if result.backup_files:
        print("Backup-файлы:")
        for file_path in result.backup_files:
            print(f"  - {file_path}")
    if result.skipped_files:
        print("Пропущенные файлы:")
        for file_path in result.skipped_files:
            print(f"  - {file_path}")
    if result.warnings:
        print("Предупреждения:")
        for warning in result.warnings:
            print(f"  - {warning}")


def main() -> None:
    print("Пайплайн валидации датасета")
    print("Reference dataset — основной датасет, чья карта классов используется как эталон.")
    print("Candidates — необязательные дополнительные датасеты для сравнения с reference.")
    print("Label — просто удобное имя в отчёте, на логику валидации не влияет.\n")

    reference_format = _ask_format("Формат reference")
    reference_path = _ask_text("Путь к reference")
    run_imbalance = _ask_yes_no("Запустить проверку дисбаланса классов?", default=False)
    run_split = _ask_yes_no("Запустить проверку пересечения сплитов?", default=False)
    run_duplicates = _ask_yes_no("Запустить поиск дубликатов?", default=False)

    run_validation_pipeline(
        reference_format=reference_format,
        reference_path=reference_path,
        interactive=True,
        run_imbalance_checks=run_imbalance,
        run_split_checks=run_split,
        run_duplicate_checks=run_duplicates,
        create_backup=True,
    )


if __name__ == "__main__":
    main()
