"""
Диагностика датасета для выявления проблем обучения
Запустить: python diagnose_dataset.py --csv datasets/data/annotations.csv --json datasets/skeletons_balanced
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def _select_json_files(json_dir: Path, max_files: int = 500, prefer_pp: bool = True) -> List[Path]:
    files = sorted(json_dir.glob("*.json"))
    by_id: Dict[str, Dict[str, Path]] = {}
    for f in files:
        stem = f.stem
        if stem.endswith("_pp"):
            key = stem[:-3]
            by_id.setdefault(key, {})["pp"] = f
        else:
            by_id.setdefault(stem, {})["raw"] = f

    chosen: List[Path] = []
    for entry in by_id.values():
        if prefer_pp and "pp" in entry:
            chosen.append(entry["pp"])
        elif "raw" in entry:
            chosen.append(entry["raw"])
        elif "pp" in entry:
            chosen.append(entry["pp"])

    chosen = sorted(chosen)
    if max_files and max_files > 0:
        chosen = chosen[:max_files]
    return chosen

def analyze_class_distribution(csv_path):
    """Анализ распределения классов"""
    import csv
    
    train_labels = []
    val_labels = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        delimiters = [',', '\t', ';']
        delimiter_scores = {d: sample.count(d) for d in delimiters}
        delimiter = max(delimiter_scores, key=delimiter_scores.get)
        if delimiter_scores[delimiter] == 0:
            delimiter = ','
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            label = row.get('text', '').strip()
            split = row.get('split', '').strip().lower()
            
            if not label or label.lower() == 'no_event':
                continue
                
            if split == 'train' or row.get('train', '').lower() in ['true', '1', 'yes']:
                train_labels.append(label)
            elif split == 'val' or row.get('train', '').lower() in ['false', '0', 'no']:
                val_labels.append(label)
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
    print(f"{'='*60}")
    
    print(f"\n📊 Общая статистика:")
    print(f"  Train: {len(train_labels)} сэмплов, {len(train_counter)} уникальных классов")
    print(f"  Val:   {len(val_labels)} сэмплов, {len(val_counter)} уникальных классов")
    
    # Статистика по количеству примеров на класс
    train_counts = list(train_counter.values())
    
    print(f"\n📈 Примеры на класс (train):")
    if train_counts:
        print(f"  Среднее:  {np.mean(train_counts):.1f}")
        print(f"  Медиана:  {np.median(train_counts):.1f}")
        print(f"  Min:      {np.min(train_counts)}")
        print(f"  Max:      {np.max(train_counts)}")
        print(f"  P25:      {np.percentile(train_counts, 25):.1f}")
        print(f"  P75:      {np.percentile(train_counts, 75):.1f}")
        
        # Классы с малым числом примеров
        few_shot_classes = sum(1 for c in train_counts if c < 5)
        medium_shot = sum(1 for c in train_counts if 5 <= c < 20)
        many_shot = sum(1 for c in train_counts if c >= 20)
        
        print(f"\n⚠️  Распределение:")
        print(f"  Few-shot (<5):    {few_shot_classes} классов ({few_shot_classes/len(train_counts)*100:.1f}%)")
        print(f"  Medium (5-19):    {medium_shot} классов ({medium_shot/len(train_counts)*100:.1f}%)")
        print(f"  Many-shot (≥20):  {many_shot} классов ({many_shot/len(train_counts)*100:.1f}%)")
        
        # Топ и худшие классы
        print(f"\n🔝 Топ-10 классов:")
        for label, count in train_counter.most_common(10):
            print(f"  {label[:40]:40s} → {count:4d} примеров")
        
        print(f"\n⚠️  10 самых редких классов:")
        for label, count in train_counter.most_common()[-10:]:
            print(f"  {label[:40]:40s} → {count:4d} примеров")
        
        # Несбалансированность
        max_count = max(train_counts)
        min_count = min(train_counts)
        imbalance_ratio = max_count / max(min_count, 1)
        
        print(f"\n⚖️  Несбалансированность:")
        print(f"  Ratio (max/min): {imbalance_ratio:.1f}x")
        
        if imbalance_ratio > 100:
            print(f"  ❌ КРИТИЧЕСКАЯ несбалансированность! (>{imbalance_ratio:.0f}x)")
        elif imbalance_ratio > 10:
            print(f"  ⚠️  Высокая несбалансированность ({imbalance_ratio:.0f}x)")
    else:
        print("  Нет train-примеров после фильтрации (проверьте колонки split/train).")
    
    return train_counter, val_counter

def check_sequence_lengths(json_path, prefer_pp: bool = True):
    """Проверка длин последовательностей"""
    print(f"\n{'='*60}")
    print("АНАЛИЗ ДЛИН ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
    print(f"{'='*60}")
    
    json_path = Path(json_path)
    lengths = []
    coverage_samples = []
    
    # Если это директория с per-video JSON
    if json_path.is_dir():
        json_files = _select_json_files(json_path, max_files=500, prefer_pp=prefer_pp)
        print(f"\n📁 Анализируем {len(json_files)} файлов...")
        
        for f in json_files:
            try:
                with open(f, 'rb') as fh:
                    data = json.load(fh)
                    frames = data.get('frames', []) if isinstance(data, dict) else data
                    lengths.append(len(frames))
                    
                    # Подсчет coverage
                    valid = sum(1 for fr in frames 
                               if (fr.get('hand 1') and fr.get('hand 1_score', 0) >= 0.45) or
                                  (fr.get('hand 2') and fr.get('hand 2_score', 0) >= 0.45))
                    coverage_samples.append(valid / len(frames) if frames else 0)
            except:
                pass
    
    if lengths:
        print(f"\n📏 Статистика длин последовательностей:")
        print(f"  Среднее:  {np.mean(lengths):.1f} кадров")
        print(f"  Медиана:  {np.median(lengths):.1f} кадров")
        print(f"  Min:      {np.min(lengths)}")
        print(f"  Max:      {np.max(lengths)}")
        
        print(f"\n🎯 Coverage (доля валидных кадров):")
        print(f"  Среднее:  {np.mean(coverage_samples):.2%}")
        print(f"  Медиана:  {np.median(coverage_samples):.2%}")
        print(f"  P25:      {np.percentile(coverage_samples, 25):.2%}")
        
        if np.median(coverage_samples) < 0.5:
            print(f"  ⚠️  Низкий coverage! Рекомендуется понизить пороги hand_score_thr")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument(
        "--prefer_pp",
        dest="prefer_pp",
        action="store_true",
        default=True,
        help="Prefer *_pp.json when using per-video directory input.",
    )
    parser.add_argument(
        "--no_prefer_pp",
        dest="prefer_pp",
        action="store_false",
        help="Use raw *.json even if *_pp.json exists.",
    )
    args = parser.parse_args()
    
    # Анализ классов
    train_counter, val_counter = analyze_class_distribution(args.csv)
    
    # Анализ последовательностей
    check_sequence_lengths(args.json, prefer_pp=args.prefer_pp)
    
    # РЕКОМЕНДАЦИИ
    print(f"\n{'='*60}")
    print("🔧 РЕКОМЕНДАЦИИ")
    print(f"{'='*60}")
    
    n_classes = len(train_counter)
    n_samples = sum(train_counter.values())
    avg_per_class = n_samples / n_classes if n_classes else 0
    
    if avg_per_class < 10:
        print(f"\n❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: {avg_per_class:.1f} примеров/класс")
        print(f"\n   Для {n_classes} классов нужно минимум ~{n_classes * 50} сэмплов")
        print(f"   (у вас только {n_samples})")
        print(f"\n   Решения:")
        print(f"   1. УМЕНЬШИТЬ число классов (оставить топ-100/200)")
        print(f"   2. СОБРАТЬ больше данных")
        print(f"   3. Использовать pre-training на другом датасете")
    
    elif avg_per_class < 20:
        print(f"\n⚠️  ПРОБЛЕМА: {avg_per_class:.1f} примеров/класс - мало!")
        print(f"\n   Попробуйте:")
        print(f"   1. Focal Loss для борьбы с дисбалансом")
        print(f"   2. Сильные аугментации")
        print(f"   3. Cosine classifier (--use_cosine_head)")
    
    print(f"\n💡 Немедленные действия:")
    print(f"   1. Запустить smoke test (см. ниже)")
    print(f"   2. Увеличить LR до 1e-3 или 2e-3")
    print(f"   3. Добавить --use_cosine_head --cosine_margin 0.3")

if __name__ == '__main__':
    main()
