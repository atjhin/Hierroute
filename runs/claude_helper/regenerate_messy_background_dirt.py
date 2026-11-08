import os
import sys
import glob
import random
from pathlib import Path
from datetime import datetime

from PIL import Image
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

from hierroute.extra_functions import add_synthetic_dirt

DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
MESSY_BACKGROUND_DIR = os.path.join(DATA_DIRECTORY, 'Messy_background')
LOG_PATH = os.path.join(MESSY_BACKGROUND_DIR, 'moved_images', 'log.md')
BASE_SEED = 2027

# Regenerates the 393 files HierNet's apply_synthetic_dirt_to_moved_images.py baked
# with its old, pre-fix dirt logic. The exact clean pixels for those files no longer
# exist (move_to_messy_background.py used shutil.move, not copy), so each target
# filename is repointed at a freshly-sampled clean image of the same class from
# DATA_DIRECTORY -- read-only there, nothing under DATA_DIRECTORY is moved/deleted.


def parse_moved_pairs(log_path):
    with open(log_path) as f:
        content = f.read()
    pairs = set()
    for section in content.split('### Moved files')[1:]:
        table_text = section.split('\n## ')[0]
        for line in table_text.splitlines():
            line = line.strip()
            if not line.startswith('|'):
                continue
            cols = [c.strip() for c in line.strip('|').split('|')]
            if len(cols) < 2:
                continue
            class_name, filename = cols[0], cols[1]
            if class_name == 'Class' or set(class_name) == {'-'}:
                continue
            pairs.add((class_name, filename))
    return pairs


moved_pairs = parse_moved_pairs(LOG_PATH)
by_class = {}
for cls_name, filename in moved_pairs:
    by_class.setdefault(cls_name, []).append(filename)

jobs = []  # (class_name, target_filename, source_path)
random.seed(666)
for cls_name in sorted(by_class):
    target_filenames = sorted(by_class[cls_name])
    clean_pool = sorted(glob.glob(os.path.join(DATA_DIRECTORY, cls_name, '*.tif')))
    if len(clean_pool) < len(target_filenames):
        raise RuntimeError(
            f'Not enough clean {cls_name} images ({len(clean_pool)}) to regenerate '
            f'{len(target_filenames)} Messy_background slots.'
        )
    source_paths = random.sample(clean_pool, len(target_filenames))
    for target_filename, source_path in zip(target_filenames, source_paths):
        jobs.append((cls_name, target_filename, source_path))

print(f'Regenerating {len(jobs)} files across {len(by_class)} classes in {MESSY_BACKGROUND_DIR}')

log_rows = []
for i, (cls_name, target_filename, source_path) in enumerate(jobs):
    target_path = os.path.join(MESSY_BACKGROUND_DIR, cls_name, target_filename)

    if os.path.exists(target_path):
        os.remove(target_path)

    clean = Image.open(source_path).convert('L')
    seed = BASE_SEED + i
    dirty = add_synthetic_dirt(clean, seed=seed)
    dirty.save(target_path)

    log_rows.append((cls_name, target_filename, os.path.basename(source_path), seed))
    print(f'  [{i + 1}/{len(jobs)}] {cls_name}/{target_filename}  <-  {os.path.basename(source_path)}  (seed={seed})')

with open(LOG_PATH, 'a') as f:
    f.write(f"\n## Synthetic dirt regenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(
        'Regenerated with the fixed `hierroute.extra_functions.add_synthetic_dirt` '
        '(blurred dirt, organism pasted back from the original untouched pixels via '
        'its silhouette mask). The clean pixels for the originally-moved files no '
        'longer exist, so each target filename below was re-dirtied from a freshly '
        f'sampled clean image of the same class in `{DATA_DIRECTORY}` instead '
        '(read-only there -- nothing was moved or deleted from it).\n\n'
    )
    f.write(f'Base seed: {BASE_SEED}\n\n')
    f.write('### Regenerated files\n\n')
    f.write('| Class | Target Filename | Source Clean Filename | Seed |\n')
    f.write('|---|---|---|---|\n')
    for cls_name, target_filename, source_filename, seed in log_rows:
        f.write(f'| {cls_name} | {target_filename} | {source_filename} | {seed} |\n')

print(f'\nDone. Appended regeneration record to {LOG_PATH}')
