import os
import sys
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
BASE_SEED = 3000

OLD_SECTION_HEADER = '## Synthetic dirt regenerated: 2026-08-17 21:14:42'

# regenerate_messy_background_dirt.py (2026-08-17) assumed the 393 originally-moved
# files' clean pixels were gone (move_to_messy_background.py used shutil.move), so it
# re-dirtied each target filename from a different, randomly-sampled clean image of the
# same class instead. That assumption was wrong: the '## Restoration: 2026-08-13' entry
# in the log shows the originals were copied back to DATA_DIRECTORY/<class>/ that same
# day, before the regenerate script ever ran. This script supersedes that regeneration:
# it re-dirties each of the 393 Moved-files entries from its own true original clean
# pixels (still present at DATA_DIRECTORY/<class>/<filename>) using the current
# hierroute.extra_functions.add_synthetic_dirt, and removes the now-obsolete
# '### Regenerated files' section from the log entirely (rather than appending a
# correction alongside it).


def parse_moved_pairs(log_path):
    with open(log_path) as f:
        content = f.read()
    section = content.split('### Moved files')[1].split('\n## ')[0]
    pairs = set()
    for line in section.splitlines():
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


moved_pairs = sorted(parse_moved_pairs(LOG_PATH))
print(f'Fixing {len(moved_pairs)} files in {MESSY_BACKGROUND_DIR}')

log_rows = []
for i, (cls_name, filename) in enumerate(moved_pairs):
    source_path = os.path.join(DATA_DIRECTORY, cls_name, filename)
    target_path = os.path.join(MESSY_BACKGROUND_DIR, cls_name, filename)

    if not os.path.exists(source_path):
        raise RuntimeError(f'True original clean image not found: {source_path}')
    if not os.path.exists(target_path):
        raise RuntimeError(f'Messy_background target not found: {target_path}')

    clean = Image.open(source_path).convert('L')
    seed = BASE_SEED + i
    dirty = add_synthetic_dirt(clean, seed=seed)
    dirty.save(target_path)

    log_rows.append((cls_name, filename, seed))
    print(f'  [{i + 1}/{len(moved_pairs)}] {cls_name}/{filename}  (seed={seed})')

with open(LOG_PATH) as f:
    content = f.read()

if OLD_SECTION_HEADER not in content:
    raise RuntimeError(f"Expected section '{OLD_SECTION_HEADER}' not found in {LOG_PATH} -- refusing to edit blindly.")

content = content.split(OLD_SECTION_HEADER)[0].rstrip() + '\n'

content += f"\n## Synthetic dirt corrected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
content += (
    'Supersedes and removes the `## Synthetic dirt regenerated: 2026-08-17 21:14:42` '
    'section that previously followed here. That regeneration assumed the 393 '
    "originally-moved files' clean pixels no longer existed and substituted a "
    "different, randomly-sampled clean image of the same class for each. That "
    "assumption was wrong -- see `## Restoration: 2026-08-13` above: the originals "
    'were copied back to their source paths under `Hier-Zooplankton/<class>/` that '
    'same day. This pass re-dirties each of the 393 Moved-files entries from its own '
    'true original clean pixels instead, using the current '
    '`hierroute.extra_functions.add_synthetic_dirt` (blurred dirt, organism pasted '
    'back via its silhouette mask).\n\n'
)
content += f'Base seed: {BASE_SEED}\n\n'
content += '### Corrected files\n\n'
content += '| Class | Filename | Seed |\n'
content += '|---|---|---|\n'
for cls_name, filename, seed in log_rows:
    content += f'| {cls_name} | {filename} | {seed} |\n'

with open(LOG_PATH, 'w') as f:
    f.write(content)

print(f'\nDone. Replaced the 2026-08-17 regeneration record in {LOG_PATH}')
