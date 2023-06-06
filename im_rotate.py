from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm
# ext = input('Input the original file extension: ')
# new = input('Input the new file extension: ')

DEGREE = Image.ROTATE_270

in_dir = '/deep/db/beverage/images/2023-06-01'

# Creates a list of all the files with the given extension in the current folder:
in_path = Path(in_dir)  # os-agnostic
files = glob.glob(str(in_path / '**' /'*.jpg'), recursive=True)
files = sorted(files)

# Converts the images:
for f in tqdm(files):
    im0 = Image.open(f)
    # Rotate Image
    rotated_image = im0.transpose(DEGREE)
    rotated_image.save(f)
