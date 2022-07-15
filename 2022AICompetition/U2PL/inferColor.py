import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# 76:boat 83:truck 80:folk 122 bus
def submit(result):
  k = -1
  a, b = np.unique(result, return_counts=True)
  if len(a) == 1:
    sh = result.shape
    w, h = sh[0], sh[1]
    return 'ship', f'{int(w * h / 2)} 5'

  a, b = a[1:], b[1:]

  m = a[b.argmax()]
  label = ''
  if m == 76:
    label = 'ship'
  elif m == 83:
    label = 'container_truck'
  elif m == 80:
    label = 'forklift'
  elif m == 122:
    label = 'reach_stacker'
  result = result.reshape(-1, )
  start = True
  ch = False
  idx = []
  cnt = 0
  for i in range(len(result)):
    if result[i] == m:
      cnt += 1
      if start:
        idx.append(str(i))
        start = False
        ch = True
    elif result[i] != m and ch:
      start = True
      ch = False
      idx.append(str(cnt))
      cnt = 0

  if result[-1] == m and start == False:
    idx.append(str(len(result) - 1 - int(idx[-1])))

  return label, ' '.join(idx)

df = pd.read_csv('/content/drive/MyDrive/SIA/sample.csv')
labels = ('background', 'container_truck', 'forklift', 'reach_stacker', 'ship')
classes = []
index = []
names = []
j = 0

for idx, i in tqdm(enumerate(df['file_name'])):
  j += 1
  img = cv2.imread(f'/content/drive/MyDrive/SIA/test/images/{i}')

  result = np.array(inference_segmentor(model_ckpt, img))
  label, idx = submit(result)
  classes.append(label)
  index.append(idx)
  names.append(i)
  if j % 1000 == 0:
    sub = pd.DataFrame({'file_name':names, 'class':classes, 'prediction':index})
    sub.to_csv(f'/content/drive/MyDrive/SIA/submission_{j}.csv', index=False)
    classes, index, names = [], [], []
