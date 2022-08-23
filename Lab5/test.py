import time
from tqdm import trange

last = -1
for i in trange(10, 100, initial=10, total=100, desc="hello"):
    time.sleep(0.1)
    last = i
print(last)

