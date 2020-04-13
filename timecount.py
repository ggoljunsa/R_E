from datetime import datetime

new = 0

for _ in range(500):
    dt = datetime.now() - new
    new = dt
    print(dt.microsecond)
