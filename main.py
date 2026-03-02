from pathlib import Path

from sandbox.src.metrics_check import MetricCheck


d = MetricCheck()
s = d.run(path=Path(r"C:\Users\meksi\Documents\GitHub\KLIN\models"))
print(s)
