import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# NumPy 配列の例
numpy_array = np.array([["a00", "a01", "a02"],
                          ["a10", "a11", "a12"],
                          ["a20", "a21", "a22"]])

# pandas DataFrameの例
df = pd.DataFrame(numpy_array, columns=["Col1", "Col2", "Col3"], index=["Row0", "Row1", "Row2"])

# NumPy配列の図示
fig1, ax1 = plt.subplots()
ax1.axis('tight')
ax1.axis('off')
table1 = ax1.table(cellText=numpy_array,
                   cellLoc='center',
                   loc='center',
                   colLabels=["0", "1", "2"],
                   rowLabels=["0", "1", "2"])
ax1.set_title("NumPy ndarray (行・列は番号のみ)")
plt.show()

# pandas DataFrameの図示
fig2, ax2 = plt.subplots()
ax2.axis('tight')
ax2.axis('off')
table2 = ax2.table(cellText=df.values,
                   cellLoc='center',
                   loc='center',
                   colLabels=df.columns,
                   rowLabels=df.index)
ax2.set_title("pandas DataFrame (ラベル付き)")
plt.show()