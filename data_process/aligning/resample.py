import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

csv_file = "../Data/pose/POSE1700831933598.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
x_df = csv_df.iloc[:,[0]]
x_df[:].plot(figsize = (15,2),color='orange')
y_df = csv_df.iloc[:,[1]]
y_df[:].plot(figsize = (15,2),color='blue')
z_df = csv_df.iloc[:,[2]]
z_df[:].plot(figsize = (15,2),color='green')
plt.title('POSE')