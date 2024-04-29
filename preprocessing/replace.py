import csv  
  
# 读取CSV文件  
with open('/remote-home/2230728/project/EMG/Emg/preprocessing/1.csv', mode='r', newline='') as file:  
    reader = csv.reader(file)  
    original_data = list(reader)  # 将读取器中的数据转化为列表  
  
# 创建一个空列表来存储复制后的数据  
repeated_data = []  
  
# 遍历原始数据中的每一行  
for row in original_data:  
    # 将当前行复制8000次并添加到repeated_data列表中  
    repeated_data.extend([row] * 40)  
  
# 将复制后的数据写入新的CSV文件  
with open('/remote-home/2230728/project/EMG/Emg/preprocessing/repeated_data.csv', mode='w', newline='') as file:  
    writer = csv.writer(file)  
    writer.writerows(repeated_data)  # 写入多行数据