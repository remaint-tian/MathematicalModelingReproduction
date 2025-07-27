import pandas as pd

# 读取数据文件，这里以读取 Excel 文件为例，你可以根据实际情况修改文件类型和路径
# 如果是 CSV 文件，可以使用 pd.read_csv('文件路径.csv')
data = pd.read_excel('./C/附件2.xlsx')

# 使用 groupby 按照农作物类型对需求量进行分组求和
total_demand = data.groupby('作物名称')['种植面积/亩'].sum().reset_index()

# 将结果写入新的 Excel 文件
total_demand.to_excel('./C/各类农作物总需求量.xlsx', index=False)