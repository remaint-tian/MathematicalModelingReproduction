import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 读取文件
excel_file_1 = pd.ExcelFile('/mnt/result1_1.xlsx')
excel_file_2 = pd.ExcelFile('/mnt/result1_2.xlsx')

# 遍历文件
dfs = []
for excel_file in [excel_file_1, excel_file_2]:
    yearly_dfs = []
    # 遍历不同年份工作表
    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        df['年份'] = sheet_name
        yearly_dfs.append(df)
    # 合并同一个文件内不同年份的数据
    combined_df = pd.concat(yearly_dfs, ignore_index=True)
    dfs.append(combined_df)

# 合并两个文件的数据
final_df = pd.concat(dfs, ignore_index=True)

# 提取蔬菜列
vegetable_columns = [col for col in final_df.columns if col not in ['地块名称', '年份']]

# 绘制不同年份的热图
for year in final_df['年份'].unique():
    year_df = final_df[final_df['年份'] == year]
    pivot_df = year_df.set_index('地块名称')[vegetable_columns]

    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap='YlGnBu', linewidths=0.5)
    plt.title(f'{year}最优种植方案')
    plt.xlabel('蔬菜名称')
    plt.ylabel('地块名称')
    plt.xticks(rotation=45)
    plt.show()