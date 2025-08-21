# dataprocessor.py
'''
1.读取附件一，的地块名称、地块类型列，形成 地块名称->地块类型 索引。
2.从附件二 2023年的农作物种植情况 工作簿中读取种植地块、作物名称、种植面积/亩 列，根据种植地块去1中搜索得到对应的地块类型。
3.读取附件二 2023年统计的相关数据 工作簿，根据2中地块类型和作物名称检索 亩产量/斤 和 种植成本/(元/亩) 列，后分别乘上2中的亩产量得到该作物在该地块的总需求量/斤和 该作物总成本/元。
4.对3中的结果通过作物名称分组，sum函数聚合，最终得到 作物名称、总需求量/斤、作物总成本/元
'''

import pandas as pd


# --- 文件和工作簿(Sheet)名称定义 ---
# 请确保这两个Excel文件与本脚本文件放置在同一个文件夹下
file_1 = './C/附件1.xlsx'
file_2 = './C/附件2.xlsx'

# 附件一的两个工作簿(Sheet)名称
sheet_land_info = '乡村的现有耕地'
sheet_corp_info = '乡村种植的农作物'
# 附件二的两个工作簿(Sheet)名称
sheet_planting_2023 = '2023年的农作物种植情况'
sheet_stats_2023 = '2023年统计的相关数据'

# 这是最终输出结果的文件名
output_filename = '各类农作物总需求量与总成本.xlsx'


# 自定义函数用于去除字符串两端空格
def strip_whitespace(value):
    if isinstance(value, str):
        return value.strip()
    return value


# 1. 读取附件一，的地块名称、地块类型列，形成 地块名称->地块类型 索引。
df_land_info = pd.read_excel(file_1, sheet_name=sheet_land_info, usecols=['地块名称', '地块类型'])
# 对相关列应用去除空格函数
df_land_info['地块名称'] = df_land_info['地块名称'].apply(strip_whitespace)
df_land_info['地块类型'] = df_land_info['地块类型'].apply(strip_whitespace)
land_type_map = df_land_info.set_index('地块名称')['地块类型'].to_dict()
print("步骤1/4: 地块类型索引创建成功。")

# 读取附件一的另一个工作簿并清洗数据
df_corp_info = pd.read_excel(file_1, sheet_name=sheet_corp_info)
for col in df_corp_info.columns:
    df_corp_info[col] = df_corp_info[col].apply(strip_whitespace)

# 2. 读取附件二的“种植情况”工作簿，并匹配地块类型。
df_planting_2023 = pd.read_excel(file_2, sheet_name=sheet_planting_2023)
# 对相关列应用去除空格函数
df_planting_2023['种植地块'] = df_planting_2023['种植地块'].apply(strip_whitespace)
df_planting_2023['作物名称'] = df_planting_2023['作物名称'].apply(strip_whitespace)
# 使用前向填充 (ffill) 处理第二季作物地块名称为空白的情况
df_planting_2023['种植地块'] = df_planting_2023['种植地块'].fillna(method='ffill')
df_planting_2023['地块类型'] = df_planting_2023['种植地块'].map(land_type_map)
print("步骤2/4: 种植情况数据读取并匹配地块类型成功。")

# 3. 读取附件二的“统计数据”工作簿，合并数据并进行计算。
df_stats_2023 = pd.read_excel(file_2, sheet_name=sheet_stats_2023)
# 对相关列应用去除空格函数
df_stats_2023['作物名称'] = df_stats_2023['作物名称'].apply(strip_whitespace)
df_stats_2023['地块类型'] = df_stats_2023['地块类型'].apply(strip_whitespace)
# 为了精确匹配，使用 '作物名称', '地块类型', '种植季次' 作为共同键进行合并
merged_df = pd.merge(
    df_planting_2023,
    df_stats_2023,
    on=['作物名称', '地块类型'],
    how='left'  # 使用左连接，以确保所有种植记录都被保留
)
merged_df.to_excel('merged_data.xlsx', index=False)
# 检查并处理合并后可能出现的空值（即在统计数据中找不到匹配项的种植记录）
if merged_df['亩产量/斤'].isnull().any():
    print("警告：部分种植记录在统计数据中未能找到匹配的亩产量或成本，这些记录的总产量和总成本将记为0。")
    # 将无法匹配的记录的产量和成本填充为0，以避免计算错误
    merged_df.fillna(0, inplace=True)
# 计算总需求量（即总产量）和总成本
merged_df['总需求量/斤'] = merged_df['种植面积/亩'] * merged_df['亩产量/斤']
merged_df['作物总成本/元'] = merged_df['种植面积/亩'] * merged_df['种植成本/(元/亩)']
print("步骤3/4: 数据合并与计算完成。")

# 4. 按作物名称分组，聚合计算最终结果。
final_summary = merged_df.groupby('作物名称').agg(
    {'总需求量/斤': 'sum', '作物总成本/元': 'sum'}
).reset_index()
print("步骤4/4: 分组聚合完成。")

# 将清洗后的数据写回原文件替换原数据
with pd.ExcelWriter(file_1, engine='openpyxl', mode='w') as writer:
    df_land_info.to_excel(writer, sheet_name=sheet_land_info, index=False)
    df_corp_info.to_excel(writer, sheet_name=sheet_corp_info, index=False)

with pd.ExcelWriter(file_2, engine='openpyxl', mode='w') as writer:
    df_planting_2023.to_excel(writer, sheet_name=sheet_planting_2023, index=False)
    df_stats_2023.to_excel(writer, sheet_name=sheet_stats_2023, index=False)

# 5. 保存结果到新的Excel文件
final_summary.to_excel(output_filename, index=False)

print("\n-------------------------------------------")
print(f"🎉 数据处理成功！结果已保存至文件: {output_filename}")
print("-------------------------------------------")