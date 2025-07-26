import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import Patch


# ---- 外部可选：pyecharts + 调色板 ----
from pyecharts.globals import CurrentConfig, NotebookType
from pyecharts.charts import Pie
from palettable.cartocolors.qualitative import Vivid_10   # 示例调色板

warnings.filterwarnings('ignore')

# Matplotlib 全局中文及负号
plt.rcParams['font.family'] = 'SimHei'        # 没装中文字体可改 ''
plt.rcParams['axes.unicode_minus'] = False



# ---- 读取 Excel （用 r'' 避免反斜杠转义）----
df1 = pd.read_excel(r'D:\2024_C\附件1.xlsx', sheet_name=0)



"""
1.对露天和大棚的耕地结构画分布图
"""


'''
#露天耕地分布
mask_land = df1['地块名称'].str.match(r'^[A-D]\d+$')
df_land = df1[mask_land].copy()
land_type_map = {'A': '平旱地', 'B': '梯田', 'C': '山坡地', 'D': '水浇地'}
df_land['地块类型'] = df_land['地块名称'].str[0].map(land_type_map)
# 外圈：每个地块面积
outer_labels = df_land['地块名称']
outer_sizes = df_land['地块面积/亩']

# 内圈：地块类型求面积和，并为每个地块映射到类型
inner_labels = df_land['地块类型'].unique()
# 每种类型的总面积
inner_sizes = [df_land[df_land['地块类型'] == t]['地块面积/亩'].sum() for t in inner_labels]
# 对应每个地块的类型颜色（用于外圈着色）
type_color_map = dict(zip(inner_labels, plt.cm.Set3.colors[:len(inner_labels)]))
outer_colors = [type_color_map[t] for t in df_land['地块类型']]
inner_colors = [type_color_map[t] for t in inner_labels]

# 画图
fig, ax = plt.subplots(figsize=(8, 8))

# 内圈：类型
ax.pie(inner_sizes, labels=inner_labels, radius=0.7, colors=inner_colors,
       labeldistance=0.4, wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90, autopct='%1.0f%%')

# 外圈：地块
ax.pie(outer_sizes, labels=outer_labels, radius=1.0, colors=outer_colors,
       labeldistance=1.1, wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90)

plt.title('露天耕地结构分布')
plt.tight_layout()
plt.show()


#大棚结构分布


mask_gh = df1['地块名称'].str.match(r'^[E-F]\d+$')
df_gh = df1[mask_gh].copy()
gh_type_map = {'E': '普通大棚', 'F': '智慧大棚'}
df_gh['大棚类型'] = df_gh['地块名称'].str[0].map(gh_type_map)

# 外圈：大棚编号
outer_labels_gh = df_gh['地块名称']
outer_sizes_gh = df_gh['地块面积/亩']

# 内圈：大棚类型
inner_labels_gh = df_gh['大棚类型'].unique()
inner_sizes_gh = [df_gh[df_gh['大棚类型'] == t]['地块面积/亩'].sum() for t in inner_labels_gh]
type_color_map_gh = dict(zip(inner_labels_gh, plt.cm.Pastel1.colors[:len(inner_labels_gh)]))
outer_colors_gh = [type_color_map_gh[t] for t in df_gh['大棚类型']]
inner_colors_gh = [type_color_map_gh[t] for t in inner_labels_gh]

fig, ax = plt.subplots(figsize=(8, 8))

# 内圈：类型
ax.pie(inner_sizes_gh, labels=inner_labels_gh, radius=0.7, colors=inner_colors_gh,
       labeldistance=0.4, wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90, autopct='%1.0f%%')

# 外圈：大棚
ax.pie(outer_sizes_gh, labels=outer_labels_gh, radius=1.0, colors=outer_colors_gh,
       labeldistance=1.1, wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90)

plt.title('大棚结构分布')
plt.tight_layout()
plt.show()

'''








'''
2.各农作物的总产量
'''

'''
# 1. 读取数据
df_area = pd.read_excel(r'D:\2024_C\附件2.xlsx', sheet_name=0)
df_yield = pd.read_excel(r'D:\2024_C\附件2.xlsx', sheet_name=1)

# 2. 合并、归类
df = pd.merge(df_area, df_yield[['作物名称', '亩产量/斤']], on='作物名称', how='left')

def map_type(x):
    if '粮食' in x:
        return '粮食'
    elif '蔬菜' in x:
        return '蔬菜'
    elif '食用菌' in x:
        return '食用菌'
    else:
        return '其他'
df['作物大类'] = df['作物类型'].apply(map_type)
df['总产量'] = df['种植面积/亩'] * df['亩产量/斤']

# 3. 只要三大类且各类内部升序排序
category_order = ['粮食', '蔬菜', '食用菌']
df_sorted = pd.concat([
    df[df['作物大类'] == cat].sort_values('总产量', ascending=True)
    for cat in category_order
])

# 4. 美化颜色
color_dict = {'粮食': '#FF6F91', '蔬菜': '#008F7A', '食用菌': '#FFC75F'}
bar_colors = df_sorted['作物大类'].map(color_dict).tolist()

# 5. 绘图
fig, ax = plt.subplots(figsize=(12, max(8, 0.35*len(df_sorted))))
bars = ax.barh(df_sorted['作物名称'], df_sorted['总产量'],
               color=bar_colors, edgecolor='black', linewidth=1.5, height=0.8)

ax.set_xlabel('总产量（斤）', fontsize=13)
ax.set_title('2023年各类农作物产量(分大类、内部升序排列)', fontsize=15, weight='bold')
plt.gca().invert_yaxis()  # 从小到大


# 6. 分大类注释
yticks = range(len(df_sorted))
ylabels = df_sorted['作物名称'].tolist()
group_counts = [sum(df_sorted['作物大类'] == cat) for cat in category_order]
group_starts = [0, group_counts[0], group_counts[0] + group_counts[1]]
for idx, cat in enumerate(category_order):
    start = group_starts[idx]
    end = group_starts[idx] + group_counts[idx] - 1
    # 居中注释
    ax.text(-0.03*df_sorted['总产量'].max(), (start+end)/2, cat,
            va='center', ha='right', fontsize=15, color=color_dict[cat], weight='bold', rotation=90)
    # 虚线分隔
    if idx > 0:
        ax.axhline(start-0.5, color='gray', linestyle='--', linewidth=1)

# 7. 图例
legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_dict.items()]
ax.legend(handles=legend_elements, title='作物大类', loc='lower right')


plt.tight_layout()
plt.show()

'''


'''
3.不同耕地条件下主要粮食作物亩产量对比——>平旱地,梯田,山坡地
'''
#代码如下




'''
4.不同耕地条件下主要蔬菜作物亩产量对比——>水浇地,普通大棚,智慧大棚
'''




'''
5.2023年各种农作物平均亩利润排名
'''

'''
import re

# 1. 读取数据
df_area = pd.read_excel(r'D:\2024_C\附件2.xlsx', sheet_name=0)  # 农作物种植情况
df_stats = pd.read_excel(r'D:\2024_C\附件2.xlsx', sheet_name=1)  # 统计数据

# 2. 销售单价区间取平均
def get_price_avg(s):
    if isinstance(s, str):
        nums = [float(i) for i in re.findall(r'\d+\.?\d*', s)]
        if len(nums) == 2:
            return sum(nums) / 2
        elif len(nums) == 1:
            return nums[0]
    return float(s) if pd.notnull(s) else None

df_stats['销售单价均值'] = df_stats['销售单价/(元/斤)'].apply(get_price_avg)

# 3. 合并数据
df = pd.merge(df_area, df_stats[['作物名称', '亩产量/斤', '种植成本/(元/亩)', '销售单价均值']], on='作物名称', how='left')

# 4. 类型归并
def map_type(x):
    if '粮食' in x:
        return '粮食'
    elif '蔬菜' in x:
        return '蔬菜'
    elif '食用菌' in x:
        return '食用菌'
    else:
        return '其他'
df['作物大类'] = df['作物类型'].apply(map_type)

# 5. 计算平均亩利润
df['总产量'] = df['种植面积/亩'] * df['亩产量/斤']
df['总利润'] = df['总产量'] * df['销售单价均值'] - df['种植面积/亩'] * df['种植成本/(元/亩)']
df['平均亩利润'] = df['总利润'] / df['种植面积/亩']

# 6. 分组排序
category_order = ['粮食', '蔬菜', '食用菌']
df_sorted = pd.concat([
    df[df['作物大类'] == cat].sort_values('平均亩利润', ascending=True)
    for cat in category_order
])

# 7. 颜色映射
color_dict = {'粮食': '#D62728', '蔬菜': '#2CA02C', '食用菌': '#FFD700'}
bar_colors = df_sorted['作物大类'].map(color_dict).tolist()

# 8. 绘图
fig, ax = plt.subplots(figsize=(12, max(8, 0.35*len(df_sorted))))
bars = ax.barh(df_sorted['作物名称'], df_sorted['平均亩利润'],
               color=bar_colors, edgecolor='black', linewidth=1.2, height=0.8)

ax.set_xlabel('平均亩利润（元/亩）', fontsize=13)
ax.set_title('2023年各类农作物平均亩利润排名(分大类、内部升序)', fontsize=15, weight='bold')
plt.gca().invert_yaxis()

# 9. 分组注释和虚线
yticks = range(len(df_sorted))
group_counts = [sum(df_sorted['作物大类'] == cat) for cat in category_order]
group_starts = [0, group_counts[0], group_counts[0] + group_counts[1]]
for idx, cat in enumerate(category_order):
    start = group_starts[idx]
    end = group_starts[idx] + group_counts[idx] - 1
    ax.text(ax.get_xlim()[0] - 0.01*abs(ax.get_xlim()[1]), (start+end)/2, cat,
            va='center', ha='right', fontsize=15, color=color_dict[cat], weight='bold', rotation=90)
    if idx > 0:
        ax.axhline(start-0.5, color='gray', linestyle='--', linewidth=1)

# 10. 图例
legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_dict.items()]
ax.legend(handles=legend_elements, title='作物大类', loc='lower right')

plt.tight_layout()
plt.show()

'''


'''



import pandas as pd

# 整理数据为列表（每个元素对应一行）
data = [
    {"作物编号": 1, "作物名称": "黄豆", "种植季次": "单季", "总产量/斤": 167580},
    {"作物编号": 2, "作物名称": "黑豆", "种植季次": "单季", "总产量/斤": 65550},
    {"作物编号": 3, "作物名称": "红豆", "种植季次": "单季", "总产量/斤": 68400},
    {"作物编号": 4, "作物名称": "绿豆", "种植季次": "单季", "总产量/斤": 95520},
    {"作物编号": 5, "作物名称": "爬豆", "种植季次": "单季", "总产量/斤": 29625},
    {"作物编号": 6, "作物名称": "小麦", "种植季次": "单季", "总产量/斤": 506160},
    {"作物编号": 7, "作物名称": "玉米", "种植季次": "单季", "总产量/斤": 384750},
    {"作物编号": 8, "作物名称": "谷子", "种植季次": "单季", "总产量/斤": 210900},
    {"作物编号": 9, "作物名称": "高粱", "种植季次": "单季", "总产量/斤": 90000},
    {"作物编号": 10, "作物名称": "黍子", "种植季次": "单季", "总产量/斤": 37500},
    {"作物编号": 11, "作物名称": "荞麦", "种植季次": "单季", "总产量/斤": 4725},
    {"作物编号": 12, "作物名称": "南瓜", "种植季次": "单季", "总产量/斤": 111150},
    {"作物编号": 13, "作物名称": "红薯", "种植季次": "单季", "总产量/斤": 113400},
    {"作物编号": 14, "作物名称": "莜麦", "种植季次": "单季", "总产量/斤": 42000},
    {"作物编号": 15, "作物名称": "大麦", "种植季次": "单季", "总产量/斤": 30000},
    {"作物编号": 16, "作物名称": "水稻", "种植季次": "单季", "总产量/斤": 21000},
    {"作物编号": 17, "作物名称": "豇豆", "种植季次": "第一季", "总产量/斤": 115640},
    {"作物编号": 18, "作物名称": "刀豆", "种植季次": "第一季", "总产量/斤": 87120},
    {"作物编号": 19, "作物名称": "芸豆", "种植季次": "第一季", "总产量/斤": 17640},
    {"作物编号": 20, "作物名称": "土豆", "种植季次": "第一季", "总产量/斤": 99000},
    {"作物编号": 21, "作物名称": "西红柿", "种植季次": "第一季", "总产量/斤": 118260},
    {"作物编号": 21, "作物名称": "西红柿", "种植季次": "第二季", "总产量/斤": 2430},
    {"作物编号": 22, "作物名称": "茄子", "种植季次": "第一季", "总产量/斤": 142560},
    {"作物编号": 22, "作物名称": "茄子", "种植季次": "第二季", "总产量/斤": 6480},
    {"作物编号": 23, "作物名称": "菠菜", "种植季次": "第二季", "总产量/斤": 2700},
    {"作物编号": 24, "作物名称": "青椒", "种植季次": "第一季", "总产量/斤": 4860},
    {"作物编号": 24, "作物名称": "青椒", "种植季次": "第二季", "总产量/斤": 2430},
    {"作物编号": 25, "作物名称": "菜花", "种植季次": "第一季", "总产量/斤": 9810},
    {"作物编号": 26, "作物名称": "包菜", "种植季次": "第一季", "总产量/斤": 11070},
    {"作物编号": 27, "作物名称": "油麦菜", "种植季次": "第一季", "总产量/斤": 12240},
    {"作物编号": 28, "作物名称": "小青菜", "种植季次": "第一季", "总产量/斤": 114480},
    {"作物编号": 28, "作物名称": "小青菜", "种植季次": "第二季", "总产量/斤": 3240},
    {"作物编号": 29, "作物名称": "黄瓜", "种植季次": "第一季", "总产量/斤": 24300},
    {"作物编号": 29, "作物名称": "黄瓜", "种植季次": "第二季", "总产量/斤": 12150},
    {"作物编号": 30, "作物名称": "生菜", "种植季次": "第一季", "总产量/斤": 4080},
    {"作物编号": 30, "作物名称": "生菜", "种植季次": "第二季", "总产量/斤": 4080},
    {"作物编号": 31, "作物名称": "辣椒", "种植季次": "第一季", "总产量/斤": 3240},
    {"作物编号": 32, "作物名称": "空心菜", "种植季次": "第一季", "总产量/斤": 9900},
    {"作物编号": 33, "作物名称": "黄心菜", "种植季次": "第一季", "总产量/斤": 4920},
    {"作物编号": 34, "作物名称": "芹菜", "种植季次": "第二季", "总产量/斤": 5430},
    {"作物编号": 35, "作物名称": "大白菜", "种植季次": "第二季", "总产量/斤": 150000},
    {"作物编号": 36, "作物名称": "白萝卜", "种植季次": "第二季", "总产量/斤": 100000},
    {"作物编号": 37, "作物名称": "红萝卜", "种植季次": "第二季", "总产量/斤": 36000},
    {"作物编号": 38, "作物名称": "榆黄菇", "种植季次": "第二季", "总产量/斤": 9000},
    {"作物编号": 39, "作物名称": "香菇", "种植季次": "第二季", "总产量/斤": 7200},
    {"作物编号": 40, "作物名称": "白灵菇", "种植季次": "第二季", "总产量/斤": 18000},
    {"作物编号": 41, "作物名称": "羊肚菌", "种植季次": "第二季", "总产量/斤": 4200}
]

df = pd.DataFrame(data)

# 保存为Excel文件（路径可自定义，如"./2023年农作物产量.xlsx"）
df.to_excel("D:/2024_c/2023年各季次农作物实际产量.xlsx", index=False)





'''





'''问题一
'''

