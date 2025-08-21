import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyecharts.options as opt
import warnings
from pyecharts.globals import CurrentConfig, NotebookType
from pypalettes import load_cmap, get_hex  
from pyecharts.charts import Pie

warnings.filterwarnings('ignore') # 这行代码的作用是忽略所有的警告信息，通常是用来避免一些不必要的警告干扰代码的执行。
plt.rcParams['font.family'] = 'Kaiti' #设置楷体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK #这行代码设置了PyEcharts的输出类型为Jupyter Notebook，这样生成的图表可以直接在Jupyter Notebook中显示。

# 读取Excel文件，注意处理可能的路径和文件存在性问题
df1 = pd.read_excel('./C/附件1.xlsx', index_col=0, sheet_name=0)

# 查看实际列名
print(df1.columns)
df1 = df1.drop('说明 ', axis=1)

# 显示df1内容
print(df1)

df2 = pd.read_excel('./C/附件1.xlsx', index_col=0, sheet_name=1)
print(df2.columns)
df2 = df2.drop('说明', axis=1) # axis：指定删除的是行还是列。axis=0 表示删除行，axis=1 表示删除列
df2

island = df1.loc["A1":"D8"]

print(island.columns)
print(island)

group_type = island.groupby('地块类型').sum().reset_index() # reset_index() 方法用于将分组后的 DataFrame 的索引重置为默认整数索引，并将原来的索引列转换为普通列。
print(group_type)
lst = group_type.values.tolist() # 将 DataFrame 转换为列表形式
lst

outer  = []
for i in lst:
    name = i[0]
    select = island.query('地块类型 == @name')['地块面积/亩']
    num = 1
    for j in select.values.tolist():
        outer.append(
            (f'{name}_{num}', j)
            )
        num += 1
print(outer)

cmap = load_cmap('pastel', cmap_type='continuous')

pie = (
    Pie()
   .add(
        series_name="地块类型",
        data_pair=lst,
        radius=["0", "50%"],
        label_opts=opt.LabelOpts(
            position='inside', 
            is_show=True, 
            formatter='{b} {b}\n{d}%', 
            rich={"b": {"fontaize": 16, "fontFamily": "KaiTi"}},
           
        )
    )
   .set_colors(cmap.colors[:4])
   .add(
        series_name="地块子类",
        radius=["60%", "75%"],
        data_pair=outer,
        label_opts=opt.LabelOpts(
            position='outside', 
            is_show=True, 
            formatter="{b}:{d}%",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
        ),
    )
   .set_global_opts(
        legend_opts=opt.LegendOpts(is_show=False),
        title_opts=opt.TitleOpts(
            title="耕地结构", 
            subtitle="露天耕地结构", 
            pos_left="center"
        ),
    )
   .set_series_opts(
        tooltip_opts=opt.TooltipOpts(
            trigger="item", 
            formatter="{a} <br/>{b}: {c} ({d}%)"
        )
    )
)
pie.render_notebook()
pie.render("耕地结构.png")  # 保存为PNG图片（需额外依赖）
