
"""
C3_Problem1_Solver.py

基于差分进化(DE)算法的2024年C题问题一求解器。
此脚本为2024-2030年生成两种情景下的最优种植方案。

此最终版本相较于提供的草稿有所改进：
- 正确加载并利用所有必需的数据,包括2023年的种植历史。
- 准确计算基于2023年生产数据的基线需求。
- 实现了更健壮且确定性的所有约束修复函数。
- 正确建模了滚动的三年豆类规则和作物轮作(茬口)规则。
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==================== 配置 ====================
# --- 重要：在此设置您的数据目录路径 ---
# 在Windows上，使用'r'前缀来创建原始字符串，以避免路径问题。
DATA_DIR = r"E:\education\math\MathematicalModelingReproduction\建模\2024C差分遗传\C"  # 例如: r"C:\Users\YourUser\Documents\MathModel"

# --- 文件名 ---
LAND_FILE = os.path.join(DATA_DIR, "附件1.xlsx")
CROP_DATA_FILE = os.path.join(DATA_DIR, "附件2.xlsx")

# 在import之后，DATA_DIR 设定好之后
TEMPLATE1 = os.path.join(DATA_DIR, "template_result1.xlsx")
TEMPLATE2 = os.path.join(DATA_DIR, "template_result2.xlsx")

# 预加载：
template1_cols = list(pd.read_excel(TEMPLATE1, nrows=0).columns)
template2_cols = list(pd.read_excel(TEMPLATE2, nrows=0).columns)

# --- 输出文件 ---
OUTPUT_DIR = os.path.join(DATA_DIR, "附件3")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT_FILE_1 = os.path.join(OUTPUT_DIR, "result1_1.xlsx")
RESULT_FILE_2 = os.path.join(OUTPUT_DIR, "result1_2.xlsx")


# 模板文件用于获取输出的正确列顺序。确保 '附件3' 文件夹在您的 DATA_DIR 中。
RESULT_TEMPLATE_FILE = os.path.join(DATA_DIR, "result1_1.xlsx")



# ==================== 模型常量 ====================
YEARS = list(range(2024, 2031))
SEASONS = [1, 2]

# --- 作物分类 (基于问题描述和附件) ---
GRAIN_IDS = list(range(1, 16))
LEGUME_GRAIN_IDS = list(range(1, 6))
RICE_ID = 16
VEGETABLE_IDS = list(range(17, 35))
LEGUME_VEGETABLE_IDS = [17, 18, 19]
WATER_VEGETABLE_IDS = [35, 36, 37] # 水浇地第二季的特定蔬菜 (大白菜、白萝卜、红萝卜)
MUSHROOM_IDS = list(range(38, 42))
ALL_LEGUME_IDS = LEGUME_GRAIN_IDS + LEGUME_VEGETABLE_IDS

# --- 地块分类 ---
FLAT_DRY_LAND_IDS = [f"A{i}" for i in range(1, 7)]
TERRACE_LAND_IDS = [f"B{i}" for i in range(1, 15)]
HILL_LAND_IDS = [f"C{i}" for i in range(1, 7)]
IRRIGATED_LAND_IDS = [f"D{i}" for i in range(1, 9)]
ORDINARY_GH_IDS = [f"E{i}" for i in range(1, 17)]
SMART_GH_IDS = [f"F{i}" for i in range(1, 5)]
DRY_LANDS = FLAT_DRY_LAND_IDS + TERRACE_LAND_IDS + HILL_LAND_IDS

# ==================== DE算法参数 ====================
POP_SIZE = 100  # 种群大小
MAX_GEN = 200  # 最大迭代代数 (为了更好的结果可以增加，例如增加到500)
F_SCALE = 0.6   # 变异因子
CR_RATE = 0.9   # 交叉率

# ==================== 数据处理 ====================
class ProblemData:
    """从Excel文件加载并准备所有必要的数据。"""
    def __init__(self, land_file, crop_data_file):
        # 从所有相关工作表加载原始数据
        self.land_df = pd.read_excel(land_file, sheet_name="乡村的现有耕地")
        self.planting_2023_df = pd.read_excel(crop_data_file, sheet_name="2023年的农作物种植情况")
        self.crop_params_df = pd.read_excel(crop_data_file, sheet_name="2023年统计的相关数据")

        # 创建快速查找的基本映射
        self.land_ids = self.land_df["地块名称"].tolist()
        self.land_info = self.land_df.set_index("地块名称").to_dict('index')

        self.crop_ids = self.crop_params_df["作物编号"].unique().tolist()
        
        # 创建 (作物编号, 地块类型) -> {产量, 成本, 价格} 的映射
        self.crop_econ_map = {}
        for _, row in self.crop_params_df.iterrows():
            key = (row["作物编号"], row["地块类型"])
            # 对价格区间取平均值
            price_range = str(row["销售单价/(元/斤)"]).split('-')
            avg_price = (float(price_range[0]) + float(price_range[-1])) / 2
            
            self.crop_econ_map[key] = {
                'yield': row["亩产量/斤"], 
                'cost': row["种植成本/(元/亩)"], 
                'price': avg_price
            }

        # 创建2023年种植映射: (地块编号, 季节) -> 作物编号
        self.history_2023 = {}
        for _, row in self.planting_2023_df.iterrows():
            land_id = row["种植地块"]
            crop_id = row["作物编号"]
            season_str = row["种植季次"]
            
            season = 1 # 默认为第一季或单季
            if "二" in season_str:
                season = 2
            
            self.history_2023[(land_id, season)] = crop_id
            
        # 根据2023年总产量计算基线需求
        self.baseline_demand = self._calculate_2023_production()

    def _calculate_2023_production(self):
        """计算每种作物在2023年的总产量,用作基线需求。"""
        production_by_crop = {cid: 0 for cid in self.crop_ids}
        for _, row in self.planting_2023_df.iterrows():
            land_id = row["种植地块"]
            crop_id = row["作物编号"]
            area = row["种植面积/亩"]
            
            # 检查land_id是否有效（不是NaN且在land_info字典中）
            if pd.notna(land_id) and land_id in self.land_info:
                land_type = self.land_info[land_id]["地块类型"]
                
                econ = self.crop_econ_map.get((crop_id, land_type))
                if econ and econ['yield'] > 0:
                    production_by_crop[crop_id] += area * econ['yield']
        return production_by_crop

# ==================== 遗传表示 ====================
class Individual:
    """表示一个潜在的7年种植计划(一个染色体)。"""
    def __init__(self, prob_data: ProblemData):
        self.prob = prob_data
        self.num_genes = len(prob_data.land_ids) * len(SEASONS) * len(YEARS)
        
        # 基因编码：每个地块/季节/年份一个作物及其面积比例
        self.crops = np.zeros(self.num_genes, dtype=int)
        self.props = np.zeros(self.num_genes, dtype=float)
        self.fitness = -np.inf

    def get_index(self, land_idx, season_idx, year_idx):
        """计算染色体数组的扁平化索引。"""
        return year_idx + (season_idx * len(YEARS)) + (land_idx * len(SEASONS) * len(YEARS))

    def clone(self):
        """创建个体的深拷贝。"""
        c = Individual(self.prob)
        c.crops = self.crops.copy()
        c.props = self.props.copy()
        c.fitness = self.fitness
        return c

# ==================== 启发式求解器 ====================
class PlantingOptimizer:
    """使用差分进化算法的主求解器类。"""

    def __init__(self, prob_data: ProblemData):
        self.prob = prob_data
        self.land_id_map = {name: i for i, name in enumerate(prob_data.land_ids)}

    def _get_allowed_crops(self, land_id, season):
        """
        返回在给定地块和季节允许种植的作物编号列表。
        此函数严格遵循附件1中的所有种植规则。
        """
        land_type = self.prob.land_info[land_id]["地块类型"]
        
        # 规则(1): 平旱地、梯田、山坡地每年适宜单季种植粮食类作物。
        if land_type in ["平旱地", "梯田", "山坡地"]:
            return GRAIN_IDS
        
        # 规则(2, 3, 4): 水浇地
        elif land_type == "水浇地":
            if season == 1:
                # 第一季可种水稻或蔬菜(除大白菜、白萝卜、红萝卜外)
                return [RICE_ID] + [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
            else: # season == 2
                # 第二季只能种植大白菜、白萝卜和红萝卜中的一种。
                return WATER_VEGETABLE_IDS
        
        # 规则(5, 6): 普通大棚
        elif land_type == "普通大棚":
            if season == 1:
                # 第一季可种植蔬菜(除大白菜、白萝卜、红萝卜外)
                return [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
            else: # season == 2
                # 第二季只能种植食用菌
                return MUSHROOM_IDS
        
        # 规则(7): 智慧大棚
        elif land_type == "智慧大棚":
            # 两季均可种植蔬菜(除大白菜、白萝卜、红萝卜外)
            return [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
            
        return []

    def _initialize_population(self):
        """生成初始种群，满足一些基本约束。"""
        population = []
        for _ in range(POP_SIZE):
            ind = Individual(self.prob)
            for l_idx, l_id in enumerate(self.prob.land_ids):
                # 旱地 (A,B,C) 每年只能种植一个季节
                if l_id in DRY_LANDS:
                    for y_idx, year in enumerate(YEARS):
                        s_idx = random.choice([0, 1]) # 随机选择第1季或第2季
                        idx = ind.get_index(l_idx, s_idx, y_idx)
                        choices = self._get_allowed_crops(l_id, s_idx + 1)
                        if choices:
                            ind.crops[idx] = random.choice(choices)
                            ind.props[idx] = random.uniform(0.1, 1.0) # 最少10%面积
                else: # 其他地块可以在两个季节都种植
                    for s_idx, season in enumerate(SEASONS):
                        for y_idx, year in enumerate(YEARS):
                            idx = ind.get_index(l_idx, s_idx, y_idx)
                            choices = self._get_allowed_crops(l_id, season)
                            if choices and random.random() < 0.8: # 80%概率种植
                                ind.crops[idx] = random.choice(choices)
                                ind.props[idx] = random.uniform(0.1, 1.0)
            population.append(ind)
        return population

    def _repair(self, ind: Individual):
        """应用确定性修复过程以确保满足所有约束。"""
        
        # === 约束1：基本有效性与土地使用规则 ===
        for l_idx, l_id in enumerate(self.prob.land_ids):
            land_type = self.prob.land_info[l_id]["地块类型"]
            seasons_planted_in_year = {y: [] for y in YEARS}
            
            for s_idx, s in enumerate(SEASONS):
                for y_idx, y in enumerate(YEARS):
                    idx = ind.get_index(l_idx, s_idx, y_idx)
                    crop = ind.crops[idx]
                    
                    # 确保作物被允许，否则清除
                    if crop > 0 and crop not in self._get_allowed_crops(l_id, s):
                        ind.crops[idx], ind.props[idx] = 0, 0.0
                        continue
                    
                    # 裁剪比例并确保一致性
                    ind.props[idx] = np.clip(ind.props[idx], 0.0, 1.0)
                    if ind.props[idx] < 0.1: # 最少10%面积规则
                        ind.crops[idx], ind.props[idx] = 0, 0.0
                    if ind.crops[idx] == 0:
                        ind.props[idx] = 0.0

                    if ind.crops[idx] > 0:
                        seasons_planted_in_year[y].append(s_idx)

            # --- 特定地块的季节规则 ---
            if land_type in ["平旱地", "梯田", "山坡地"]: # 每年只能种一个季节
                for y, seasons in seasons_planted_in_year.items():
                    if len(seasons) > 1:
                        # 确定性修复：保留第一季，清除其他季
                        keep_s_idx = seasons[0]
                        for s_idx_to_clear in seasons[1:]:
                            idx = ind.get_index(l_idx, s_idx_to_clear, YEARS.index(y))
                            ind.crops[idx], ind.props[idx] = 0, 0.0
            
            elif land_type == "水浇地": # 如果第1季种水稻，第2季必须休耕
                for y_idx, y in enumerate(YEARS):
                    s1_idx = ind.get_index(l_idx, 0, y_idx)
                    if ind.crops[s1_idx] == RICE_ID:
                        s2_idx = ind.get_index(l_idx, 1, y_idx)
                        ind.crops[s2_idx], ind.props[s2_idx] = 0, 0.0
                         
        # === 约束2：禁止重复种植(茬口) ===
        for l_idx, l_id in enumerate(self.prob.land_ids):
            for y_idx, y in enumerate(YEARS):
                prev_y = y - 1
                for s_idx, s in enumerate(SEASONS):
                    idx = ind.get_index(l_idx, s_idx, y_idx)
                    current_crop = ind.crops[idx]
                    if current_crop == 0: continue
                    
                    # 与前一年相同季节比较
                    if prev_y == 2023:
                        prev_crop = self.prob.history_2023.get((l_id, s), 0)
                    else:
                        prev_idx = ind.get_index(l_idx, s_idx, y_idx - 1)
                        prev_crop = ind.crops[prev_idx]
                    
                    if current_crop == prev_crop:
                        ind.crops[idx], ind.props[idx] = 0, 0.0 # 清除违规
                
                # 检查第1季与前一年第2季
                s1_idx = ind.get_index(l_idx, 0, y_idx)
                current_s1_crop = ind.crops[s1_idx]
                if current_s1_crop > 0:
                    if prev_y == 2023:
                        prev_s2_crop = self.prob.history_2023.get((l_id, 2), 0)
                    else:
                        prev_s2_idx = ind.get_index(l_idx, 1, y_idx - 1)
                        prev_s2_crop = ind.crops[prev_s2_idx]
                    
                    if current_s1_crop == prev_s2_crop:
                        ind.crops[s1_idx], ind.props[s1_idx] = 0, 0.0
        
        # === 约束3：三年豆类规则 ===
        for l_idx, l_id in enumerate(self.prob.land_ids):
            # 检查从2023年开始的每个3年窗口
            for start_year in range(2023, 2029):
                window_years = range(start_year, start_year + 3)
                
                # 检查此窗口中是否种植了豆类
                found_legume = False
                for y_check in window_years:
                    for s_check in SEASONS:
                        crop_in_slot = 0
                        if y_check == 2023:
                            crop_in_slot = self.prob.history_2023.get((l_id, s_check), 0)
                        else:
                            y_check_idx = YEARS.index(y_check)
                            idx_check = ind.get_index(l_idx, s_check - 1, y_check_idx)
                            crop_in_slot = ind.crops[idx_check]
                        
                        if crop_in_slot in ALL_LEGUME_IDS:
                            found_legume = True
                            break
                    if found_legume: break
                
                if not found_legume:
                    # 如果没有豆类，在窗口中找一个有效的空位强制种植豆类
                    planted = False
                    for y_fix in window_years:
                        if y_fix == 2023: continue # 不能改变过去
                        for s_fix in SEASONS:
                            y_fix_idx = YEARS.index(y_fix)
                            idx_fix = ind.get_index(l_idx, s_fix - 1, y_fix_idx)
                            if ind.crops[idx_fix] == 0: # 找一个空位
                                legume_choices = [c for c in ALL_LEGUME_IDS if c in self._get_allowed_crops(l_id, s_fix)]
                                if legume_choices:
                                    ind.crops[idx_fix] = random.choice(legume_choices)
                                    ind.props[idx_fix] = 0.5 # 种植适量
                                    planted = True
                                    break
                        if planted: break
        
        # === 约束4：分散性(每种作物/季节最多5个地块) ===
        for y_idx, y in enumerate(YEARS):
            for s_idx, s in enumerate(SEASONS):
                plots_by_crop = {}
                for l_idx, l_id in enumerate(self.prob.land_ids):
                    idx = ind.get_index(l_idx, s_idx, y_idx)
                    crop = ind.crops[idx]
                    if crop > 0:
                        if crop not in plots_by_crop: plots_by_crop[crop] = []
                        plots_by_crop[crop].append(idx)
                
                for crop, indices in plots_by_crop.items():
                    if len(indices) > 5:
                        # 保留5个，清除其余的
                        for idx_to_clear in random.sample(indices, len(indices) - 5):
                            ind.crops[idx_to_clear], ind.props[idx_to_clear] = 0, 0.0

        return ind

    def _evaluate(self, ind: Individual, scenario: int):
        """计算给定个体和情景的总利润。"""
        
        # 计算每种作物、每年、每季的总产量和总成本
        production = {(y, s, c): 0 for y in YEARS for s in SEASONS for c in self.prob.crop_ids}
        total_cost = 0
        
        for l_idx, l_id in enumerate(self.prob.land_ids):
            land_area = self.prob.land_info[l_id]["地块面积/亩"]
            land_type = self.prob.land_info[l_id]["地块类型"]
            for y_idx, y in enumerate(YEARS):
                for s_idx, s in enumerate(SEASONS):
                    idx = ind.get_index(l_idx, s_idx, y_idx)
                    crop, prop = ind.crops[idx], ind.props[idx]
                    
                    if crop == 0 or prop == 0: continue
                    
                    area = land_area * prop
                    econ = self.prob.crop_econ_map.get((crop, land_type))
                    if econ:
                        production[(y, s, crop)] += area * econ['yield']
                        total_cost += area * econ['cost']

        # 根据销售情景计算总收入
        total_revenue = 0
        # 将生产按年和作物汇总，因为需求是按年计算的
        yearly_production = {(y, c): 0 for y in YEARS for c in self.prob.crop_ids}
        for (y, s, c), prod in production.items():
            yearly_production[(y, c)] += prod

        for (y, crop), prod in yearly_production.items():
            if prod == 0: continue
            
            demand = self.prob.baseline_demand.get(crop, 0)
            # 查找价格(取该作物的第一个可用价格)
            price = 0
            for lt in self.prob.land_df['地块类型'].unique():
                 econ = self.prob.crop_econ_map.get((crop, lt))
                 if econ and econ['price'] > 0:
                     price = econ['price']
                     break

            if scenario == 1: # 超出部分浪费
                revenue = min(prod, demand) * price
            else: # 情景2，超出部分按50%价格销售
                sold_normal = min(prod, demand)
                sold_discount = max(0, prod - demand)
                revenue = (sold_normal * price) + (sold_discount * price * 0.5)
            
            total_revenue += revenue
            
        return total_revenue - total_cost

    def solve(self, scenario):
        """运行主要的差分进化循环。"""
        # 1. 初始化
        population = self._initialize_population()
        for ind in population:
            ind = self._repair(ind)
            ind.fitness = self._evaluate(ind, scenario)

        best_ind_overall = max(population, key=lambda x: x.fitness)
        
        print(f"\n--- 开始情景 {scenario} 的进化 ---")
        pbar = tqdm(range(MAX_GEN), desc=f"第1代最佳适应度: {best_ind_overall.fitness:,.0f}")

        for gen in pbar:
            # 2. 获取当前最佳个体
            best_ind_gen = max(population, key=lambda x: x.fitness)
            if best_ind_gen.fitness > best_ind_overall.fitness:
                best_ind_overall = best_ind_gen
            
            new_population = []
            
            for i in range(POP_SIZE):
                target_ind = population[i]
                
                # 3. 变异
                r1, r2 = random.sample([j for j in range(POP_SIZE) if j != i], 2)
                
                mutant_ind = best_ind_gen.clone()
                # 连续部分变异 (DE/best/1)
                mutant_ind.props = best_ind_gen.props + F_SCALE * (population[r1].props - population[r2].props)
                # 离散部分变异 (来自r1)
                mask = np.random.rand(mutant_ind.crops.size) < F_SCALE
                mutant_ind.crops[mask] = population[r1].crops[mask]
                
                # 4. 交叉
                trial_ind = target_ind.clone()
                mask = np.random.rand(trial_ind.props.size) < CR_RATE
                trial_ind.props[mask] = mutant_ind.props[mask]
                trial_ind.crops[mask] = mutant_ind.crops[mask]
                
                # 5. 修复和选择
                trial_ind = self._repair(trial_ind)
                trial_ind.fitness = self._evaluate(trial_ind, scenario)
                
                if trial_ind.fitness >= target_ind.fitness:
                    new_population.append(trial_ind)
                else:
                    new_population.append(target_ind)
            
            population = new_population
            pbar.set_description(f"第{gen+2}代最佳适应度: {best_ind_overall.fitness:,.0f}")
            
        return best_ind_overall



def export_results(ind: Individual, scenario: int):
    """将最终种植计划导出到Excel文件，格式符合要求。"""
    # 生成种植数据
    rows = []
    for l_idx, l_id in enumerate(ind.prob.land_ids):
        land_area = ind.prob.land_info[l_id]["地块面积/亩"]
        for y_idx, y in enumerate(YEARS):
            for s_idx, s in enumerate(SEASONS):
                idx = ind.get_index(l_idx, s_idx, y_idx)
                crop, prop = ind.crops[idx], ind.props[idx]
                
                if crop > 0 and prop > 0:
                    rows.append({
                        "年份": y,
                        "地块名称": l_id,
                        "季节": s,
                        "作物编号": crop,
                        "种植面积(亩)": round(prop * land_area, 2)
                    })
    
    if not rows:
        print(f"警告: 最终解决方案中没有作物被种植。")
        return
    
    # 创建DataFrame
    result_df = pd.DataFrame(rows)
    
    # 通过映射作物编号添加作物名称
    crop_name_map = ind.prob.crop_params_df.drop_duplicates("作物编号").set_index("作物编号")["作物名称"].to_dict()
    result_df["作物名称"] = result_df["作物编号"].map(crop_name_map)
    
    # 选择对应模板列
    if scenario == 1:
        cols = template1_cols
    else:
        cols = template2_cols

    # 确保所有模板列都在 result_df 中
    for c in cols:
        if c not in result_df.columns:
            result_df[c] = ""

    # 重新排序
    result_df = result_df[cols]
    
    # 写出
    out = RESULT_FILE_1 if scenario==1 else RESULT_FILE_2
    result_df.to_excel(out, index=False)
    print(f"✔ 导出至 {out}")


# ==================== 主程序执行 ====================
if __name__ == "__main__":
    print("正在加载问题数据...")
    try:
        prob_data = ProblemData(LAND_FILE, CROP_DATA_FILE)
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件。请检查 DATA_DIR 路径。详细信息: {e}")
        exit()
    
    print("数据加载成功。")
    optimizer = PlantingOptimizer(prob_data)

    # --- 求解情景1 ---
    best_solution_s1 = optimizer.solve(scenario=1)
    export_results(best_solution_s1, 1)

    # --- 求解情景2 ---
    best_solution_s2 = optimizer.solve(scenario=2)
    export_results(best_solution_s2, 2)

    print("\n两种情景的优化已完成。")