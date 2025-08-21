"""
农作物种植策略优化 - 差分进化遗传算法 (DEGA)
2024年高教社杯全国大学生数学建模竞赛 C题解决方案

主要改进：
1. [MODIFIED] 直接从 '各类农作物总需求量与总成本.xlsx' 文件读取预计算的需求量。
2. [MODIFIED] 更新文件路径以匹配项目结构，并自动创建输出目录。
3. [MODIFIED] 重写 save_results 函数以生成宽表格式的 Excel 输出，每年一个工作表。
4. [IMPROVED] 增加了对价格范围（如 "2.5-4.0"）的处理，取平均值。
5. [RETAINED] 保留并验证了与图片模型一致的12个约束条件的实现。
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 文件路径配置 (确保此脚本的运行目录包含 "C" 子目录)
DATA_DIR = r"./C"
LAND_FILE = os.path.join(DATA_DIR, "附件1.xlsx")
CROP_FILE = os.path.join(DATA_DIR, "附件2.xlsx")
# [NEW] 直接读取预计算的需求文件
DEMAND_FILE = os.path.join(DATA_DIR, "各类农作物总需求量与总成本.xlsx")


# [MODIFIED] 创建输出目录并设置输出文件路径
OUTPUT_DIR = os.path.join(DATA_DIR, "附件3")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT1_1 = os.path.join(OUTPUT_DIR, "result1_1.xlsx")  # 场景1：超出滞销
RESULT1_2 = os.path.join(OUTPUT_DIR, "result1_2.xlsx")  # 场景2：50%价格出售

# 时间范围
YEARS = list(range(2024, 2031))  # 2024-2030年
SEASONS = [1, 2]  # 第一季、第二季

# 作物分类
GRAIN_IDS = list(range(1, 16))        # 1-15: 粮食作物
LEGUME_GRAIN_IDS = list(range(1, 6))  # 1-5: 豆类粮食
RICE_ID = 16                          # 16: 水稻
VEGETABLE_IDS = list(range(17, 35))   # 17-34: 蔬菜
LEGUME_VEGETABLE_IDS = [17, 18, 19]   # 17-19: 豆类蔬菜
WATER_VEGETABLE_IDS = [35, 36, 37]    # 35-37: 水生蔬菜
MUSHROOM_IDS = [38, 39, 40, 41]       # 38-41: 食用菌

# 地块分类
FLAT_DRY_LANDS = [f"A{i}" for i in range(1, 7)]    # A1-A6: 平旱地
TERRACE_LANDS = [f"B{i}" for i in range(1, 15)]    # B1-B14: 梯田
HILL_LANDS = [f"C{i}" for i in range(1, 7)]        # C1-C6: 山坡地
IRRIGATED_LANDS = [f"D{i}" for i in range(1, 9)]   # D1-D8: 水浇地
ORDINARY_GH = [f"E{i}" for i in range(1, 17)]      # E1-E16: 普通大棚
SMART_GH = [f"F{i}" for i in range(1, 5)]          # F1-F4: 智慧大棚

# DEGA算法参数
POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_FACTOR = 0.5    # F参数
CROSSOVER_RATE = 0.9     # CR参数
ELITE_RATIO = 0.1        # 精英保留比例


class CropOptimizationProblem:
    """农作物种植优化问题数据处理类"""

    def __init__(self):
        # [FIX] Centralize problem dimensions (years, seasons) in this class
        self.years = YEARS
        self.seasons = SEASONS
        self.num_years = len(self.years)
        self.num_seasons = len(self.seasons)
        
        self.load_data()
        self.setup_constraints()

    def load_data(self):
        """[MODIFIED] 加载数据，直接从 '各类农作物总需求量与总成本.xlsx' 读取需求"""
        try:
            # ===== 读取地块信息（附件1.xlsx）=====
            # 假设附件1的第一个sheet包含'地块名称', '地块类型', '地块面积/亩'
            self.land_df = pd.read_excel(LAND_FILE, sheet_name=0)
            self.land_ids = self.land_df["地块名称"].tolist()
            self.land_types = dict(zip(self.land_df["地块名称"], self.land_df["地块类型"]))
            self.land_areas = dict(zip(self.land_df["地块名称"], self.land_df["地块面积/亩"]))

            # ===== 读取作物亩产量及成本信息（附件2.xlsx sheet1）=====
            crop_yield_df = pd.read_excel(CROP_FILE, sheet_name=1)
            self.crop_ids = crop_yield_df["作物编号"].tolist()
            # [NEW] 创建作物ID和名称的双向映射
            self.crop_id_to_name = dict(zip(crop_yield_df["作物编号"], crop_yield_df["作物名称"]))
            self.crop_name_to_id = dict(zip(crop_yield_df["作物名称"], crop_yield_df["作物编号"]))

            # 构建查找字典
            self.yield_dict = {}
            self.cost_dict = {}
            self.price_dict = {}
            for _, row in crop_yield_df.iterrows():
                key = (row["作物编号"], row["地块类型"])
                self.yield_dict[key] = row["亩产量/斤"]
                self.cost_dict[key] = row["种植成本/(元/亩)"]
                
                # [IMPROVED] 处理价格范围，取平均值
                price_val = row["销售单价/(元/斤)"]
                if isinstance(price_val, str) and '-' in price_val:
                    low, high = map(float, price_val.split('-'))
                    self.price_dict[key] = (low + high) / 2
                else:
                    self.price_dict[key] = float(price_val)

            # ===== [MODIFIED] 直接读取预计算的需求产量 =====
            demand_input_df = pd.read_excel(DEMAND_FILE)
            demand_input_df["作物编号"] = demand_input_df["作物名称"].map(self.crop_name_to_id)
            
            if demand_input_df["作物编号"].isnull().any():
                unmapped = demand_input_df[demand_input_df["作物编号"].isnull()]["作物名称"].tolist()
                warnings.warn(f"以下作物的需求量无法映射，将被忽略: {unmapped}")
                demand_input_df.dropna(subset=["作物编号"], inplace=True)
            
            demand_input_df["作物编号"] = demand_input_df["作物编号"].astype(int)
            
            self.demand_df = demand_input_df[["作物编号", "总需求量/斤"]].copy()
            self.demand_df.rename(columns={"总需求量/斤": "产量"}, inplace=True)

            print(f"✓ 成功加载数据：{len(self.land_ids)}个地块，{len(self.crop_ids)}种作物")
            print(f"✓ 从 '{os.path.basename(DEMAND_FILE)}' 加载了 {len(self.demand_df)} 条需求记录")

        except KeyError as e:
            print(f"✗ 数据加载失败：找不到列 {e}。请检查Excel文件 '附件1.xlsx' 是否包含 '地块面积/亩' 列。")
            raise
        except FileNotFoundError as e:
            print(f"✗ 数据加载失败：文件未找到 {e}。请确保所有数据文件都在 '{DATA_DIR}' 目录下。")
            raise
        except Exception as e:
            print(f"✗ 数据加载时发生未知错误：{e}")
            raise


    def setup_constraints(self):
        """设置约束条件"""
        self.allowed_crops_dict = {}
        for land in self.land_ids:
            for season in self.seasons:
                self.allowed_crops_dict[(land, season)] = self.get_allowed_crops(land, season)

    def get_allowed_crops(self, land: str, season: int) -> List[int]:
        """获取指定地块和季节允许种植的作物"""
        land_type_prefix = land[0]
        if land_type_prefix in ['A', 'B', 'C']: # 平旱地, 梯田, 山坡地
            return GRAIN_IDS
        elif land_type_prefix == 'D': # 水浇地
            if season == 1:
                return [RICE_ID] + VEGETABLE_IDS
            else:
                return VEGETABLE_IDS + WATER_VEGETABLE_IDS
        elif land_type_prefix == 'E': # 普通大棚
            if season == 1:
                return VEGETABLE_IDS
            else:
                return MUSHROOM_IDS
        elif land_type_prefix == 'F': # 智慧大棚
            # 蔬菜，但不包括水生蔬菜
            return [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
        return []

    def get_demand(self, crop_id: int, year: int) -> float:
        """
        获取指定作物的需求量（斤）
        需求量被假定为对所有年份（2024-2030）都相同。
        """
        demand_data = self.demand_df[self.demand_df["作物编号"] == crop_id]
        if not demand_data.empty:
            return float(demand_data["产量"].iloc[0])
        return 0.0


class Individual:
    """个体表示类"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
        # [FIX] Read dimensions from the problem object for consistency
        self.num_lands = len(problem.land_ids)
        self.num_seasons = problem.num_seasons
        self.num_years = problem.num_years
        self.total_vars = self.num_lands * self.num_seasons * self.num_years
        
        # 基因编码：作物ID和种植面积比例
        self.crop_genes = np.zeros(self.total_vars, dtype=int)
        self.area_genes = np.zeros(self.total_vars, dtype=float)
        
        self.fitness = -np.inf
        self.is_feasible = False
    
    def get_index(self, land_idx: int, season_idx: int, year_idx: int) -> int:
        """获取基因索引"""
        return year_idx + self.num_years * (season_idx + self.num_seasons * land_idx)
    
    def clone(self):
        """复制个体"""
        new_ind = Individual(self.problem)
        new_ind.crop_genes = self.crop_genes.copy()
        new_ind.area_genes = self.area_genes.copy()
        new_ind.fitness = self.fitness
        new_ind.is_feasible = self.is_feasible
        return new_ind


class ConstraintHandler:
    """约束处理器"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
    
    def repair(self, individual: Individual) -> Individual:
        """修复个体以满足所有约束"""
        self._basic_constraints(individual)
        self._land_specific_constraints(individual)
        self._crop_rotation_constraints(individual)
        self._legume_constraints(individual)
        self._dispersity_constraints(individual)
        self._yield_constraints(individual)
        
        return individual
    
    def _basic_constraints(self, ind: Individual):
        """基本约束：作物选择和面积限制 (对应模型约束 1, 2, 3)"""
        for land_idx, land in enumerate(self.problem.land_ids):
            for season_idx, season in enumerate(self.problem.seasons):
                for year_idx, year in enumerate(self.problem.years):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    
                    allowed = self.problem.allowed_crops_dict[(land, season)]
                    if ind.crop_genes[idx] not in allowed:
                        ind.crop_genes[idx] = 0
                    
                    ind.area_genes[idx] = np.clip(ind.area_genes[idx], 0.0, 1.0)
                    
                    if ind.crop_genes[idx] == 0:
                        ind.area_genes[idx] = 0.0
                    elif ind.area_genes[idx] < 0.5:
                        ind.area_genes[idx] = 0.5
    
    def _land_specific_constraints(self, ind: Individual):
        """地块特定约束 (对应模型约束 4, 6, 7, 8, 9, 10)"""
        # A/B/C类地块每年只能种一季
        abc_lands = FLAT_DRY_LANDS + TERRACE_LANDS + HILL_LANDS
        
        for land_idx, land in enumerate(self.problem.land_ids):
            if land in abc_lands:
                for year_idx, year in enumerate(self.problem.years):
                    idx1 = ind.get_index(land_idx, 0, year_idx)
                    idx2 = ind.get_index(land_idx, 1, year_idx)
                    if ind.crop_genes[idx1] != 0 and ind.crop_genes[idx2] != 0:
                        if random.random() < 0.5:
                            ind.crop_genes[idx2], ind.area_genes[idx2] = 0, 0.0
                        else:
                            ind.crop_genes[idx1], ind.area_genes[idx1] = 0, 0.0
        
        for land_idx, land in enumerate(self.problem.land_ids):
            if land in IRRIGATED_LANDS:
                for year_idx in range(len(self.problem.years)):
                    idx1 = ind.get_index(land_idx, 0, year_idx)
                    idx2 = ind.get_index(land_idx, 1, year_idx)
                    if ind.crop_genes[idx1] == RICE_ID: # 第一季种水稻，第二季必须休耕
                        ind.crop_genes[idx2], ind.area_genes[idx2] = 0, 0.0
    
    def _crop_rotation_constraints(self, ind: Individual):
        """轮作约束：防止重茬"""
        for land_idx in range(len(self.problem.land_ids)):
            for season_idx in range(len(self.problem.seasons)):
                for year_idx in range(len(self.problem.years) - 1):
                    idx1 = ind.get_index(land_idx, season_idx, year_idx)
                    idx2 = ind.get_index(land_idx, season_idx, year_idx + 1)
                    if ind.crop_genes[idx1] != 0 and ind.crop_genes[idx1] == ind.crop_genes[idx2]:
                        ind.crop_genes[idx2], ind.area_genes[idx2] = 0, 0.0
            
            for year_idx in range(len(self.problem.years) - 1):
                idx1 = ind.get_index(land_idx, 1, year_idx)
                idx2 = ind.get_index(land_idx, 0, year_idx + 1)
                if ind.crop_genes[idx1] != 0 and ind.crop_genes[idx1] == ind.crop_genes[idx2]:
                    ind.crop_genes[idx2], ind.area_genes[idx2] = 0, 0.0
    
    def _legume_constraints(self, ind: Individual):
        """豆类约束：三年内至少种一次豆类 (对应模型约束 5)"""
        legume_crops = set(LEGUME_GRAIN_IDS + LEGUME_VEGETABLE_IDS)
        for land_idx, land in enumerate(self.problem.land_ids):
            for start_year_idx in range(len(self.problem.years) - 2):
                window_years = range(start_year_idx, start_year_idx + 3)
                has_legume = any(
                    ind.crop_genes[ind.get_index(land_idx, s_idx, y_idx)] in legume_crops
                    for y_idx in window_years for s_idx in range(len(self.problem.seasons))
                )
                if not has_legume:
                    # 在窗口第一年随机一季强制种豆类
                    season_to_plant_idx = random.choice([0, 1])
                    season_to_plant = self.problem.seasons[season_to_plant_idx]
                    idx_to_plant = ind.get_index(land_idx, season_to_plant_idx, start_year_idx)
                    allowed = self.problem.allowed_crops_dict[(land, season_to_plant)]
                    available_legumes = list(legume_crops.intersection(allowed))
                    if available_legumes:
                        ind.crop_genes[idx_to_plant] = random.choice(available_legumes)
                        ind.area_genes[idx_to_plant] = random.uniform(0.5, 1.0)

    def _dispersity_constraints(self, ind: Individual):
        """分散性约束:每种作物每季最多5个地块 (对应模型约束 11)"""
        for year_idx in range(len(self.problem.years)):
            for season_idx in range(len(self.problem.seasons)):
                crop_lands = {}
                for land_idx in range(len(self.problem.land_ids)):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    crop = ind.crop_genes[idx]
                    if crop != 0:
                        crop_lands.setdefault(crop, []).append(idx)
                
                for crop, indices in crop_lands.items():
                    if len(indices) > 5:
                        to_remove = random.sample(indices, len(indices) - 5)
                        for idx in to_remove:
                            ind.crop_genes[idx], ind.area_genes[idx] = 0, 0.0
    
    def _yield_constraints(self, ind: Individual):
        """产量约束：产量应满足至少90%的需求 (对应模型约束 12)"""
        for year_idx, year in enumerate(self.problem.years):
            for crop_id in self.problem.crop_ids:
                demand = self.problem.get_demand(crop_id, year)
                if demand <= 0:
                    continue
                
                required_production = 0.9 * demand
                total_production = 0
                
                # 计算当前总产量
                for land_idx, land in enumerate(self.problem.land_ids):
                    for season_idx in range(len(self.problem.seasons)):
                        idx = ind.get_index(land_idx, season_idx, year_idx)
                        if ind.crop_genes[idx] == crop_id:
                            land_type = self.problem.land_types[land]
                            land_area = self.problem.land_areas[land]
                            yield_per_acre = self.problem.yield_dict.get((crop_id, land_type), 0)
                            total_production += ind.area_genes[idx] * land_area * yield_per_acre
                
                # 如果产量不足，尝试增加面积
                if total_production < required_production:
                    shortfall = required_production - total_production
                    # 找到所有种植该作物的地块
                    candidate_indices = []
                    for land_idx, land in enumerate(self.problem.land_ids):
                        for season_idx in range(len(self.problem.seasons)):
                           idx = ind.get_index(land_idx, season_idx, year_idx)
                           if ind.crop_genes[idx] == crop_id:
                               candidate_indices.append((idx, land))

                    # 随机增加面积直到满足需求
                    random.shuffle(candidate_indices)
                    for idx, land in candidate_indices:
                        if shortfall <= 0: break
                        land_type = self.problem.land_types[land]
                        land_area = self.problem.land_areas[land]
                        yield_per_acre = self.problem.yield_dict.get((crop_id, land_type), 0)
                        if yield_per_acre > 0:
                            current_area_ratio = ind.area_genes[idx]
                            potential_increase_ratio = 1.0 - current_area_ratio
                            if potential_increase_ratio > 0:
                                production_per_ratio = land_area * yield_per_acre
                                if production_per_ratio > 0:
                                    required_ratio_increase = shortfall / production_per_ratio
                                    increase_ratio = min(potential_increase_ratio, required_ratio_increase)
                                    
                                    ind.area_genes[idx] += increase_ratio
                                    shortfall -= increase_ratio * production_per_ratio


class FitnessEvaluator:
    """适应度评估器"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
    
    def evaluate(self, individual: Individual, scenario: int = 1, alpha: float = 0.5) -> float:
        """评估个体适应度 (对应模型最大化目标)"""
        total_profit = 0.0
        
        for year_idx, year in enumerate(self.problem.years):
            year_production = {} # {crop_id: production}
            year_cost = 0.0
            
            # 1. 计算该年度的总成本和总产量
            for land_idx, land in enumerate(self.problem.land_ids):
                for season_idx in range(len(self.problem.seasons)):
                    idx = individual.get_index(land_idx, season_idx, year_idx)
                    crop_id = individual.crop_genes[idx]
                    area_ratio = individual.area_genes[idx]
                    
                    if crop_id > 0 and area_ratio > 0:
                        land_type = self.problem.land_types[land]
                        land_area = self.problem.land_areas[land]
                        
                        planted_area = area_ratio * land_area
                        cost_per_acre = self.problem.cost_dict.get((crop_id, land_type), 0)
                        yield_per_acre = self.problem.yield_dict.get((crop_id, land_type), 0)
                        
                        year_cost += planted_area * cost_per_acre
                        production = planted_area * yield_per_acre
                        year_production[crop_id] = year_production.get(crop_id, 0) + production

            # 2. 计算该年度的总收入
            year_revenue = 0.0
            for crop_id, production in year_production.items():
                demand = self.problem.get_demand(crop_id, year)
                
                # 寻找该作物的价格 (简化：假设价格不依赖于地块类型)
                price = 0
                for land_type in set(self.problem.land_types.values()):
                    price = self.problem.price_dict.get((crop_id, land_type), 0)
                    if price > 0:
                        break
                
                if price > 0:
                    if demand > 0:
                        if scenario == 1: # 超出部分滞销
                            saleable = min(production, demand)
                            year_revenue += saleable * price
                        else: # 场景2：超出部分半价出售
                            saleable = min(production, demand)
                            excess = max(0, production - demand)
                            year_revenue += saleable * price + excess * price * alpha
                    else: # 无需求，假设全部可售
                        year_revenue += production * price

            total_profit += (year_revenue - year_cost)
            
        return total_profit


class DEGAOptimizer:
    """差分进化-遗传算法优化器"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
        self.constraint_handler = ConstraintHandler(problem)
        self.fitness_evaluator = FitnessEvaluator(problem)
    
    def initialize_individual(self) -> Individual:
        """初始化个体"""
        individual = Individual(self.problem)
        for idx in range(individual.total_vars):
            year_idx = idx % self.problem.num_years
            season_idx = (idx // self.problem.num_years) % self.problem.num_seasons
            land_idx = idx // (self.problem.num_years * self.problem.num_seasons)
            
            land = self.problem.land_ids[land_idx]
            season = self.problem.seasons[season_idx]
            
            allowed_crops = self.problem.allowed_crops_dict[(land, season)]
            if allowed_crops and random.random() < 0.7:
                individual.crop_genes[idx] = random.choice(allowed_crops)
                individual.area_genes[idx] = random.uniform(0.5, 1.0)
        
        return self.constraint_handler.repair(individual)
    
    def mutate(self, population: List[Individual], best: Individual) -> List[Individual]:
        """变异操作"""
        mutants = []
        n = len(population)
        for i in range(n):
            r1, r2 = random.sample([j for j in range(n) if j != i], 2)
            mutant = population[i].clone()
            
            # 差分变异
            mutant.area_genes = best.area_genes + MUTATION_FACTOR * (population[r1].area_genes - population[r2].area_genes)
            mask = np.random.rand(len(mutant.crop_genes)) < MUTATION_FACTOR
            mutant.crop_genes[mask] = population[r1].crop_genes[mask]
            mutants.append(mutant)
        return mutants
    
    def crossover(self, target: Individual, mutant: Individual) -> Individual:
        """交叉操作"""
        trial = target.clone()
        mask = np.random.rand(len(trial.crop_genes)) < CROSSOVER_RATE
        trial.crop_genes[mask] = mutant.crop_genes[mask]
        trial.area_genes[mask] = mutant.area_genes[mask]
        return trial
    
    def optimize(self, scenario: int = 1, max_stagnation: int = 50) -> Individual:
        """主优化循环"""
        print(f"开始优化场景 {scenario}...")
        population = [self.initialize_individual() for _ in range(POPULATION_SIZE)]
        for ind in population:
            ind.fitness = self.fitness_evaluator.evaluate(ind, scenario)
        
        best_fitness_history = [-np.inf]
        stagnation_count = 0
        
        for generation in range(MAX_GENERATIONS):
            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]
            
            if best.fitness > best_fitness_history[-1]:
                best_fitness_history.append(best.fitness)
                stagnation_count = 0
            else:
                best_fitness_history.append(best_fitness_history[-1])
                stagnation_count += 1
            
            if stagnation_count >= max_stagnation:
                print(f"在第 {generation+1} 代停止（停滞 {max_stagnation} 代）")
                break
            
            mutants = self.mutate(population, best)
            
            new_population = []
            for i in range(POPULATION_SIZE):
                trial = self.crossover(population[i], mutants[i])
                trial = self.constraint_handler.repair(trial)
                trial.fitness = self.fitness_evaluator.evaluate(trial, scenario)
                
                if trial.fitness > population[i].fitness:
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
            
            population = new_population
            
            if (generation + 1) % 50 == 0:
                print(f"第 {generation + 1}/{MAX_GENERATIONS} 代：最佳利润 = {best.fitness:,.2f}")
        
        final_best = max(population, key=lambda x: x.fitness)
        print(f"优化完成：最终最佳利润 = {final_best.fitness:,.2f}")
        return final_best


def save_results(individual: Individual, filename: str):
    """[MODIFIED] 保存结果到Excel，每年一个sheet，宽表格式"""
    results = []
    problem = individual.problem
    
    for land_idx, land in enumerate(problem.land_ids):
        land_area = problem.land_areas.get(land, 0)
        for season_idx, season in enumerate(problem.seasons):
            for year_idx, year in enumerate(problem.years):
                idx = individual.get_index(land_idx, season_idx, year_idx)
                crop_id = individual.crop_genes[idx]
                area_ratio = individual.area_genes[idx]
                
                if crop_id != 0 and area_ratio > 0:
                    actual_area = area_ratio * land_area
                    crop_name = problem.crop_id_to_name.get(crop_id, f"未知{crop_id}")
                    results.append({
                        "年份": year,
                        "地块名称": land,
                        "作物名称": crop_name,
                        "种植面积": round(actual_area, 3)
                    })
    
    if not results:
        print("✗ 警告：没有生成任何有效的种植方案，无法保存结果。")
        return

    df_long = pd.DataFrame(results)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        all_crop_names = sorted(list(problem.crop_id_to_name.values()))
        all_land_names = problem.land_ids

        for year in problem.years:
            df_year = df_long[df_long['年份'] == year]
            if df_year.empty:
                pivot_df = pd.DataFrame(index=all_land_names, columns=all_crop_names).fillna(0)
            else:
                pivot_df = df_year.pivot_table(
                    index='地块名称', columns='作物名称', values='种植面积', aggfunc='sum'
                )
                pivot_df = pivot_df.reindex(index=all_land_names, columns=all_crop_names).fillna(0)
            
            pivot_df.to_excel(writer, sheet_name=str(year))
            
    print(f"✓ 结果已保存至 {filename} (每个年份一个工作表)")


def main():
    """主函数"""
    print("="*60)
    print("      农作物种植策略优化 - 差分进化遗传算法 (DEGA)")
    print("="*60)
    
    try:
        problem = CropOptimizationProblem()
        optimizer = DEGAOptimizer(problem)
        
        print("\n--- 问题1 场景1: 超出部分滞销 ---")
        best1 = optimizer.optimize(scenario=1)
        save_results(best1, RESULT1_1)

        print("\n--- 问题1 场景2: 超出部分50%价格出售 ---")
        best2 = optimizer.optimize(scenario=2)
        save_results(best2, RESULT1_2)
        
        print("\n" + "="*60)
        print("优化全部完成！")
        print(f"场景1最优总利润 (2024-2030): {best1.fitness:,.2f} 元")
        print(f"场景2最优总利润 (2024-2030): {best2.fitness:,.2f} 元")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 程序执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()