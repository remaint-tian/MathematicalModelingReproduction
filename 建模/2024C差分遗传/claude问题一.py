"""
农作物种植策略优化 - 差分进化遗传算法 (DEGA)
2024年高教社杯全国大学生数学建模竞赛 C题解决方案

主要改进：
1. 完善约束处理机制
2. 优化种群初始化策略
3. 改进适应度函数
4. 增强可读性和维护性
5. 去掉 DEMAND_FILE，直接从 LAND_FILE 和 CROP_FILE 计算需求产量
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 文件路径配置
DATA_DIR = r"E:\education\math\MathematicalModelingReproduction\建模\2024C差分遗传\C"  # 当前目录，根据实际情况修改
LAND_FILE = os.path.join(DATA_DIR, "附件1.xlsx")  # 地块信息
CROP_FILE = os.path.join(DATA_DIR, "附件2.xlsx")  # 作物信息

# 输出文件
RESULT1_1 = os.path.join(DATA_DIR, "result1_1.xlsx")  # 场景1：超出滞销
RESULT1_2 = os.path.join(DATA_DIR, "result1_2.xlsx")  # 场景2：50%价格出售
RESULT2 = os.path.join(DATA_DIR, "result2.xlsx")      # 问题2：考虑不确定性

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
        self.load_data()
        self.setup_constraints()

    def load_data(self):
        """加载数据（去除 DEMAND_FILE，直接计算需求产量）"""
        try:
            # ===== 读取地块信息（附件1.xlsx sheet0）=====
            self.land_df = pd.read_excel(LAND_FILE, sheet_name=0)
            self.land_ids = self.land_df["地块名称"].tolist()
            self.land_types = dict(zip(self.land_df["地块名称"], self.land_df["地块类型"]))
            self.land_areas = dict(zip(self.land_df["地块名称"], self.land_df["地块面积/亩"]))

            # ===== 读取作物种植面积信息（附件2.xlsx sheet0）=====
            crop_area_df = pd.read_excel(CROP_FILE, sheet_name=0)  # 地块名称、作物名称、种植面积/亩

            # ===== 读取作物亩产量及成本信息（附件2.xlsx sheet1）=====
            crop_yield_df = pd.read_excel(CROP_FILE, sheet_name=1)
            self.crop_ids = crop_yield_df["作物编号"].tolist()

            # 构建查找字典
            self.yield_dict = {}
            self.cost_dict = {}
            self.price_dict = {}
            for _, row in crop_yield_df.iterrows():
                key = (row["作物编号"], row["地块类型"])
                self.yield_dict[key] = row["亩产量/斤"]
                self.cost_dict[key] = row["种植成本/(元/亩)"]
                self.price_dict[key] = row["销售单价/(元/斤)"]

            # ===== 计算需求产量 =====
            # 将地块类型合并到种植面积表
            merged_df = crop_area_df.merge(
                self.land_df[["地块名称", "地块类型"]],
                on="地块名称",
                how="left"
            ).merge(
                crop_yield_df[["作物编号", "作物名称", "地块类型", "亩产量/斤"]],
                on=["作物名称", "地块类型"],
                how="left"
            )

            # 计算需求产量（亩产量 × 种植面积）
            merged_df["需求产量/斤"] = merged_df["亩产量/斤"] * merged_df["种植面积/亩"]

            # 按作物编号汇总总需求产量
            demand_df = merged_df.groupby(["作物编号", "作物名称"], as_index=False)["需求产量/斤"].sum()
            demand_df.rename(columns={"需求产量/斤": "产量"}, inplace=True)

            # 存为类变量
            self.demand_df = demand_df

            print(f"✓ 成功加载数据：{len(self.land_ids)}个地块，{len(self.crop_ids)}种作物")
            print(f"✓ 计算得到 {len(self.demand_df)} 条需求产量记录")

        except Exception as e:
            print(f"✗ 数据加载失败：{e}")
            raise

    def setup_constraints(self):
        """设置约束条件"""
        self.allowed_crops_dict = {}
        for land in self.land_ids:
            for season in SEASONS:
                self.allowed_crops_dict[(land, season)] = self.get_allowed_crops(land, season)

    def get_allowed_crops(self, land: str, season: int) -> List[int]:
        """获取指定地块和季节允许种植的作物"""
        if land in FLAT_DRY_LANDS + TERRACE_LANDS + HILL_LANDS:
            return GRAIN_IDS
        elif land in IRRIGATED_LANDS:
            if season == 1:
                return [RICE_ID] + VEGETABLE_IDS
            else:
                return VEGETABLE_IDS + WATER_VEGETABLE_IDS
        elif land in ORDINARY_GH:
            if season == 1:
                return VEGETABLE_IDS
            else:
                return MUSHROOM_IDS
        elif land in SMART_GH:
            return [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
        return []

    def get_demand(self, crop_id: int, year: int) -> float:
        """
        获取指定作物的需求量（斤）
        当前实现不区分年份，因为需求量是按种植面积和亩产量计算的总量。
        """
        try:
            demand_data = self.demand_df[self.demand_df["作物编号"] == crop_id]
            if not demand_data.empty:
                return float(demand_data["产量"].iloc[0])
            else:
                return 0.0
        except Exception as e:
            print(f"警告: 无法获取作物ID={crop_id}的需求量: {e}")
            return 0.0




class Individual:
    """个体表示类"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
        self.num_lands = len(problem.land_ids)
        self.num_seasons = len(SEASONS)
        self.num_years = len(YEARS)
        self.total_vars = self.num_lands * self.num_seasons * self.num_years
        
        # 基因编码：作物ID和种植面积比例
        self.crop_genes = np.zeros(self.total_vars, dtype=int)
        self.area_genes = np.zeros(self.total_vars, dtype=float)
        
        self.fitness = None
        self.is_feasible = False
    
    def get_index(self, land_idx: int, season_idx: int, year_idx: int) -> int:
        """获取基因索引"""
        return (land_idx * self.num_seasons + season_idx) * self.num_years + year_idx
    
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
        """基本约束：作物选择和面积限制"""
        for land_idx, land in enumerate(self.problem.land_ids):
            for season_idx, season in enumerate(SEASONS):
                for year_idx, year in enumerate(YEARS):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    
                    # 约束1：作物必须在允许列表中
                    allowed = self.problem.allowed_crops_dict[(land, season)]
                    if ind.crop_genes[idx] not in allowed:
                        ind.crop_genes[idx] = 0
                    
                    # 约束2：面积比例在[0,1]范围内
                    ind.area_genes[idx] = np.clip(ind.area_genes[idx], 0.0, 1.0)
                    
                    # 约束3：无作物时面积为0，有作物时面积至少50%
                    if ind.crop_genes[idx] == 0:
                        ind.area_genes[idx] = 0.0
                    elif ind.area_genes[idx] < 0.5:
                        ind.area_genes[idx] = 0.5
    
    def _land_specific_constraints(self, ind: Individual):
        """地块特定约束"""
        # A/B/C类地块每年只能种一季
        abc_lands = FLAT_DRY_LANDS + TERRACE_LANDS + HILL_LANDS
        
        for land_idx, land in enumerate(self.problem.land_ids):
            if land in abc_lands:
                for year_idx, year in enumerate(YEARS):
                    idx1 = ind.get_index(land_idx, 0, year_idx)  # 第一季
                    idx2 = ind.get_index(land_idx, 1, year_idx)  # 第二季
                    
                    if ind.crop_genes[idx1] != 0 and ind.crop_genes[idx2] != 0:
                        # 随机保留一季
                        if random.random() < 0.5:
                            ind.crop_genes[idx2] = 0
                            ind.area_genes[idx2] = 0.0
                        else:
                            ind.crop_genes[idx1] = 0
                            ind.area_genes[idx1] = 0.0
        
        # 水浇地特殊规则
        for land_idx, land in enumerate(self.problem.land_ids):
            if land in IRRIGATED_LANDS:
                for year_idx in range(len(YEARS)):
                    idx1 = ind.get_index(land_idx, 0, year_idx)
                    idx2 = ind.get_index(land_idx, 1, year_idx)
                    
                    # 第一季种水稻，第二季必须休耕
                    if ind.crop_genes[idx1] == RICE_ID:
                        ind.crop_genes[idx2] = 0
                        ind.area_genes[idx2] = 0.0
        
        # 普通大棚约束
        for land_idx, land in enumerate(self.problem.land_ids):
            if land in ORDINARY_GH:
                for year_idx in range(len(YEARS)):
                    idx1 = ind.get_index(land_idx, 0, year_idx)
                    idx2 = ind.get_index(land_idx, 1, year_idx)
                    
                    # 第一季必须蔬菜
                    if ind.crop_genes[idx1] not in VEGETABLE_IDS:
                        ind.crop_genes[idx1] = random.choice(VEGETABLE_IDS)
                        ind.area_genes[idx1] = 0.5
                    
                    # 第二季必须食用菌
                    ind.crop_genes[idx2] = random.choice(MUSHROOM_IDS)
                    ind.area_genes[idx2] = 0.5
    
    def _crop_rotation_constraints(self, ind: Individual):
        """轮作约束：防止重茬"""
        for land_idx in range(len(self.problem.land_ids)):
            # 同季不同年不能重茬
            for season_idx in range(len(SEASONS)):
                for year_idx in range(len(YEARS) - 1):
                    idx1 = ind.get_index(land_idx, season_idx, year_idx)
                    idx2 = ind.get_index(land_idx, season_idx, year_idx + 1)
                    
                    if (ind.crop_genes[idx1] == ind.crop_genes[idx2] and 
                        ind.crop_genes[idx1] != 0):
                        ind.crop_genes[idx2] = 0
                        ind.area_genes[idx2] = 0.0
            
            # 跨季跨年不能重茬
            for year_idx in range(len(YEARS) - 1):
                idx1 = ind.get_index(land_idx, 1, year_idx)      # 当年第二季
                idx2 = ind.get_index(land_idx, 0, year_idx + 1)  # 次年第一季
                
                if (ind.crop_genes[idx1] == ind.crop_genes[idx2] and 
                    ind.crop_genes[idx1] != 0):
                    ind.crop_genes[idx2] = 0
                    ind.area_genes[idx2] = 0.0
    
    def _legume_constraints(self, ind: Individual):
        """豆类约束：三年内至少种一次豆类"""
        legume_crops = set(LEGUME_GRAIN_IDS + LEGUME_VEGETABLE_IDS)
        
        for land_idx, land in enumerate(self.problem.land_ids):
            # 检查每个三年窗口
            for start_year in range(len(YEARS) - 2):
                window_years = range(start_year, start_year + 3)
                
                # 检查窗口内是否有豆类
                has_legume = False
                for year_idx in window_years:
                    for season_idx in range(len(SEASONS)):
                        idx = ind.get_index(land_idx, season_idx, year_idx)
                        if ind.crop_genes[idx] in legume_crops:
                            has_legume = True
                            break
                    if has_legume:
                        break
                
                # 如果没有豆类，在第一年第一季强制种植
                if not has_legume:
                    idx = ind.get_index(land_idx, 0, start_year)
                    allowed = self.problem.allowed_crops_dict[(land, 1)]
                    available_legumes = list(legume_crops.intersection(allowed))
                    
                    if available_legumes:
                        ind.crop_genes[idx] = random.choice(available_legumes)
                        ind.area_genes[idx] = 0.5
    
    def _dispersity_constraints(self, ind: Individual):
        """分散性约束:每种作物每季最多5个地块"""
        for year_idx in range(len(YEARS)):
            for season_idx in range(len(SEASONS)):
                crop_lands = {}
                
                # 统计每种作物的种植地块
                for land_idx in range(len(self.problem.land_ids)):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    crop = ind.crop_genes[idx]
                    
                    if crop != 0:
                        if crop not in crop_lands:
                            crop_lands[crop] = []
                        crop_lands[crop].append((land_idx, idx))
                
                # 限制每种作物最多5个地块
                for crop, lands in crop_lands.items():
                    if len(lands) > 5:
                        # 随机移除多余的地块
                        to_remove = random.sample(lands, len(lands) - 5)
                        for land_idx, idx in to_remove:
                            ind.crop_genes[idx] = 0
                            ind.area_genes[idx] = 0.0
    
    def _yield_constraints(self, ind: Individual):
        """产量约束：产量应满足至少90%的需求"""
        # 跟踪每种作物每年的总产量
        crop_year_production = {}
        
        # 第一遍：计算当前种植方案下的产量
        for land_idx, land in enumerate(self.problem.land_ids):
            land_type = self.problem.land_types[land]
            land_area = self.problem.land_areas[land]
            
            for season_idx in range(len(SEASONS)):
                for year_idx, year in enumerate(YEARS):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    crop = ind.crop_genes[idx]
                    
                    if crop == 0:
                        continue
                    
                    # 获取产量
                    yield_per_acre = self.problem.yield_dict.get((crop, land_type), 0)
                    if yield_per_acre > 0:
                        current_yield = ind.area_genes[idx] * land_area * yield_per_acre
                        key = (year, crop)
                        if key not in crop_year_production:
                            crop_year_production[key] = 0
                        crop_year_production[key] += current_yield
        
        # 第二遍：调整种植面积以满足需求
        for land_idx, land in enumerate(self.problem.land_ids):
            land_type = self.problem.land_types[land]
            land_area = self.problem.land_areas[land]
            
            for year_idx, year in enumerate(YEARS):
                for season_idx in range(len(SEASONS)):
                    idx = ind.get_index(land_idx, season_idx, year_idx)
                    crop = ind.crop_genes[idx]
                    
                    if crop == 0:
                        continue
                    
                    try:
                        # 获取产量和需求
                        yield_per_acre = self.problem.yield_dict.get((crop, land_type), 0)
                        if yield_per_acre <= 0:
                            continue
                            
                        demand = self.problem.get_demand(crop, year)
                        if demand <= 0:
                            continue
                        
                        # 检查当前总产量与需求
                        key = (year, crop)
                        total_production = crop_year_production.get(key, 0)
                        required_production = 0.9 * demand  # 至少满足90%需求
                        
                        # 只有在总产量不足时才调整
                        if total_production < required_production:
                            # 计算需要增加的比例
                            shortfall = required_production - total_production
                            current_production = ind.area_genes[idx] * land_area * yield_per_acre
                            
                            if current_production > 0:
                                # 按比例增加面积
                                increase_ratio = min(1.0, ind.area_genes[idx] + shortfall / (land_area * yield_per_acre))
                                ind.area_genes[idx] = increase_ratio
                                
                                # 更新总产量记录
                                new_production = increase_ratio * land_area * yield_per_acre
                                crop_year_production[key] = crop_year_production.get(key, 0) + (new_production - current_production)
                    except Exception as e:
                        # 出错时不调整面积，继续处理下一个
                        continue


class FitnessEvaluator:
    """适应度评估器"""
    
    def __init__(self, problem: CropOptimizationProblem):
        self.problem = problem
    
    def evaluate(self, individual: Individual, scenario: int = 1, alpha: float = 0.5) -> float:
        """
        评估个体适应度
        scenario: 1-超出滞销, 2-50%价格出售
        alpha: 场景2中超出部分的价格折扣
        """
        # 跟踪每种作物每年的产量和需求
        crop_year_production = {}
        total_profit = 0.0
        
        # 第一遍：计算每种作物的总产量
        for land_idx, land in enumerate(self.problem.land_ids):
            land_type = self.problem.land_types[land]
            land_area = self.problem.land_areas[land]
            
            for season_idx in range(len(SEASONS)):
                for year_idx, year in enumerate(YEARS):
                    idx = individual.get_index(land_idx, season_idx, year_idx)
                    crop = individual.crop_genes[idx]
                    area_ratio = individual.area_genes[idx]
                    
                    if crop == 0 or area_ratio == 0:
                        continue
                    
                    # 获取作物参数
                    yield_per_acre = self.problem.yield_dict.get((crop, land_type), 0)
                    cost_per_acre = self.problem.cost_dict.get((crop, land_type), 0)
                    
                    # 计算种植面积、产量和成本
                    planted_area = area_ratio * land_area
                    production = planted_area * yield_per_acre
                    cost = planted_area * cost_per_acre
                    
                    # 累计成本
                    total_profit -= cost
                    
                    # 累计产量
                    key = (year, crop)
                    if key not in crop_year_production:
                        crop_year_production[key] = {
                            'production': 0,
                            'land_type': land_type  # 用于确定价格
                        }
                    crop_year_production[key]['production'] += production
        
        # 第二遍：根据总产量和需求计算收入
        for (year, crop), data in crop_year_production.items():
            try:
                production = data['production']
                land_type = data['land_type']
                
                # 获取价格
                price = 0
                for lt in set(self.problem.land_types.values()):
                    price_check = self.problem.price_dict.get((crop, lt), 0)
                    if price_check > 0:
                        price = price_check
                        break
                
                if price <= 0:
                    continue
                
                # 获取需求
                demand = self.problem.get_demand(crop, year)
                if demand <= 0:
                    # 如果没有明确需求，假设全部可销售
                    revenue = production * price
                else:
                    # 根据情景计算收入
                    if scenario == 1:
                        # 场景1：超出部分滞销
                        saleable = min(production, demand)
                        revenue = saleable * price
                    else:
                        # 场景2：超出部分50%价格
                        saleable = min(production, demand)
                        excess = max(0, production - demand)
                        revenue = saleable * price + excess * price * alpha
                
                # 累计收入
                total_profit += revenue
            except Exception as e:
                # 出错时跳过这种作物
                continue
        
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
        
        for land_idx, land in enumerate(self.problem.land_ids):
            for season_idx, season in enumerate(SEASONS):
                for year_idx, year in enumerate(YEARS):
                    idx = individual.get_index(land_idx, season_idx, year_idx)
                    
                    # 获取允许的作物
                    allowed_crops = self.problem.allowed_crops_dict[(land, season)]
                    
                    if allowed_crops and random.random() < 0.8:  # 80%概率种植
                        individual.crop_genes[idx] = random.choice(allowed_crops)
                        individual.area_genes[idx] = random.uniform(0.5, 1.0)
                    else:
                        individual.crop_genes[idx] = 0
                        individual.area_genes[idx] = 0.0
        
        return self.constraint_handler.repair(individual)
    
    def mutate(self, population: List[Individual], best: Individual) -> List[Individual]:
        """变异操作"""
        mutants = []
        n = len(population)
        
        for i in range(n):
            # 选择三个不同的个体
            candidates = [j for j in range(n) if j != i]
            r1, r2 = random.sample(candidates, 2)
            
            # 创建变异个体
            mutant = best.clone()
            
            # 连续变量的差分变异
            mutant.area_genes = (best.area_genes + 
                               MUTATION_FACTOR * (population[r1].area_genes - population[r2].area_genes))
            
            # 离散变量的变异
            mask = np.random.rand(len(mutant.crop_genes)) < MUTATION_FACTOR
            mutant.crop_genes[mask] = population[r1].crop_genes[mask]
            
            mutants.append(mutant)
        
        return mutants
    
    def crossover(self, target: Individual, mutant: Individual) -> Individual:
        """交叉操作"""
        trial = target.clone()
        
        # 随机选择交叉位置
        mask = np.random.rand(len(trial.crop_genes)) < CROSSOVER_RATE
        
        trial.crop_genes[mask] = mutant.crop_genes[mask]
        trial.area_genes[mask] = mutant.area_genes[mask]
        
        return trial
    
    def optimize(self, scenario: int = 1, max_stagnation: int = 50) -> Individual:
        """主优化循环"""
        print(f"开始优化场景 {scenario}...")
        
        # 初始化种群
        population = [self.initialize_individual() for _ in range(POPULATION_SIZE)]
        
        # 评估初始种群
        for ind in population:
            ind.fitness = self.fitness_evaluator.evaluate(ind, scenario)
        
        best_fitness_history = []
        stagnation_count = 0
        
        for generation in range(MAX_GENERATIONS):
            # 找到当前最佳个体
            best = max(population, key=lambda x: x.fitness)
            best_fitness_history.append(best.fitness)
            
            # 检查是否停滞
            if generation > 0 and best.fitness <= best_fitness_history[-2]:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            if stagnation_count >= max_stagnation:
                print(f"在第 {generation} 代停止（停滞 {max_stagnation} 代）")
                break
            
            # 变异
            mutants = self.mutate(population, best)
            
            # 交叉和选择
            new_population = []
            for target, mutant in zip(population, mutants):
                trial = self.crossover(target, mutant)
                trial = self.constraint_handler.repair(trial)
                trial.fitness = self.fitness_evaluator.evaluate(trial, scenario)
                
                # 选择更好的个体
                if trial.fitness > target.fitness:
                    new_population.append(trial)
                else:
                    new_population.append(target)
            
            population = new_population
            
            # 输出进度
            if (generation + 1) % 50 == 0:
                print(f"第 {generation + 1} 代：最佳适应度 = {best.fitness:,.2f}")
        
        final_best = max(population, key=lambda x: x.fitness)
        print(f"优化完成：最终适应度 = {final_best.fitness:,.2f}")
        
        return final_best


def save_results(individual: Individual, filename: str):
    """保存结果到Excel文件"""
    results = []
    problem = individual.problem
    
    for land_idx, land in enumerate(problem.land_ids):
        land_area = problem.land_areas[land]
        
        for season_idx, season in enumerate(SEASONS):
            for year_idx, year in enumerate(YEARS):
                idx = individual.get_index(land_idx, season_idx, year_idx)
                crop = individual.crop_genes[idx]
                area_ratio = individual.area_genes[idx]
                
                if crop != 0 and area_ratio > 0:
                    actual_area = area_ratio * land_area
                    results.append({
                        "地块名称": land,
                        "季节": season,
                        "年份": year,
                        "作物编号": crop,
                        "种植面积": round(actual_area, 3)
                    })
    
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"✓ 结果已保存至 {filename}")


def main():
    """主函数"""
    print("="*60)
    print("农作物种植策略优化 - DEGA算法")
    print("="*60)
    
    try:
        # 加载问题数据
        problem = CropOptimizationProblem()
        
        # 创建优化器
        optimizer = DEGAOptimizer(problem)
        # 问题1 - 场景1:超出部分滞销
        print("\n--- 问题1 场景1:超出部分滞销 ---")
        best1 = optimizer.optimize(scenario=1)
        save_results(best1, RESULT1_1)

        # 问题1 - 场景2:超出部分50%价格出售
        print("\n--- 问题1 场景2:超出部分50%价格出售 ---")
        best2 = optimizer.optimize(scenario=2)
        save_results(best2, RESULT1_2)
        
        print("\n" + "="*60)
        print("优化完成！")
        print(f"场景1最优利润:{best1.fitness:,.2f}")
        print(f"场景2最优利润:{best2.fitness:,.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"✗ 程序执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()