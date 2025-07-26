#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dega_crop_optimization_full.py
差分进化-遗传算法 (DEGA) 求解 2024-2030 农作物种植策略 (场景 1 & 2)。

Author: ChatGPT (o3)
"""

import os, random, numpy as np, pandas as pd

# ==================== 文件路径 ====================
DATA_DIR    = "D:\2024_C"
LAND_FILE   = os.path.join(DATA_DIR, "附件1.xlsx")
CROP_FILE   = os.path.join(DATA_DIR, "附件2.xlsx")
DEMAND_FILE = os.path.join(DATA_DIR, "2023年各季次农作物实际产量.xlsx")
RESULT1     = os.path.join(DATA_DIR, "result1_1.xlsx")
RESULT2     = os.path.join(DATA_DIR, "result1_2.xlsx")

# ==================== 全局常量 ====================
YEARS   = list(range(2024, 2031))     # 7 年
SEASONS = [1, 2]                      # 每年两季

# ----- 作物编号 -----
GRAIN_IDS             = list(range(1, 16))        # 1-15 粮食
LEGUME_GRAIN_IDS      = list(range(1, 6))         # 1-5  粮豆
RICE_ID               = 16                        # 16  水稻
VEGETABLE_IDS         = list(range(17, 35))       # 17-34 蔬菜
LEGUME_VEGETABLE_IDS  = [17, 18, 19]              # 17-19 蔬菜豆类
WATER_VEGETABLE_IDS   = [35, 36, 37]              # 35-37 水二季蔬
MUSHROOM_IDS          = [38, 39, 40, 41]          # 38-41 食用菌

# ----- 地块编号 -----
FLAT_DRY_LAND_IDS       = [f"A{i}" for i in range(1, 7)]
TERRACE_LAND_IDS        = [f"B{i}" for i in range(1, 15)]
HILL_LAND_IDS           = [f"C{i}" for i in range(1, 7)]
IRRIGATED_LAND_IDS      = [f"D{i}" for i in range(1, 9)]
ORDINARY_GH_IDS         = [f"E{i}" for i in range(1, 17)]
SMART_GH_IDS            = [f"F{i}" for i in range(1, 5)]

# ==================== DE 参数 ====================
POP_SIZE = 120
MAX_GEN  = 600
F_SCALE  = 0.5
CR_RATE  = 0.9

# =================================================
# 0. allowed_crops —— 按地块+季节给出可选作物集合
# =================================================
def allowed_crops(land: str, season: int):
    # A/B/C：仅允许 1-15（粮食）；后续约束会限制“每年仅一季”
    if land in FLAT_DRY_LAND_IDS + TERRACE_LAND_IDS + HILL_LAND_IDS:
        return GRAIN_IDS
    # D（水浇地）
    if land in IRRIGATED_LAND_IDS:
        if season == 1:
            return [RICE_ID] + VEGETABLE_IDS        # 第一季：水稻 或 蔬菜
        else:
            return VEGETABLE_IDS + WATER_VEGETABLE_IDS  # 第二季：蔬菜 or 水二季蔬
    # E（普通大棚）
    if land in ORDINARY_GH_IDS:
        return VEGETABLE_IDS if season == 1 else MUSHROOM_IDS
    # F（智慧大棚）
    if land in SMART_GH_IDS:
        # 两季均蔬菜，但禁止 35-37
        return [v for v in VEGETABLE_IDS if v not in WATER_VEGETABLE_IDS]
    return []

# =================================================
# 1. Problem 数据
# =================================================
class ProblemData:
    def __init__(self):
        self.land_df   = pd.read_excel(LAND_FILE,   sheet_name=0)
        self.crop_df   = pd.read_excel(CROP_FILE,   sheet_name=1)
        self.demand_df = pd.read_excel(DEMAND_FILE, sheet_name=0)

        self.land_ids   = self.land_df["地块名称"].tolist()
        self.land_type  = dict(zip(self.land_df["地块名称"], self.land_df["地块类型"]))
        self.land_area  = dict(zip(self.land_df["地块名称"], self.land_df["面积"]))
        self.crop_ids   = self.crop_df["作物编号"].tolist()

        self.yield_map  = {(r["作物编号"], r["地块类型"]): r["亩产量"]
                           for _,r in self.crop_df.iterrows()}
        self.cost_map   = {(r["作物编号"], r["地块类型"]): r["种植成本"]
                           for _,r in self.crop_df.iterrows()}
        self.price_map  = {(r["作物编号"], r["地块类型"]): r["销售单价"]
                           for _,r in self.crop_df.iterrows()}

# =================================================
# 2. 染色体表示
# =================================================
class Individual:
    def __init__(self, prob: ProblemData):
        self.prob  = prob
        self.size  = len(prob.land_ids)*len(SEASONS)*len(YEARS)
        self.crops = np.zeros(self.size, dtype=int)     # 离散基因
        self.props = np.zeros(self.size, dtype=float)   # 面积比例 [0,1]
        self.fitness = None
    def clone(self):
        c = Individual(self.prob)
        c.crops  = self.crops.copy()
        c.props  = self.props.copy()
        c.fitness= self.fitness
        return c

# =================================================
# 3. repair —— 12 条约束全覆盖
# =================================================
def repair(ind: Individual):
    prob = ind.prob
    idx = lambda l,s,y: (prob.land_ids.index(l)*len(SEASONS)+ (s-1))*len(YEARS)+ (y-2024)

    # ---------- 1) 基本合法性 ----------
    for l in prob.land_ids:
        for s in SEASONS:
            for y in YEARS:
                k  = idx(l,s,y)
                if ind.crops[k] not in allowed_crops(l,s):
                    ind.crops[k] = 0
                ind.props[k] = np.clip(ind.props[k],0,1)
                if ind.crops[k]==0:
                    ind.props[k]=0.0
                elif ind.props[k] < 0.5:               # 下限 0.5A
                    ind.props[k]=0.5

    # ---------- 7) A/B/C 每年仅一季 ----------
    ABC = FLAT_DRY_LAND_IDS + TERRACE_LAND_IDS + HILL_LAND_IDS
    for l in ABC:
        for y in YEARS:
            k1,k2 = idx(l,1,y), idx(l,2,y)
            # 若两季都有作物，随机保留一季
            if ind.crops[k1] and ind.crops[k2]:
                if random.random()<0.5:
                    ind.crops[k2]=0; ind.props[k2]=0.0
                else:
                    ind.crops[k1]=0; ind.props[k1]=0.0

    # ---------- 8) D 地块复杂规则 ----------
    for l in IRRIGATED_LAND_IDS:
        for y in YEARS:
            k1,k2 = idx(l,1,y), idx(l,2,y)
            # 8-1: 第一季若为水稻，则第二季必须休耕
            if ind.crops[k1]==RICE_ID:
                ind.crops[k2]=0; ind.props[k2]=0.0
            # 8-2: 第二季水二季蔬 35-37 允许，其余地块不得 35-37
            if ind.crops[k2] in WATER_VEGETABLE_IDS and ind.crops[k1]==RICE_ID:
                # 若第一季水稻禁止 35-37，改为普通蔬菜
                ind.crops[k2] = random.choice(VEGETABLE_IDS)
            # 若第一季空且第二季为水二季蔬 OK；否则若第二季种 35-37 则强制 0.5 面积
            if ind.crops[k2] in WATER_VEGETABLE_IDS:
                ind.props[k2]=max(ind.props[k2],0.5)

    # ---------- 9) E 地块 ----------
    for l in ORDINARY_GH_IDS:
        for y in YEARS:
            k1,k2 = idx(l,1,y), idx(l,2,y)
            # 第一季必须蔬菜 17-34
            if ind.crops[k1] not in VEGETABLE_IDS:
                ind.crops[k1] = random.choice(VEGETABLE_IDS); ind.props[k1]=0.5
            # 第二季必须食用菌 38-41
            ind.crops[k2] = random.choice(MUSHROOM_IDS); ind.props[k2]=0.5

    # ---------- 10) F 地块 ----------
    for l in SMART_GH_IDS:
        for y in YEARS:
            for s in SEASONS:
                k=idx(l,s,y)
                if ind.crops[k] in WATER_VEGETABLE_IDS or ind.crops[k]==RICE_ID:
                    ind.crops[k]=random.choice([c for c in VEGETABLE_IDS if c not in WATER_VEGETABLE_IDS])
                    ind.props[k]=0.5

    # ---------- 11) 食用菌仅 E-S2 ----------
    for l in prob.land_ids:
        if l in ORDINARY_GH_IDS: continue
        for y in YEARS:
            for s in SEASONS:
                k=idx(l,s,y)
                if ind.crops[k] in MUSHROOM_IDS:
                    ind.crops[k]=0; ind.props[k]=0.0

    # ---------- 2/3/4/5/6 其余轮作、豆类、分散度、产量下限 ----------
    # 2) 同季跨年不得重茬
    for l in prob.land_ids:
        for s in SEASONS:
            for y1,y2 in zip(YEARS[:-1],YEARS[1:]):
                k1,k2=idx(l,s,y1),idx(l,s,y2)
                if ind.crops[k1]==ind.crops[k2]!=0:
                    ind.crops[k2]=0; ind.props[k2]=0.0
    # 3) 跨季跨年
    for l in prob.land_ids:
        for y in YEARS[:-1]:
            k2=idx(l,2,y); k1=idx(l,1,y+1)
            if ind.crops[k2]==ind.crops[k1]!=0:
                ind.crops[k1]=0; ind.props[k1]=0.0
    # 4) 三年窗豆类
    beans=set(LEGUME_GRAIN_IDS+LEGUME_VEGETABLE_IDS)
    for l in prob.land_ids:
        for st in range(len(YEARS)-2):
            win=YEARS[st:st+3]
            if not any(ind.crops[idx(l,s,y)] in beans for y in win for s in SEASONS):
                # 在窗口首年第一季强制种豆
                k=idx(l,1,win[0])
                choices=list(beans & set(allowed_crops(l,1)))
                ind.crops[k]=random.choice(choices); ind.props[k]=0.5
    # 5) 分散度 ≤5
    for y in YEARS:
        for s in SEASONS:
            bucket={}
            for l in prob.land_ids:
                k=idx(l,s,y); c=ind.crops[k]
                if c: bucket.setdefault(c,[]).append(k)
            for c,ks in bucket.items():
                for k in random.sample(ks,max(0,len(ks)-5)):
                    ind.crops[k]=0; ind.props[k]=0.0
    # 6) 产量 ≥0.9需求
    for l in prob.land_ids:
        for s in SEASONS:
            for y in YEARS:
                k=idx(l,s,y); c=ind.crops[k]
                if c==0: continue
                A=prob.land_area[l]; yld=prob.yield_map.get((c,prob.land_type[l]),0)
                demand=prob.demand_df.query("year==@y and crop==@c")["demand"]
                d=float(demand) if not demand.empty else 0
                prod=ind.props[k]*A*yld
                if prod<0.9*d and yld>0:
                    need=(0.9*d)/(A*yld); ind.props[k]=min(1.0,max(ind.props[k],need))
    return ind

# =================================================
# 4. 适应度
# =================================================
def evaluate(ind: Individual, scenario=1, alpha=0.5):
    prob=ind.prob; total=0.0
    idx=lambda l,s,y:(prob.land_ids.index(l)*len(SEASONS)+ (s-1))*len(YEARS)+ (y-2024)
    for l in prob.land_ids:
        lt=prob.land_type[l]; A=prob.land_area[l]
        for s in SEASONS:
            for y in YEARS:
                k=idx(l,s,y); c=ind.crops[k]; p=ind.props[k]
                if c==0 or p==0: continue
                yld=prob.yield_map.get((c,lt),0); cost=prob.cost_map.get((c,lt),0); price=prob.price_map.get((c,lt),0)
                prod=A*p*yld
                demand=prob.demand_df.query("year==@y and crop==@c")["demand"]
                d=float(demand) if not demand.empty else 0
                rev=min(prod,d)*price if scenario==1 else (min(prod,d)+alpha*max(0,prod-d))*price
                total+=rev-cost*A*p
    return total

# =================================================
# 5. DEGA 算子
# =================================================
def mutate(pop,best,F):
    new=[]
    n=len(pop)
    for i in range(n):
        r1,r2=random.sample([j for j in range(n) if j!=i],2)
        m=best.clone()
        m.props = best.props + F*(pop[r1].props-pop[r2].props)
        mask=np.random.rand(m.crops.size)<F
        m.crops[mask]=pop[r1].crops[mask]
        new.append(m)
    return new

def crossover(tgt,mut,CR):
    tri=tgt.clone()
    mask=np.random.rand(tri.props.size)<CR
    tri.props[mask]=mut.props[mask]; tri.crops[mask]=mut.crops[mask]
    return tri

# =================================================
# 6. 初始化 & 演化
# =================================================
def random_individual(prob):
    ind=Individual(prob)
    for l in prob.land_ids:
        for s in SEASONS:
            for y in YEARS:
                idx=(prob.land_ids.index(l)*len(SEASONS)+(s-1))*len(YEARS)+(y-2024)
                choices=allowed_crops(l,s)
                if choices and random.random()<0.8:       # 80% 概率种植
                    ind.crops[idx]=random.choice(choices)
                    ind.props[idx]=random.uniform(0.5,1)
    return ind

def evolve(prob,scenario):
    pop=[repair(random_individual(prob)) for _ in range(POP_SIZE)]
    for ind in pop: ind.fitness=evaluate(ind,scenario)
    for g in range(MAX_GEN):
        best=max(pop,key=lambda x:x.fitness)
        muts=mutate(pop,best,F_SCALE)
        new=[]
        for tgt,mut in zip(pop,muts):
            tri=repair(crossover(tgt,mut,CR_RATE))
            tri.fitness=evaluate(tri,scenario)
            new.append(tri if tri.fitness>tgt.fitness else tgt)
        pop=new
        if (g+1)%50==0:
            print(f"Gen {g+1}:  best = {best.fitness:,.1f}")
    return max(pop,key=lambda x:x.fitness)

# =================================================
# 7. 导出
# =================================================
def write_plan(ind,path):
    rows=[]; prob=ind.prob
    for l in prob.land_ids:
        A=prob.land_area[l]
        for s in SEASONS:
            for y in YEARS:
                idx=(prob.land_ids.index(l)*len(SEASONS)+(s-1))*len(YEARS)+(y-2024)
                c,p=ind.crops[idx],ind.props[idx]
                if c and p>0:
                    rows.append({"land":l,"season":s,"year":y,"crop":c,"area":round(p*A,3)})
    pd.DataFrame(rows).to_excel(path,index=False)
    print("✔ 结果写入",path)

# ==================== 主程序 ====================
if __name__=="__main__":
    prob=ProblemData()
    best1=evolve(prob,scenario=1); write_plan(best1,RESULT1)
    best2=evolve(prob,scenario=2); write_plan(best2,RESULT2)
