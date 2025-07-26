# 适应度函数：加入CVaR（条件风险价值）惩罚项
def calculate_fitness_with_cvar(individual, alpha=0.95):
    fitness = 0
    penalty = 0
    losses = []
    for i, land in enumerate(land_types):
        for k, crop in enumerate(crop_types):
            for j in seasons:
                for t in years:
                    area = individual['x'][i, k, j-1, t-2024]
                    z_val = individual['z'][i, k, j-1, t-2024]
                    crop_info = crop_data[crop_data['组合编号']==crop]
                    if not crop_info.empty:
                        # 基准价格、成本、产量
                        base_price = crop_info['销售单价'].values[0]
                        base_cost = crop_info['种植成本'].values[0]
                        base_yield = crop_info['亩产量/斤'].values[0]
                        # 考虑作物价格和成本的年变化趋势及随机波动
                        if '粮食' in crop:
                            price = base_price  # 粮食类价格稳定
                        elif '蔬菜' in crop:
                            price = base_price * (1.05 ** (t - 2023))  # 蔬菜每年上涨5%
                        elif '食用菌' in crop:
                            price_decline = random.uniform(0.01, 0.05)
                            price = base_price * ((1 - price_decline) ** (t - 2023))
                        else:
                            price = base_price
                        cost = base_cost * (1.05 ** (t - 2023))  # 成本每年增长5%
                        yield_per_acre = base_yield * (1 + random.uniform(-0.1, 0.1))  # 产量±10%波动
                        demand = demand_data[demand_data['作物编号']==int(crop.split('_')[0])]['需求量'].values[0]
                        demand = demand * (1 + random.uniform(-0.05, 0.05))  # 需求±5%波动
                        # 计算收益
                        profit = min(yield_per_acre * area, demand) * price - cost * area
                        fitness += profit
                        # 若收益为负则记录损失，用于CVaR计算
                        if profit < 0:
                            losses.append(abs(profit))
                    # 重复附录3中的约束罚则
                    total_area = individual['x'][i, :, j-1, t-2024].sum()
                    max_area = land_data.loc[land_data['地块名称']==land, '地块面积'].values[0]
                    if total_area > max_area:
                        penalty += 1000
                    if area <= 0.5 * max_area and z_val == 1:
                        penalty += 500
                    if area > max_area * z_val:
                        penalty += 1000
    # 计算CVaR：对所有负收益取平均（损失最严重的(1-alpha)部分）
    if losses:
        sorted_losses = sorted(losses)
        index = int(np.ceil((1 - alpha) * len(sorted_losses)))
        cvar = np.mean(sorted_losses[:index])
        penalty += cvar
    return fitness - penalty

# 主算法流程：与附录3类似，只是在适应度计算时调用新函数
def differential_evolution_with_cvar():
    population = initialize_population()
    best_fitness_history = []
    for generation in range(NUM_GENERATIONS):
        new_population = []
        for i in range(POPULATION_SIZE):
            donor = differential_mutation(population)[i]
            trial = crossover(population[i], donor)
            # 选择时使用含CVaR的适应度函数
            ind_selected = trial if calculate_fitness_with_cvar(trial) > calculate_fitness_with_cvar(population[i]) else population[i]
            new_population.append(ind_selected)
        population = new_population
        fitness_values = [calculate_fitness_with_cvar(ind) for ind in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        if generation > 0 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping at generation {generation}.")
            break
        print(f"Generation {generation}: Best Fitness (with CVaR) = {best_fitness}")
    return max(population, key=lambda ind: calculate_fitness_with_cvar(ind))

# 运行含CVaR的算法并输出结果
best_solution_cvar = differential_evolution_with_cvar()
output_data_cvar = []
for i, land in enumerate(land_types):
    for k, crop in enumerate(crop_types):
        for j in seasons:
            for t in years:
                area = best_solution_cvar['x'][i, k, j-1, t-2024]
                if area > 0:
                    output_data_cvar.append([land, crop, j, t, area])
print(output_data_cvar)
