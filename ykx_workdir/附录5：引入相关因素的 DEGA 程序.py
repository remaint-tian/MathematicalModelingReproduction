# 创建作物相关性矩阵：手动设定不同作物之间的替代性/互补性相关度
def create_correlation_matrix():
    n = len(crop_types)
    corr_matrix = np.identity(n)  # 自相关为1
    # 示例：蔬菜类之间相关性设为0.7，粮食类之间设为0.5，其它设为0.2
    for i in range(n):
        for j in range(i+1, n):
            if '蔬菜' in crop_types[i] and '蔬菜' in crop_types[j]:
                corr = 0.7
            elif '粮食' in crop_types[i] and '粮食' in crop_types[j]:
                corr = 0.5
            else:
                corr = 0.2
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    return corr_matrix

# 适应度函数：加入CVaR和相关性因素
def calculate_fitness_with_cvar_and_correlation(individual, corr_matrix, alpha=0.95):
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
                        base_price = crop_info['销售单价'].values[0]
                        base_cost = crop_info['种植成本'].values[0]
                        base_yield = crop_info['亩产量/斤'].values[0]
                        # 相关性影响：用每种作物与其他作物平均相关度来调整价格和成本
                        price_corr = 1 + corr_matrix[k].mean()
                        cost_corr = 1 + corr_matrix[k].mean()
                        if '粮食' in crop:
                            price = base_price * price_corr
                        elif '蔬菜' in crop:
                            price = base_price * (1.05 ** (t - 2023)) * price_corr
                        elif '食用菌' in crop:
                            price_decline = random.uniform(0.01, 0.05)
                            price = base_price * ((1 - price_decline) ** (t - 2023)) * price_corr
                        else:
                            price = base_price * price_corr
                        cost = base_cost * (1.05 ** (t - 2023)) * cost_corr
                        yield_per_acre = base_yield * (1 + random.uniform(-0.1, 0.1))
                        demand = demand_data[demand_data['作物编号']==int(crop.split('_')[0])]['需求量'].values[0]
                        demand = demand * (1 + random.uniform(-0.05, 0.05))
                        profit = min(yield_per_acre * area, demand) * price - cost * area
                        fitness += profit
                        if profit < 0:
                            losses.append(abs(profit))
                    # 同样的约束罚则
                    total_area = individual['x'][i, :, j-1, t-2024].sum()
                    max_area = land_data.loc[land_data['地块名称']==land, '地块面积'].values[0]
                    if total_area > max_area:
                        penalty += 1000
                    if area <= 0.5 * max_area and z_val == 1:
                        penalty += 500
                    if area > max_area * z_val:
                        penalty += 1000
    # CVaR惩罚
    if losses:
        sorted_losses = sorted(losses)
        index = int(np.ceil((1 - alpha) * len(sorted_losses)))
        cvar = np.mean(sorted_losses[:index])
        penalty += cvar
    return fitness - penalty

# 主算法流程：初始化相关性矩阵后与CVaR函数结合求解
def differential_evolution_with_correlation():
    population = initialize_population()
    corr_matrix = create_correlation_matrix()
    best_fitness_history = []
    for generation in range(NUM_GENERATIONS):
        new_population = []
        for i in range(POPULATION_SIZE):
            donor = differential_mutation(population)[i]
            trial = crossover(population[i], donor)
            ind_selected = trial if calculate_fitness_with_cvar_and_correlation(trial, corr_matrix) > calculate_fitness_with_cvar_and_correlation(population[i], corr_matrix) else population[i]
            new_population.append(ind_selected)
        population = new_population
        fitness_values = [calculate_fitness_with_cvar_and_correlation(ind, corr_matrix) for ind in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        if generation > 0 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping at generation {generation}.")
            break
        print(f"Generation {generation}: Best Fitness (with Correlation) = {best_fitness}")
    return max(population, key=lambda ind: calculate_fitness_with_cvar_and_correlation(ind, corr_matrix))

# 运行含相关性因素的算法并输出结果
best_solution_corr = differential_evolution_with_correlation()
output_data_corr = []
for i, land in enumerate(land_types):
    for k, crop in enumerate(crop_types):
        for j in seasons:
            for t in years:
                area = best_solution_corr['x'][i, k, j-1, t-2024]
                if area > 0:
                    output_data_corr.append([land, crop, j, t, area])
print(output_data_corr)



