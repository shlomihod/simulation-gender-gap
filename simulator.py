import numpy as np
import itertools as it
import functools as ft

import pandas as pd
from scipy.stats import binom_test
from tqdm import tqdm


N_POSITIONS = [500, 350, 200, 150, 100, 75, 40, 10]
LEAVING_PROPORTION = 0.15


def generate_employee(level, founder=False, effect_size=None,
                      p=None, mu=50, sigma=10):
    
    if effect_size is None:
        effect_size = 0

    if p is None:
        p = 0.5

    while True:
        gender = 'Male' if np.random.rand() < p else 'Female'
        shift =  effect_size * sigma if gender == 'Female' else 0
        score = np.random.normal(mu, sigma) + shift

        yield {'level': level,
               'founder': founder,
               'gender': gender,
               'score': score}

        
def initialize(n_positions, employees_generator):

    employees_by_levels = (it.islice(employees_generator(level=level,
                                                         founder=True),
                                     n_position_in_level)
                          for level, n_position_in_level
                           in enumerate(n_positions))

    
    df_by_levels = [pd.DataFrame(employees) for employees in employees_by_levels]

    employees_df = pd.concat(df_by_levels,
                             ignore_index=True)

    return employees_df


def leave(employees_df, leaving_proportion):
    employees_df = employees_df.sample(frac=1-leaving_proportion)
    return employees_df


def promote_level(employees_df, level, n_positions):
    upper_level = employees_df[employees_df['level'] == level]
    lower_level = employees_df[employees_df['level'] == level-1]

    n_to_promote = n_positions[level] - len(upper_level)

    index_to_promote = lower_level.nlargest(n_to_promote, 'score').index

    employees_df.loc[index_to_promote, 'level'] = level
    
    return employees_df


def recruit(employees_df, n_positions, employees_generator):
    first_level = employees_df[employees_df['level'] == 0]

    n_to_recruit = n_positions[0] - len(first_level)

    new_employees = it.islice(employees_generator(level=0,
                                                founder=False),
                              n_to_recruit)
    new_employees_df = pd.DataFrame(new_employees)
    employees_df = employees_df.append(new_employees_df)
    return employees_df


def simulate_timestep(employees_df, leaving_proportion, n_positions,
                      effect_size=None):
    employees_df = leave(employees_df, leaving_proportion)
    
    for level in range(len(n_positions)-1, 0, -1):
        employees_df = promote_level(employees_df, level, n_positions)
    
    employees_df = recruit(employees_df, n_positions, effect_size)

    employees_df = employees_df.sort_values(by='level')
    employees_df = employees_df.reset_index(drop=True)
    
    return employees_df


def simulate_until_no_founder(n_positions, leaving_proportion,
                              effect_size=None, p=None):
    
    employees_generator = ft.partial(generate_employee,
                                     effect_size=effect_size,
                                     p=p)

    

    employees_df = initialize(n_positions, employees_generator)
    
    with tqdm(total=float('inf')) as pbar:
        while employees_df['founder'].any():
            employees_df = simulate_timestep(employees_df,
                                             leaving_proportion,
                                             n_positions,
                                             employees_generator)
            pbar.update(1)

    return employees_df


def calc_binom_pvalue(female_proportion, positions, p):
    return binom_test(female_proportion * positions,
                      positions,
                      1-p)


def generate_one_simulation_results(n_positions,
                                    leaving_proportion,
                                    p=None,
                                    effect_size=None):
    
    employees_df = simulate_until_no_founder(n_positions,
                                             leaving_proportion,
                                             p=p,
                                             effect_size=effect_size)

    employees_by_level = employees_df.groupby('level')

    simulation_result = pd.concat({'positions': employees_by_level.size(),
                                   'mean': employees_by_level['score'].mean(),
                                   'female_proportion': (employees_by_level['gender']
                                                         .apply(lambda g: (g == 'Female').mean())),
                          }, axis=1).sort_index(ascending=False)
    
    simulation_result['pvalue'] = (simulation_result
                                   .apply(lambda r:
                                       calc_binom_pvalue(
                                           r['female_proportion'],
                                           r['positions'],
                                           p),
                                          axis=1)
                                  )

    simulation_result = simulation_result[['positions',
                                           'mean',
                                           'female_proportion',
                                           'pvalue']]


    return simulation_result


def run_simulation(effect_size=0, male_prop=0.5, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    simulation_result = generate_one_simulation_results(N_POSITIONS,
                                                        LEAVING_PROPORTION,
                                                        effect_size=effect_size,
                                                        p=male_prop)

    simulation_result = simulation_result.round(2)
    simulation_result.index.name = simulation_result.index.name.title()
    simulation_result.columns = [name.replace('_', ' ').title()
                                 for name in simulation_result.columns]

    gender_proportions = {'Female': simulation_result['Female Proportion'],
                          'Male': 1-simulation_result['Female Proportion']}
    (pd.DataFrame(gender_proportions)[::-1]
     .plot(kind='barh', stacked=True))    

    return simulation_result