import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from io import StringIO

# Replace this with your actual CSV data

df=pd.read_csv("data/dairy_cows.csv")



nb_indicators=4
weight_animal_hazard_coverage = 1  # High weight to prioritize hazard coverage
weight_consequence_coverage = 1    # High weight to prioritize consequence coverage

beta = 10    # Weight for Indicator_Ease
gamma = 10   # Weight for Ease_of_Hazard_Mitigation
delta = 10   # Weight for Indicator_Resources



# Encode ordinal variables
ease_mapping = {'Easy': 1, 'Moderate': 2, 'Difficult': 3}
resources_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
impact_mapping = {'Low': 1, 'High': 2}

df['Indicator_Ease_Num'] = df['Indicator_Ease'].map(ease_mapping)
df['Ease_of_Hazard_Mitigation_Num'] = df['Ease_of_Hazard_Mitigation'].map(ease_mapping)
df['Indicator_Resources_Num'] = df['Indicator_Resources'].map(resources_mapping)
df['Welfare_Hazards_Impact_Num'] = df['Welfare_Hazards_Impact'].map(impact_mapping)

# Identify unique Welfare Indicators, Hazards, and Consequences
indicators_list = df['Welfare_Indicator'].unique().tolist()
hazards_animal_list = df['Welfare_Hazards_Animal'].unique().tolist()
hazards_consequence_list = df['Welfare_Hazards_Consequences'].unique().tolist()

# Create hazard-indicator association matrices
a_hi_animal = {h: {i: 0 for i in indicators_list} for h in hazards_animal_list}
for idx, row in df.iterrows():
    h = row['Welfare_Hazards_Animal']
    i = row['Welfare_Indicator']
    a_hi_animal[h][i] = 1

a_ci_consequence = {c: {i: 0 for i in indicators_list} for c in hazards_consequence_list}
for idx, row in df.iterrows():
    c = row['Welfare_Hazards_Consequences']
    i = row['Welfare_Indicator']
    a_ci_consequence[c][i] = 1

# Prepare parameter dictionaries
indicator_attributes = df.groupby('Welfare_Indicator').agg({
    'Indicator_Ease_Num': 'mean',
    'Ease_of_Hazard_Mitigation_Num': 'mean',
    'Indicator_Resources_Num': 'mean'
}).to_dict('index')

# Initialize the solver
solver = pywraplp.Solver.CreateSolver('CBC')

# Decision variables
x_vars = {}
for i in indicators_list:
    x_vars[i] = solver.IntVar(0, 1, f'Select_Indicator_{i}')

y_vars_animal = {}
for h in hazards_animal_list:
    y_vars_animal[h] = solver.IntVar(0, 1, f'Cover_Hazard_Animal_{h}')

y_vars_consequence = {}
for c in hazards_consequence_list:
    y_vars_consequence[c] = solver.IntVar(0, 1, f'Cover_Hazard_Consequence_{c}')

# Constraints
# 1. Select exactly 4 indicators
solver.Add(solver.Sum([x_vars[i] for i in indicators_list]) == nb_indicators)

# 2. Hazard coverage constraints for animal hazards
for h in hazards_animal_list:
    indicators_covering_h = [x_vars[i] for i in indicators_list if a_hi_animal[h][i] == 1]
    solver.Add(y_vars_animal[h] <= solver.Sum(indicators_covering_h))
    solver.Add(y_vars_animal[h] * len(indicators_covering_h) >= solver.Sum(indicators_covering_h))

# 3. Hazard coverage constraints for consequences
for c in hazards_consequence_list:
    indicators_covering_c = [x_vars[i] for i in indicators_list if a_ci_consequence[c][i] == 1]
    solver.Add(y_vars_consequence[c] <= solver.Sum(indicators_covering_c))
    solver.Add(y_vars_consequence[c] * len(indicators_covering_c) >= solver.Sum(indicators_covering_c))

# Objective function components
objective_terms = []

# Maximize hazard coverage
for h in hazards_animal_list:
    objective_terms.append(weight_animal_hazard_coverage * y_vars_animal[h])

for c in hazards_consequence_list:
    objective_terms.append(weight_consequence_coverage * y_vars_consequence[c])

# Minimize indicator penalties
for i in indicators_list:
    indicator_penalty = (
        beta * indicator_attributes[i]['Indicator_Ease_Num'] +
        gamma * indicator_attributes[i]['Ease_of_Hazard_Mitigation_Num'] +
        delta * indicator_attributes[i]['Indicator_Resources_Num']
    )
    objective_terms.append(-indicator_penalty * x_vars[i])

# Set the objective
solver.Maximize(solver.Sum(objective_terms))

# Solve the problem
status = solver.Solve()

# Check the solver status
if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
else:
    print('The solver did not find an optimal solution.')

# Retrieve selected indicators
selected_indicators = [i for i in indicators_list if x_vars[i].solution_value() == 1]
print("\nSelected Welfare Indicators:")
for ind in selected_indicators:
    print(f"- {ind}")

# Compute hazards and consequences covered
covered_hazards_animal = [h for h in hazards_animal_list if y_vars_animal[h].solution_value() == 1]
print("\nCovered Animal Hazards:")
for h in covered_hazards_animal:
    print(f"- {h}")
print(f"Total Number of Animal Hazards Covered: {len(covered_hazards_animal)}")

covered_hazards_consequence = [c for c in hazards_consequence_list if y_vars_consequence[c].solution_value() == 1]
print("\nCovered Consequences:")
for c in covered_hazards_consequence:
    print(f"- {c}")
print(f"Total Number of Consequences Covered: {len(covered_hazards_consequence)}")

# Objective function value
print(f"\nObjective Function Value: {solver.Objective().Value()}")

# Display selected indicators and their attributes
print("\nSelected Indicators and Their Attributes:")
for ind in selected_indicators:
    attrs = indicator_attributes[ind]
    print(f"- {ind}:")
    print(f"  Indicator_Ease_Num: {attrs['Indicator_Ease_Num']}")
    print(f"  Ease_of_Hazard_Mitigation_Num: {attrs['Ease_of_Hazard_Mitigation_Num']}")
    print(f"  Indicator_Resources_Num: {attrs['Indicator_Resources_Num']}")
