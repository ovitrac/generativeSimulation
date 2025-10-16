#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 22:08:22 2025

@author: olivi
"""

# %% Data generation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants for the scaling law
Ka = 140  # K
Kb = 40   # K

# Polymers with their Tg and r values
polymers = {
    'PP': {'Tg': 273, 'r': 1.0},
    'HDPE': {'Tg': 173, 'r': 1.0}
}

# Reference values
M0 = 100  # g/mol, arbitrary reference
D0 = 1e-9  # m²/s, arbitrary reference

# Generate a grid of M and T values
M_values = np.linspace(40, 500, 7)  # g/mol
T_values_C = np.linspace(23, 100, 7)  # °C
T_values = T_values_C + 273.15  # Convert to Kelvin

# Generate synthetic data
data = []

for polymer, props in polymers.items():
    Tg = props['Tg']
    r = props['r']
    for M in M_values:
        for T in T_values:
            alpha = 1 + Ka / (Kb + r * (T - Tg))
            D = D0 * (M / M0) ** (-alpha)
            data.append({
                'Polymer': polymer,
                'M': M,
                'T': T,
                'T_C': T - 273.15,
                'Tg': Tg,
                'alpha': alpha,
                'D': D
            })

df = pd.DataFrame(data)
df.head()

# %% Contrastive Data
from itertools import combinations

# Create contrastive pairs
df['logD'] = np.log(df['D'])
contrastive_data = []

for polymer in df['Polymer'].unique():
    subset = df[df['Polymer'] == polymer].reset_index(drop=True)
    for i, j in combinations(range(len(subset)), 2):
        row_i = subset.loc[i]
        row_j = subset.loc[j]

        logD_ratio = row_i['logD'] - row_j['logD']
        logM_ratio = np.log(row_i['M'] / row_j['M'])
        invT_diff = (1 / row_i['T']) - (1 / row_j['T'])
        Tg_diff = row_i['Tg'] - row_j['Tg']

        contrastive_data.append({
            'Polymer': polymer,
            'logM_ratio': logM_ratio,
            'invT_diff': invT_diff,
            'Tg_diff': Tg_diff,
            'logD_ratio': logD_ratio
        })

contrastive_df = pd.DataFrame(contrastive_data)

# %%
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



# ----- Preprocessing -----
X = contrastive_df[['logM_ratio', 'invT_diff', 'Tg_diff']].values
y = contrastive_df['logD_ratio'].values.reshape(-1, 1)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ----- Neural Network -----
class ContrastiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ContrastiveNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# ----- Training Loop -----
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

# ----- Evaluation -----
model.eval()
with torch.no_grad():
    pred = model(X_test).numpy()
    y_test_rescaled = y_scaler.inverse_transform(y_test)
    pred_rescaled = y_scaler.inverse_transform(pred)
    r2 = r2_score(y_test_rescaled, pred_rescaled)

print(f"Contrastive NN R² score: {r2:.4f}")


# %% symbolic regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Split the contrastive_df into training and test sets
train_df, test_df = train_test_split(contrastive_df, test_size=0.2, random_state=0)

# Training data
X_train = train_df[['logM_ratio', 'invT_diff', 'Tg_diff']].values
y_train = train_df['logD_ratio'].values

# Test data
X_test = test_df[['logM_ratio', 'invT_diff', 'Tg_diff']].values
y_test = test_df['logD_ratio'].values

# Symbolic regression surrogate: polynomial regression
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
y_train_pred = poly_model.predict(X_train)
y_test_pred = poly_model.predict(X_test)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Training set
sns.scatterplot(x=y_train, y=y_train_pred, ax=axs[0])
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--')
axs[0].set_title("Train set: Symbolic (poly) regression")
axs[0].set_xlabel("True logD_ratio")
axs[0].set_ylabel("Predicted logD_ratio")

# Test set
sns.scatterplot(x=y_test, y=y_test_pred, ax=axs[1])
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axs[1].set_title("Test set: Symbolic (poly) regression")
axs[1].set_xlabel("True logD_ratio")
axs[1].set_ylabel("Predicted logD_ratio")

plt.tight_layout()
plt.show()

# Return test and training R² scores
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_train, r2_test

# %% NN
# Re-run neural network using torch and compute residual statistics
import torch
import torch.nn as nn

# Standardize input and output
X_full = contrastive_df[['logM_ratio', 'invT_diff', 'Tg_diff']].values
y_full = contrastive_df['logD_ratio'].values.reshape(-1, 1)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X_full)
y_scaled = y_scaler.fit_transform(y_full)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Define model
class ContrastiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ContrastiveNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Train the model
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# Predict and inverse transform
model.eval()
with torch.no_grad():
    train_pred_scaled = model(X_train_tensor).numpy()
    val_pred_scaled = model(X_val_tensor).numpy()

y_train_true = y_scaler.inverse_transform(y_train)
y_val_true = y_scaler.inverse_transform(y_val)
train_pred = y_scaler.inverse_transform(train_pred_scaled)
val_pred = y_scaler.inverse_transform(val_pred_scaled)

# Compute residuals
train_residuals = y_train_true - train_pred
val_residuals = y_val_true - val_pred

# Compute residual statistics
train_stats = {
    'mean': np.mean(train_residuals),
    'std': np.std(train_residuals),
    'min': np.min(train_residuals),
    'max': np.max(train_residuals),
    'rmse': np.sqrt(np.mean(train_residuals**2))
}

val_stats = {
    'mean': np.mean(val_residuals),
    'std': np.std(val_residuals),
    'min': np.min(val_residuals),
    'max': np.max(val_residuals),
    'rmse': np.sqrt(np.mean(val_residuals**2))
}

train_stats, val_stats

# %%
import numpy as np
from sklearn.metrics import mean_squared_error

# After training and evaluation
model.eval()
with torch.no_grad():
    train_pred_scaled = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
    val_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

# Inverse transform back to logD_ratio
y_train_true = y_scaler.inverse_transform(y_train)
#y_val_true = y_scaler.inverse_transform(y_test)
y_val_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))

train_pred = y_scaler.inverse_transform(train_pred_scaled)
val_pred = y_scaler.inverse_transform(val_pred_scaled)

# Residuals
train_residuals = y_train_true - train_pred
val_residuals = y_val_true - val_pred

# Statistics
def residual_stats(residuals):
    return {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'rmse': float(np.sqrt(np.mean(residuals**2)))
    }

train_stats = residual_stats(train_residuals)
val_stats = residual_stats(val_residuals)

print("🔍 NN Residuals on Training Set:", train_stats)
print("🔍 NN Residuals on Validation Set:", val_stats)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Predicted vs Actual
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_train_true, train_pred, alpha=0.6, edgecolor='k')
ax[0].plot([y_train_true.min(), y_train_true.max()],
           [y_train_true.min(), y_train_true.max()], 'r--')
ax[0].set_title("Training Set: Predicted vs Actual")
ax[0].set_xlabel("True log(Di/Dj)")
ax[0].set_ylabel("Predicted log(Di/Dj)")

ax[1].scatter(y_val_true, val_pred, alpha=0.6, edgecolor='k')
ax[1].plot([y_val_true.min(), y_val_true.max()],
           [y_val_true.min(), y_val_true.max()], 'r--')
ax[1].set_title("Validation Set: Predicted vs Actual")
ax[1].set_xlabel("True log(Di/Dj)")
ax[1].set_ylabel("Predicted log(Di/Dj)")
plt.tight_layout()
plt.show()

# 2. Residuals Histogram
plt.figure(figsize=(10, 4))
sns.histplot(train_residuals.flatten(), bins=20, kde=True, color='blue', label='Train', stat='density')
sns.histplot(val_residuals.flatten(), bins=20, kde=True, color='orange', label='Val', stat='density')
plt.axvline(0, color='k', linestyle='--')
plt.title("Histogram of Residuals")
plt.xlabel("Residual (True - Predicted)")
plt.legend()
plt.show()

# 3. Residuals vs Predicted
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(train_pred, train_residuals, alpha=0.6, edgecolor='k')
ax[0].axhline(0, color='r', linestyle='--')
ax[0].set_title("Training Set: Residuals vs Predicted")
ax[0].set_xlabel("Predicted log(Di/Dj)")
ax[0].set_ylabel("Residual")

ax[1].scatter(val_pred, val_residuals, alpha=0.6, edgecolor='k')
ax[1].axhline(0, color='r', linestyle='--')
ax[1].set_title("Validation Set: Residuals vs Predicted")
ax[1].set_xlabel("Predicted log(Di/Dj)")
ax[1].set_ylabel("Residual")
plt.tight_layout()
plt.show()

# %% Visual diagnostics
# Predicted vs Actual
#   - For both NN and symbolic regression
# Residuals
#  - Residuals vs predicted
#  - Histograms
#  - QQ plots (normality)
#  - Shapiro-Wilk test for normality
# Comparison Summary
#  - Side-by-side performance of both models
#  - Symbolic is interpretable, NN is slightly better at generalization

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import statsmodels.api as sm

sns.set(style="whitegrid")

# ----- 1. Predicted vs Actual -----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_train_true, train_pred, alpha=0.6, edgecolor='k')
ax[0].plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'r--')
ax[0].set_title("Training Set: Predicted vs Actual")
ax[0].set_xlabel("True log(Di/Dj)")
ax[0].set_ylabel("Predicted log(Di/Dj)")

ax[1].scatter(y_val_true, val_pred, alpha=0.6, edgecolor='k')
ax[1].plot([y_val_true.min(), y_val_true.max()], [y_val_true.min(), y_val_true.max()], 'r--')
ax[1].set_title("Validation Set: Predicted vs Actual")
ax[1].set_xlabel("True log(Di/Dj)")
ax[1].set_ylabel("Predicted log(Di/Dj)")
plt.tight_layout()
plt.show()

# ----- 2. Histogram of Residuals -----
plt.figure(figsize=(10, 4))
sns.histplot(train_residuals.flatten(), bins=20, kde=True, color='blue', label='Train', stat='density')
sns.histplot(val_residuals.flatten(), bins=20, kde=True, color='orange', label='Val', stat='density')
plt.axvline(0, color='k', linestyle='--')
plt.title("Histogram of Residuals")
plt.xlabel("Residual (True - Predicted)")
plt.legend()
plt.show()

# ----- 3. Residuals vs Predicted -----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(train_pred, train_residuals, alpha=0.6, edgecolor='k')
ax[0].axhline(0, color='r', linestyle='--')
ax[0].set_title("Training Set: Residuals vs Predicted")
ax[0].set_xlabel("Predicted log(Di/Dj)")
ax[0].set_ylabel("Residual")

ax[1].scatter(val_pred, val_residuals, alpha=0.6, edgecolor='k')
ax[1].axhline(0, color='r', linestyle='--')
ax[1].set_title("Validation Set: Residuals vs Predicted")
ax[1].set_xlabel("Predicted log(Di/Dj)")
ax[1].set_ylabel("Residual")
plt.tight_layout()
plt.show()

# ----- 4. QQ-Plot -----
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
sm.qqplot(train_residuals.flatten(), line='s', ax=ax1)
ax1.set_title('QQ-Plot of Training Residuals')

ax2 = fig.add_subplot(122)
sm.qqplot(val_residuals.flatten(), line='s', ax=ax2)
ax2.set_title('QQ-Plot of Validation Residuals')
plt.tight_layout()
plt.show()

# ----- 5. Shapiro-Wilk Test -----
shapiro_train = shapiro(train_residuals.flatten())
shapiro_val = shapiro(val_residuals.flatten())

print("🧪 Shapiro-Wilk Test (Training Residuals):", {"W": shapiro_train.statistic, "p-value": shapiro_train.pvalue})
print("🧪 Shapiro-Wilk Test (Validation Residuals):", {"W": shapiro_val.statistic, "p-value": shapiro_val.pvalue})

# ----- 6. Compare with Symbolic Regression -----
# Run symbolic regression again to get residuals on same splits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X_train_sym = X_train
X_val_sym = X_test
y_train_sym = y_train_true
y_val_sym = y_val_true

poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_model.fit(X_train_sym, y_train_sym)
train_pred_sym = poly_model.predict(X_train_sym)
val_pred_sym = poly_model.predict(X_val_sym)

train_residuals_sym = y_train_sym - train_pred_sym
val_residuals_sym = y_val_sym - val_pred_sym

def compare_models_stats(res1, res2, label1="NN", label2="PolyReg"):
    def stats(res):
        return {
            'mean': float(np.mean(res)),
            'std': float(np.std(res)),
            'rmse': float(np.sqrt(np.mean(res**2))),
            'min': float(np.min(res)),
            'max': float(np.max(res))
        }
    return {
        label1 + "_train": stats(train_residuals),
        label1 + "_val": stats(val_residuals),
        label2 + "_train": stats(train_residuals_sym),
        label2 + "_val": stats(val_residuals_sym)
    }

comparison = compare_models_stats(train_residuals, train_residuals_sym)
import pprint; pprint.pprint(comparison, sort_dicts=False)

# %% NN vs Polynomial approximation

# Visual comparisons between Neural Network (NN) and Symbolic (Polynomial) Regression
# Comparison Plots:
#   - Predicted vs True (y_true) — side-by-side plots for both models.
#   - Residuals vs y_true — to check bias and structure.
#   - Residual Histograms — to compare spread and distribution.

# 1. Predicted vs True
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(y_val_true, val_pred, label="NN", alpha=0.6, edgecolor='k')
axs[0].plot([y_val_true.min(), y_val_true.max()],
            [y_val_true.min(), y_val_true.max()], 'r--')
axs[0].set_title("Neural Network: Predicted vs True")
axs[0].set_xlabel("True log(Di/Dj)")
axs[0].set_ylabel("Predicted log(Di/Dj)")

axs[1].scatter(y_val_true, val_pred_sym, label="Symbolic", alpha=0.6, edgecolor='k')
axs[1].plot([y_val_true.min(), y_val_true.max()],
            [y_val_true.min(), y_val_true.max()], 'r--')
axs[1].set_title("Symbolic Regression: Predicted vs True")
axs[1].set_xlabel("True log(Di/Dj)")
axs[1].set_ylabel("Predicted log(Di/Dj)")

plt.tight_layout()
plt.show()

# 2. Residuals vs True
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(y_val_true, val_residuals, label="NN", alpha=0.6, edgecolor='k')
axs[0].axhline(0, color='r', linestyle='--')
axs[0].set_title("NN: Residuals vs True log(Di/Dj)")
axs[0].set_xlabel("True log(Di/Dj)")
axs[0].set_ylabel("Residual")

axs[1].scatter(y_val_true, val_residuals_sym, label="Symbolic", alpha=0.6, edgecolor='k')
axs[1].axhline(0, color='r', linestyle='--')
axs[1].set_title("Symbolic: Residuals vs True log(Di/Dj)")
axs[1].set_xlabel("True log(Di/Dj)")
axs[1].set_ylabel("Residual")

plt.tight_layout()
plt.show()

# 3. Residual Histogram Comparison
plt.figure(figsize=(10, 4))
sns.histplot(val_residuals.flatten(), bins=20, kde=True, color='skyblue', label='NN', stat='density')
sns.histplot(val_residuals_sym.flatten(), bins=20, kde=True, color='orange', label='Symbolic', stat='density')
plt.axvline(0, color='k', linestyle='--')
plt.title("Validation Residuals: NN vs Symbolic")
plt.xlabel("Residual (True - Predicted)")
plt.legend()
plt.tight_layout()
plt.show()
