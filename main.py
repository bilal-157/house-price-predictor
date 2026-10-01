import joblib
import pandas as pd
import numpy as np

model = joblib.load('house_price_model.pkl')
features = joblib.load('feature_names.pkl')
base_house = pd.read_csv('sample_house.csv').iloc[0].to_dict()

def predict(house):
    df = pd.DataFrame([house])[features]
    pred = model.predict(df.values)
    return np.expm1(pred[0]) if hasattr(pred, '__len__') else np.expm1(pred)

# ============================================
# Test 1: Minimum House
# ============================================
print("\n" + "="*60)
print("🏚️ TEST 1: MINIMUM HOUSE (300 sqft, 1900, Qual 1)")
print("="*60)

h = dict(base_house)
h['OverallQual'] = 1
h['OverallCond'] = 1
h['GrLivArea'] = np.log1p(300)
h['TotalBsmtSF'] = np.log1p(0)
h['1stFlrSF'] = np.log1p(300)
h['2ndFlrSF'] = np.log1p(0)
h['YearBuilt'] = 1900
h['YearRemodAdd'] = 1900
h['FullBath'] = 0
h['HalfBath'] = 0
h['BedroomAbvGr'] = 1
h['GarageCars'] = 0
h['GarageArea'] = 0
h['Fireplaces'] = 0
h['HouseAge'] = 2026 - 1900
h['RemodAge'] = 2026 - 1900
total_sf = 0 + 300 + 0
h['TotalSF'] = np.log1p(total_sf)
h['Qual_TotalSF'] = 1 * total_sf

print(f"Price: ${predict(h):,.0f}")

# ============================================
# Test 2: Maximum Mansion
# ============================================
print("\n" + "="*60)
print("🏰 TEST 2: MAXIMUM MANSION (6000 sqft, 2024, Qual 10)")
print("="*60)

h = dict(base_house)
h['OverallQual'] = 10
h['OverallCond'] = 10
h['GrLivArea'] = np.log1p(6000)
h['TotalBsmtSF'] = np.log1p(3000)
h['1stFlrSF'] = np.log1p(3000)
h['2ndFlrSF'] = np.log1p(3000)
h['YearBuilt'] = 2024
h['YearRemodAdd'] = 2024
h['FullBath'] = 5
h['HalfBath'] = 2
h['BedroomAbvGr'] = 6
h['TotRmsAbvGrd'] = 15
h['GarageCars'] = 4
h['GarageArea'] = 1200
h['Fireplaces'] = 3
h['PoolArea'] = 500
h['HouseAge'] = 2
h['RemodAge'] = 2
total_sf = 3000 + 3000 + 3000
h['TotalSF'] = np.log1p(total_sf)
h['Qual_TotalSF'] = 10 * total_sf

print(f"Price: ${predict(h):,.0f}")

# ============================================
# Test 3: Negative Values (Invalid)
# ============================================
print("\n" + "="*60)
print("❌ TEST 3: INVALID VALUES (Negative)")
print("="*60)

h = dict(base_house)
h['OverallQual'] = -5
h['GrLivArea'] = np.log1p(-500)
h['YearBuilt'] = 2050
h['FullBath'] = -2

try:
    print(f"Price: ${predict(h):,.0f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================
# Test 4: Zero Area House
# ============================================
print("\n" + "="*60)
print("🕳️ TEST 4: ZERO AREA HOUSE")
print("="*60)

h = dict(base_house)
h['GrLivArea'] = np.log1p(0)
h['TotalBsmtSF'] = np.log1p(0)
h['1stFlrSF'] = np.log1p(0)
h['2ndFlrSF'] = np.log1p(0)
h['TotalSF'] = np.log1p(0)
h['Qual_TotalSF'] = 0
h['OverallQual'] = 7

print(f"Price: ${predict(h):,.0f}")

# ============================================
# Test 5: Quality vs Price
# ============================================
print("\n" + "="*60)
print("📊 TEST 5: QUALITY vs PRICE (Same House)")
print("="*60)

h = dict(base_house)
h['GrLivArea'] = np.log1p(2000)
h['TotalSF'] = np.log1p(2000)

for qual in [1, 3, 5, 7, 9, 10]:
    h['OverallQual'] = qual
    h['Qual_TotalSF'] = qual * 2000
    print(f"Quality {qual:2d} → ${predict(h):,.0f}")

# ============================================
# Test 6: Area vs Price
# ============================================
print("\n" + "="*60)
print("📊 TEST 6: AREA vs PRICE (Same Quality)")
print("="*60)

h = dict(base_house)
h['OverallQual'] = 7

for area in [500, 1000, 1500, 2000, 3000, 4000, 5000]:
    h['GrLivArea'] = np.log1p(area)
    h['TotalSF'] = np.log1p(area)
    h['Qual_TotalSF'] = 7 * area
    print(f"Area {area:5d} sqft → ${predict(h):,.0f}")

# ============================================
# Test 7: Age vs Price
# ============================================
print("\n" + "="*60)
print("📊 TEST 7: AGE vs PRICE (Same House)")
print("="*60)

h = dict(base_house)
h['OverallQual'] = 7
h['GrLivArea'] = np.log1p(2000)
h['TotalSF'] = np.log1p(2000)
h['Qual_TotalSF'] = 7 * 2000

for year in [1900, 1950, 1970, 1990, 2000, 2010, 2020, 2024]:
    h['YearBuilt'] = year
    h['YearRemodAdd'] = year
    h['HouseAge'] = 2026 - year
    h['RemodAge'] = 2026 - year
    print(f"Year {year} → ${predict(h):,.0f}")

# ============================================
# Test 8: Random Tests
# ============================================
print("\n" + "="*60)
print("🎲 TEST 8: RANDOM TESTS")
print("="*60)

import random
random.seed(42)

for i in range(5):
    h = dict(base_house)
    h['OverallQual'] = random.randint(1, 10)
    h['GrLivArea'] = np.log1p(random.randint(500, 4000))
    h['YearBuilt'] = random.randint(1900, 2024)
    h['YearRemodAdd'] = random.randint(h['YearBuilt'], 2024)
    h['TotalSF'] = h['GrLivArea']
    h['Qual_TotalSF'] = h['OverallQual'] * np.expm1(h['GrLivArea'])
    
    area = int(np.expm1(h['GrLivArea']))
    print(f"Test {i+1}: Qual={h['OverallQual']:2d}, "
          f"Area={area:4d} sqft, Year={h['YearBuilt']} → ${predict(h):,.0f}")

print("\n" + "="*60)
print("✅ ALL TESTS COMPLETE")
print("="*60)