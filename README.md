# BeeAware

# Pitch

# BeeAware Website

A web application that helps farmers and beekeepers identify optimal flower bloom locations for bee colonies using NASA satellite data and machine learning predictions.

## Overview

BeeAware is a platform designed to support agricultural communities by leveraging climate data and machine learning to predict wildflower superbloom events in California. The website provides:

- Interactive maps
- Historical bloom data visualization
- 2026 superbloom predictions
- Monthly satellite imagery timeline
- Educational resources about bee conservation

  ## Pages

### 1. Home (`index.html`)
Landing page

### 2. Map (`map.html`)
Interactive bloom location map featuring:
- Find distance between you and Antellope Valley and/or Carrizo Plain
- Scroll animation of bloom satellite images

### 3. 2026 Predictions (`2026page.html`)
Forecast dashboard displaying:
- Spring 2026 superbloom probability
- Carrizo Plain forecast (0.674 score)
- Antelope Valley forecast (0.654 score)

### 4. Historical Data (`data.html`)
Comprehensive historical analysis:
- Year-by-year bloom performance (2020-2025)
- Quality of Bloom and contributing statistics
- Gradient Calendar

### 5. About Us (`about.html`)
Project information:
- Team background
- Mission

---

# ML Model
## California Superbloom Prediction Model

A machine learning system that predicts California wildflower superbloom probability using NASA POWER climate data and NASA Worldview Superbloom Visualizations. Trained on 5 years of historical data (2020-2025) to forecast bloom conditions for Carrizo Plain National Monument and Antelope Valley California Poppy Preserve.

---

## Project Overview

This project uses a Random Forest regression model to predict superbloom likelihood months in advance, enabling better planning for:
- Beekeeping operations and hive placement
- Regional economic forecasting
- Environmental monitoring

**Key Achievement**: 95% accuracy (R² = 0.949) in predicting bloom scores

---

## Model Performance

### Training Metrics
- **R² Score**: 0.949 (explains 94.9% of bloom variance)
- **RMSE**: 0.015 (very low prediction error)
- **Cross-Validation RMSE**: 0.031 (consistent performance)

### Feature Importance (Top 5)
1. **Soil Moisture (1-month lag)**: 40.5%
2. **Soil Moisture (3-month rolling avg)**: 19.2%
3. **Temperature (6-month rolling avg)**: 10.9%
4. **Temperature (2-month lag)**: 9.5%
5. **Solar Radiation (3-month lag)**: 6.3%

**Key Insight**: Soil moisture from the previous 1-3 months is the dominant predictor of superblooms.

---


## 2026 Forecast

### Spring 2026 Prediction: **GOOD BLOOM EXPECTED**

**Carrizo Plain National Monument**
- Forecast Score: **0.674**
- Category: Good Bloom Expected
- Superbloom Probability: 50-70% (Moderate)
- Comparison: 88% of 2023 superbloom intensity

**Antelope Valley California Poppy Preserve**
- Forecast Score: **0.654**
- Category: Good Bloom Expected
- Superbloom Probability: 50-70% (Moderate)
- Comparison: 87% of 2023 superbloom intensity

### Interpretation Scale
- **0.80+**: Exceptional superbloom (send bees immediately)
- **0.70-0.79**: Superbloom likely (send bees)
- **0.50-0.69**: Good bloom expected (worth sending bees)
- **0.30-0.49**: Moderate bloom (patchy displays)
- **<0.30**: Minimal bloom (not recommended)

### Critical Monitoring Period
**October 2025 - February 2026**

The forecast can upgrade to superbloom status if:
- October-December 2025 precipitation exceeds 2.0 mm/day average
- Soil moisture rises above 0.35 by January 2026
- Multiple well-distributed rain events occur (3+ storms)

---

## Key Findings

### Historical Patterns (2020-2025)

**Typical Bloom Years:**
- Average scores: 0.55-0.68 (good blooms but not superblooms)
- Weakest year: 2022 (0.567-0.569)

**Peak Bloom Months:**
- **March & April** consistently shows highest bloom scores at both locations

**3-Year Cycle Observed:**
- 2023: Superbloom peak
- 2024-2025: Decline phase (3-6% reduction)
- 2026: Potential recovery

### Climate Predictors

**Top 3 Most Important Factors:**
1. **Soil Moisture** (60% combined importance)
   - Previous month's moisture is #1 predictor
   - 3-month rolling average is #2 predictor

2. **Temperature Patterns** (20% combined importance)
   - 6-month rolling average matters most
   - Cool winter temps (8-15°C) are optimal

3. **Precipitation** (8% combined importance)
   - Multiple storm events better than single large storm
   - Well-distributed rainfall crucial

### Seasonal Anomalies (2025)

**Spring 2025 showed minor negative anomalies:**
- Carrizo Plain: 5-7% below historical average
- Antelope Valley: 5-6% below historical average
- Status: Normal variation, not concerning

---

## Beekeeping Applications

### Decision Matrix for Hive Deployment

**Forecast Score ≥0.80**: Send bees immediately (exceptional nectar flow)
**Forecast Score 0.70-0.79**: Send bees (superbloom likely, high ROI)
**Forecast Score 0.50-0.69**: Worth sending bees (good bloom, decent nectar)
**Forecast Score 0.30-0.49**: Marginal - evaluate alternatives
**Forecast Score <0.30**: Don't deploy

### Monthly Monitoring Schedule

**October-November 2025**: Watch soil moisture trends (target >0.25)
**December-January 2026**: Monitor precipitation (need >2.0 mm/day average)
**January 2026**: Decision point - if soil moisture >0.35, upgrade forecast
**February 2026**: Final forecast refinement before bloom
**March 2026**: Peak nectar flow window (deploy late February)

### 2026 Recommendation

**Deploy bees to both locations** - current forecast scores (0.674 and 0.654) indicate good nectar availability. Both sites bloom simultaneously in March-April, so choose based on logistics and access.

Re-run predictions monthly (October-February) to refine deployment timing.

---

## Climate Change Considerations

### Observed Trends
- 2024-2025 show declining bloom quality (down 3-6% from 2023)
- Spring 2025 exhibited minor negative anomalies
- Warm winters (>18°C) significantly reduce bloom probability

### Warning Signals
- Single large storms (increasingly common) are less effective than distributed rainfall
- Early season drying (Feb-Mar) is shortening bloom windows
- Temperature increases may disrupt optimal vernalization period

### Resilience
- 2023's superbloom proves exceptional conditions still possible
- Predictive model enables adaptive management
- Soil moisture remains adequate as of August 2025

---

## Future Enhancements

- [ ] Real-time NASA POWER API integration
- [ ] Automated monthly forecast updates
- [ ] NOAA precipitation forecast integration
- [ ] Satellite vegetation index (NDVI) tracking
- [ ] Mobile app for field observations
- [ ] Multi-year ensemble forecasting
- [ ] Additional California wildflower regions

---

# Acknowledgements, Citations, & License
This project is licensed under the MIT License.

## Background Research

Hubbart, S. (2023, March 24). The early blooms of spring: How climate change impacts growing seasons and you. The National Environmental Education Foundation (NEEF). https://www.neefusa.org/story/climate-change/early-blooms-spring-how-climate-change-impacts-growing-seasons-and-you

Juda, E. (2022a, November 30). Colony collapse, Climate Change and Public Health explained. GW.

Kerlin, K. E. (2023, April 18). Climate change is ratcheting up the pressure on bees. UC Davis.
https://www.ucdavis.edu/climate/blog/bees-face-many-challenges-and-climate-change-ratcheting-pressure

US Department of Agriculture. (n.d.). Bolstering bees in a changing climate. Tellus.
https://tellus.ars.usda.gov/stories/articles/bolstering-bees-changing-climate

U.S. Department of Agriculture. (n.d.). The Importance of Pollinators. USDA. https://www.usda.gov/about-
usda/general-information/initiatives-and-highlighted-programs/peoples-garden/importance-pollinators


## Data
The data was obtained from the POWER Project's Monthly & Annual v2.5.22 on 2025/10/04
NASA Worldview Snapshots


## Thank You!
NASA Space Apps Challenge for encouraging the creation of this project!

# Team - Bees Needs
- Nadia Chestnut
- Adam Gibson
- Robert Johnson
- Alejandra Rodriguez
- Bryson Still
- Kai Tucker


