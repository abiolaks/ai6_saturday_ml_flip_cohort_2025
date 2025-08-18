# Real Estate EDA: Plot Explanation and Insights

## 📊 Overview
This document summarizes the key findings and visual insights from the exploratory data analysis (EDA) of the real estate dataset. The analysis leverages various plots to uncover trends, relationships, and data quality issues that are crucial for stakeholders and data-driven decision-making.

---

## 🏠 Key Insights

### Price Distribution by Property Type
- **Single-family homes** tend to have a wider price range and a higher median price compared to apartments and other property types.

### Feature Relationships
- **Transaction Price vs. Square Footage (`tx_price` vs. `sqft`):**
  - Shows a clear positive linear relationship—larger properties tend to sell for higher prices.
- **Transaction Price vs. Property Tax (`tx_price` vs. `property_tax`):**
  - Reveals a strong positive correlation, as property taxes are often based on assessed value.
- **Bubble Plot (Lot Size):**
  - Properties with larger lot sizes (`lot_size`) also tend to have higher prices, even for similar square footage, confirming lot size as an important price driver.

---

## 📝 Summary of Insights
- **Price is strongly driven by property size:**
  - There is a clear positive correlation between `tx_price`, `sqft`, and `property_tax`.
- **Single-family homes are the most valuable property type:**
  - The price distribution for single-family homes is higher and wider than for other property types.
- **Lot size also plays a significant role:**
  - Properties with larger lots tend to have higher prices, independent of their square footage.
- **Property Type is a Major Factor:**
  - Single-family homes consistently have a wider price range and a higher median transaction price compared to apartments, condos, and townhouses.
- **Size Matters:**
  - Larger homes generally sell for higher prices.
- **Lot Size is a Key Differentiator:**
  - Especially for single-family homes, larger lots are associated with higher transaction prices, even for similar square footage.

---

## 🗺️ Public Records and Location
- **Property Tax Reflects Value:**
  - The monthly property tax (`property_tax`) has a very strong positive correlation with the transaction price, confirming its direct relationship to assessed value.
- **Location Conveniences:**
  - The number of restaurants, grocery stores, nightlife, and other amenities within one mile are potential drivers of property value. Descriptive statistics and visualizations hint at their influence.

---

## 🧹 Data Quality and Preparation
- **Missing Data:**
  - Missing values in the `basement` column likely indicate the absence of a basement and were filled with 0.
  - Missing values for `exterior_walls` and `roof` were filled with the most common values.
- **Outliers:**
  - Outliers in numerical features like `tx_price` and `sqft` were observed through box plots. These may represent luxury properties and should be addressed during modeling for accuracy and robustness.

### Outlier Features (from Box Plot)
- `insurance`
- `sqft`
- `property_tax`
- `lot_size`


### Skewness (from Histogram/KDE Plot)
- Most numerical features are skewed to the right, suggesting the need for transformation.

---

## 🏗️ Impact of Property Age on Transaction Price

To effectively analyze the impact of a property's age on its transaction price, the continuous `age_of_property` variable was grouped into meaningful categories:

- **Relatively New:** Less than 10 years old
- **Not Too Old:** 10 to 30 years old
- **Old House:** More than 30 years old

The box plot below visualizes the distribution of transaction prices for each of these categories.

### Insights from the Binned Data

- **Relatively New Houses Command the Highest Price:**
  - The 'Relatively New' category has the highest median transaction price, indicating that newer homes, on average, sell for more. The price distribution for these properties is also higher than the other two categories.
- **Older Houses Have a Lower Median Price:**
  - The 'Old House' category has the lowest median transaction price. This suggests that as a property ages, its value tends to decrease.
- **Wider Price Spread in 'Old Houses':**
  - While the median price is lower, the 'Old House' category shows a wider interquartile range (the box part of the plot) and more potential outliers, indicating a greater variability in price for older properties. This could be due to factors like renovations, historical value, or specific architectural features.

## Trend Analysis: Year Built and Transaction Price

### Key Patterns

- **Pre-1930:** Almost no observations, which explains why the “average price by year built” is very noisy for the oldest years.
- **1940s–1970s:** Steady rise in counts as building activity ramps up; occasional spikes and dips from small samples.
- **Mid-1980s–early 2000s:** Peak volume (40–60+ per year)—lots of homes from these vintages, so price estimates are more reliable.
- **Sharp drop after ~2005–2015:** Likely a data cutoff or right-censoring (few very recent builds in the dataset) and/or a real post-2008 slowdown.

### Big Takeaways

- **Pre-1950s:** Very volatile. Large spikes (e.g., ~1890, ~1910, mid-1940s) and dips occur because there are likely very few sales per build-year. With tiny sample sizes, one luxury or distressed sale can swing the average a lot.
- **1955–1985:** Stable plateau. Averages hover roughly in the \$350k–\$450k band with modest wiggles—suggesting more consistent stock (lots of mid-century homes).
- **1985–2010:** Gradual rise. A slow upward trend toward \$450k–\$500k.
- **~2015–2016:** A sharp spike near the end. That \$700k point is almost certainly an outlier year (few observations or a cluster of high-end

**Conclusion:**
This analysis shows that the age of a property is a significant factor in determining its value, with newer properties generally having a higher price.

---

## 🔎 Additional Observations
1. **6 numeric features show ≥5% outliers.** Consider trimming/winsorization or robust scalers. Notable: cafes, nightlife, lot_size, shopping, restaurants, etc.
2. **Top numeric correlations:**
   - `tx_price`–`basement`: nan
   - `beds`–`sqft`: 0.69
   - `beds`–`basement`: nan
3. **Key drivers of `tx_price`:**
   - `insurance`, `property_tax`, `lot_size`, `sqft`, `married`
4. **Median `tx_price` is up ~19.3% over the last 3 months vs prior 3.**

---

## ⚠️ Next Steps
- Address missing values (NaNs) and outliers before modeling.
- Consider feature engineering and transformation for skewed variables.
- Further analyze the impact of location-based features and amenities.

---

