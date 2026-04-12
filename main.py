import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import os

# Ensure output directory exists
os.makedirs('./outputs/visualizations', exist_ok=True)

def wfh_by_city():
    excel_path = "./data/WFH_TimeSeries.xlsx"
    city_mapping = {
        "Atlanta": "Atlanta-Sandy Springs-Alpharetta, GA",
        "BayArea": "San Francisco-Oakland-Berkeley, CA",
        "Chicagoland": "Chicago-Naperville-Elgin, IL-IN-WI",
        "DC": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
        "Dallas": "Dallas-Fort Worth-Arlington, TX",
        "Houston": "Houston-The Woodlands-Sugar Land, TX",
        "LosAngeles": "Los Angeles-Long Beach-Anaheim, CA",
        "Miami": "Miami-Fort Lauderdale-Pompano Beach, FL",
        "NewYork": "New York-Newark-Jersey City, NY-NJ-PA"
    }

    df = pd.read_excel(excel_path, sheet_name="WFH by city", usecols="A, E:M")
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year != 2020]

    # Clean City Column Names & Melt
    df.columns = [col.split('_')[-1] if '_' in str(col) else col for col in df.columns]
    df_long = df.melt(id_vars=['date'], var_name='City_Shorthand', value_name='WFH_Score')
    
    df_long['Metro'] = df_long['City_Shorthand'].map(city_mapping)
    return df_long.dropna(subset=["Metro"])

def home_value_index_by_zip():
    df = pd.read_csv("./data/ZHVI_ZIP.csv")
    id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 
               'State', 'City', 'Metro', 'CountyName']
    
    df_long = df.melt(id_vars=id_vars, var_name='date', value_name='ZHVI')
    df_long['date'] = pd.to_datetime(df_long['date'])
    return df_long[df_long['date'].dt.year >= 2021]

def classify_urban_suburban(df):
    core_cities_map = {
        "Atlanta-Sandy Springs-Alpharetta, GA": ["Atlanta"],
        "San Francisco-Oakland-Berkeley, CA": ["San Francisco", "Oakland", "Berkeley"],
        "Chicago-Naperville-Elgin, IL-IN-WI": ["Chicago"],
        "Washington-Arlington-Alexandria, DC-VA-MD-WV": ["Washington", "Alexandria", "Arlington"],
        "Dallas-Fort Worth-Arlington, TX": ["Dallas", "Fort Worth", "Arlington"],
        "Houston-The Woodlands-Sugar Land, TX": ["Houston"],
        "Los Angeles-Long Beach-Anaheim, CA": ["Los Angeles", "Long Beach", "Anaheim", "Santa Ana"],
        "Miami-Fort Lauderdale-Pompano Beach, FL": ["Miami", "Fort Lauderdale"],
        "New York-Newark-Jersey City, NY-NJ-PA": ["New York", "Newark", "Jersey City"]
    }

    def check_urban(row):
        urban_list = core_cities_map.get(row['Metro'], [])
        return "Urban" if row['City'] in urban_list else "Suburban"

    df['LocationType'] = df.apply(check_urban, axis=1)
    return df

def calculate_price_growth_index(merged_df):
    # Aggregate to City/Date/Type level
    wfh_scores = merged_df.groupby(['City_Shorthand', 'date'])['WFH_Score'].mean().reset_index()
    avg_zhvi = merged_df.groupby(['City_Shorthand', 'date', 'LocationType'])['ZHVI'].mean().reset_index()
    pivot_df = avg_zhvi.pivot(index=['City_Shorthand', 'date'], columns='LocationType', values='ZHVI')
    
    # Growth Index (Base 100)
    growth_index = pivot_df.groupby(level='City_Shorthand').transform(lambda x: (x / x.iloc[0]) * 100)
    
    # Log-Differences (Growth Rates)
    log_diff = np.log(pivot_df).groupby(level='City_Shorthand').diff()
    
    final_df = growth_index.reset_index()
    final_df['Suburban_Growth_Rate'] = log_diff['Suburban'].values
    final_df['Urban_Growth_Rate'] = log_diff['Urban'].values
    
    return pd.merge(final_df, wfh_scores, on=['City_Shorthand', 'date'], how='left').dropna()

def generate_wfh_impact_report(growth_df):
    wfh_stats = growth_df.groupby('City_Shorthand')['WFH_Score'].mean().reset_index()
    final_indices = growth_df[growth_df['date'] == growth_df['date'].max()].copy()
    
    final_indices['Suburban_Pct'] = final_indices['Suburban'] - 100
    final_indices['Urban_Pct'] = final_indices['Urban'] - 100
    
    report = pd.merge(final_indices, wfh_stats, on='City_Shorthand', suffixes=('', '_avg'))
    report['Donut_Gap'] = report['Suburban_Pct'] - report['Urban_Pct']
    return report.sort_values(by='Donut_Gap', ascending=False)

def visualization_wfh_vs_donut(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['WFH_Score'], df['Donut_Gap'], color='#2c3e50', s=100, zorder=3)
    
    m, b = np.polyfit(df['WFH_Score'], df['Donut_Gap'], 1)
    plt.plot(df['WFH_Score'], m*df['WFH_Score'] + b, color='#e74c3c', linestyle='--')

    for _, row in df.iterrows():
        plt.annotate(row['City_Shorthand'], (row['WFH_Score'], row['Donut_Gap']), xytext=(5, 5), textcoords='offset points')

    plt.title('WFH Intensity vs. The Donut Effect Gap')
    plt.xlabel('Average WFH Intensity')
    plt.ylabel('Donut Gap (Suburban - Urban Growth %)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./outputs/visualizations/wfh_vs_donut_correlation.png')
    plt.close()

def plot_price_trends_by_year(df):
    plt.figure(figsize=(12, 6))
    timeline = df.groupby('date')[['Suburban', 'Urban']].mean()
    plt.plot(timeline.index, timeline['Suburban'], label='Suburban Index', color='skyblue', linewidth=3)
    plt.plot(timeline.index, timeline['Urban'], label='Urban Index', color='navy', linewidth=3)
    plt.title('Home Price Index (Base 100) 2021-2025')
    plt.ylabel('Growth Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./outputs/visualizations/price_trends_raw.png')
    plt.close()

def regression_analysis(df):
    df_melted = df.melt(
        id_vars=['City_Shorthand', 'date', 'WFH_Score'],
        value_vars=['Urban_Growth_Rate', 'Suburban_Growth_Rate'],
        var_name='LocationType', value_name='Growth_Rate'
    )
    model = smf.ols(formula="Growth_Rate ~ WFH_Score * LocationType", data=df_melted).fit()
    with open("./outputs/regressions/regression_results.txt", "w") as f:
        f.write(model.summary().as_text())

def regression_robustness_check(df):
    """
    Robustness check using Time Fixed Effects and Log-Growth.
    Formula: Growth_Rate ~ WFH_Score * LocationType + C(date)
    """
    # 1. Prepare data (ensure LocationType is categorical)
    df_melted = df.melt(
        id_vars=['City_Shorthand', 'date', 'WFH_Score'],
        value_vars=['Urban_Growth_Rate', 'Suburban_Growth_Rate'],
        var_name='LocationType', 
        value_name='Growth_Rate'
    )
    
    # 2. Run the model with Time Fixed Effects [C(date)]
    # This 'soaks up' the variation caused by specific months/years
    formula = "Growth_Rate ~ WFH_Score * LocationType + C(date)"
    model = smf.ols(formula=formula, data=df_melted).fit()
    
    with open("./outputs/regressions/robustness_check.txt", "w") as f:
        f.write(model.summary().as_text())

def run_heterogeneity_analysis(df):
    """
    Splits the sample into 'Mega Cities' (Top 3) vs others
    to see if the Donut Effect is stronger in larger metros.
    """
    # 1. Define your tiers based on population/size
    mega_cities = ['NewYork', 'LosAngeles', 'Chicagoland']
    
    # 2. Prepare the long-form data
    df_melted = df.melt(
        id_vars=['City_Shorthand', 'date', 'WFH_Score'],
        value_vars=['Urban_Growth_Rate', 'Suburban_Growth_Rate'],
        var_name='LocationType', 
        value_name='Growth_Rate'
    )
    
    # 3. Create the group flag
    df_melted['Is_Mega'] = df_melted['City_Shorthand'].isin(mega_cities)
    
    # 4. Run regressions for each group
    formula = "Growth_Rate ~ WFH_Score * LocationType + C(date)"
    
    model_mega = smf.ols(formula=formula, data=df_melted[df_melted['Is_Mega'] == True]).fit()
    model_others = smf.ols(formula=formula, data=df_melted[df_melted['Is_Mega'] == False]).fit()

    # Save results
    with open("./outputs/heterogeneity/city_size.txt", "w") as f:
        f.write("MEGA CITIES RESULTS\n" + model_mega.summary().as_text() + 
                "\n\nOTHER CITIES RESULTS\n" + model_others.summary().as_text())

def run_walkability_heterogeneity(df):
    """
    Analyzes how the Donut Effect varies between walkable/transit-heavy 
    cities and car-dependent ones. SOURCE: https://www.walkscore.com/cities-and-neighborhoods/
    """
    # 1. Prepare the long-form (melted) data
    df_melted = df.melt(
        id_vars=['City_Shorthand', 'date', 'WFH_Score'],
        value_vars=['Urban_Growth_Rate', 'Suburban_Growth_Rate'],
        var_name='LocationType', 
        value_name='Growth_Rate'
    )

    # 2. Define the split
    walkable_group = ['NewYork', 'Chicagoland', 'BayArea', 'DC', 'Miami']
    df_melted['Is_Walkable'] = df_melted['City_Shorthand'].isin(walkable_group)

    # 3. Model for the Walkable/Transit Cities
    # Use C(date) for Time Fixed Effects
    model_walkable = smf.ols(
        formula="Growth_Rate ~ WFH_Score * LocationType + C(date)", 
        data=df_melted[df_melted['Is_Walkable'] == True]
    ).fit()

    # 4. Model for the Car-Dependent Cities
    model_car_centric = smf.ols(
        formula="Growth_Rate ~ WFH_Score * LocationType + C(date)", 
        data=df_melted[df_melted['Is_Walkable'] == False]
    ).fit()

    # 5. Output the results for comparison
    with open("./outputs/heterogeneity/walkable.txt", "w") as f:
        f.write("MEGA CITIES RESULTS\n" + model_walkable.summary().as_text() + 
                "\n\nOTHER CITIES RESULTS\n" + model_walkable.summary().as_text())
    
    return model_walkable, model_car_centric

def main():
    print("Starting analysis...")
    wfh_df = wfh_by_city()
    zhvi_df = home_value_index_by_zip()

    # 1. Create merge key
    wfh_df['year_month'] = wfh_df['date'].dt.to_period('M')
    zhvi_df['year_month'] = zhvi_df['date'].dt.to_period('M')

    # 2. Merge datasets
    merged_df = pd.merge(zhvi_df, wfh_df[['year_month', 'Metro', 'WFH_Score', 'City_Shorthand']], 
                         on=['year_month', 'Metro'], how='inner').dropna(subset=['WFH_Score'])
    
    # 3. Process data
    merged_df = classify_urban_suburban(merged_df)
    growth_index = calculate_price_growth_index(merged_df)
    impact_report = generate_wfh_impact_report(growth_index)

    # 4. Generate Visualizations
    visualization_wfh_vs_donut(impact_report)
    plot_price_trends_by_year(growth_index)

    # 5. Primary Regression Analysis
    print("\n--- Running Primary OLS Regression ---")
    regression_analysis(growth_index)

    # 6. Robustness Check (Time Fixed Effects)
    print("\n--- Running Robustness Check ---")
    regression_robustness_check(growth_index)

    # 7. Heterogeneity - By City Size.
    run_heterogeneity_analysis(growth_index)

    # 8. Heterogeneity - By transit score
    run_walkability_heterogeneity(growth_index)

    # 9. Write data to output folder.
    impact_report.to_csv('./outputs/FINAL_OUTPUTS.csv', index=False)
    merged_df.to_csv("./outputs/dfs/merged_df.csv", index=False)
    growth_index.to_csv("./outputs/dfs/city_growth_index_df.csv", index=False)
    wfh_df.to_csv("./outputs/dfs/wfh_data.csv", index=False)
    zhvi_df.to_csv("./outputs/dfs/zhvi_data.csv", index=False)
    
    print("\nAnalysis Complete. Check the 'outputs' folder.")
if __name__ == "__main__":
    main()