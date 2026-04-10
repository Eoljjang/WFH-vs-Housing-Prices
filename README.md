### Usage
1. Make sure you have git's large fle size extension ("lfs") installed. Installation varies between Windows & MacOS.
2. In addition to `git pull origin main` make sure you `git lfs pull`. 
3. Run `python3 main.py`.
4. The data analysis will run and outputs will be saved to the `outputs` folder.

**Output File Descriptions:**
1. `wfh_data.csv` = The WFH score of various cities from 2021 - 2025.
2. `zhvi_data.csv` = ZHVI index of housing across the U.S
3. `merged_df.csv` = Filters (2) for the cities in (1). Categorizes each zip code by "Urban" or "Suburban".
4. `city_growth_index.csv` = Calculates the change in housing prices for suburban vs. urban parts of a city, along with its WFH score.
    - Standardized to base 100.
    - Resets annually to base 100.
5. `FINAL_REPORT.csc` / `.xlsx` = Final combined data demonstrating the net change in urban & suburban pricing, and donut gap.
6. `regression.txt` = Regression metrics.

**Note:** If all you need is the final data, (e) and (f) are the most relevant.

