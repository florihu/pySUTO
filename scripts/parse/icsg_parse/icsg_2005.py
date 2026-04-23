import pandas as pd
import numpy as np
import re
import os


from pysuto.util import get_path
from pysuto.logging import get_logger

logger = get_logger(__name__)


out_folder = r'data/proc/datafeed/icsg_clean'


REGION_RENAME = {
        # typos
        'Philippppines':    'Philippines',
        'OthOtheersrs':     'Others',
        'kyrgyzstan':       'Kyrgyzstan',
        # footnote suffixes
        'Germany 1/':       'Germany',
        'Zambia 2/':        'Zambia',
        'Zambia 3/':        'Zambia',
        'Congo, D.R.  1/':  'Congo, D.R.',
        'Chile 1/':         'Chile',
        'China 3/ 9/':      'China',
        'United States 1/': 'United States',
        'Others 6/':        'Others',
        'Benelux 3/':        'Benelu',
        # name consolidation
        'U.S.':             'United States',
        'Russian Fed.':     'Russia',
        'Czech Rep.':       'Czech Republic',
        'Dominican Rep.':   'Dominican Republic',
        'Korean Rep.':      'South Korea',
        'Korean rep.':      'South Korea',
        'Korean':           'South Korea',
        # Belgium variants
        'Belgium-':             'Belgium-Luxembourg',
        'Belgium- Luxembourg':  'Belgium-Luxembourg',
        # Taiwan
        'Taiwan (China)':   'Taiwan',
        'Taipei, China':    'Taiwan',
        # Congo
        'Congo DR':         'Congo, D.R.',
        'Congo Rep.':       'Congo',
        # Serbia
        'Serbia & Montenegro': 'Serbia',
        # Macedonia
        'Macedonia':        'North Macedonia',
        # ambiguous truncations -> drop
        'South':            None,
        'United':           None,
     'Philippppines': 'Philippines', 
     'Belgium-': 'Belgium-Luxembourg'
    }




def icsg_2005(id=2005):

    file_name = "ICSG 2005 Statisctial Yearbook.xlsx"
    path = get_path(file_name)
    

    # Define the list as [Data_type, Process, Supply/Use]
    sheet_dict = {
        'Semi_supply_1': ('Flow', 'Semi', 'Supply'),
        'Semi_supply_2': ('Flow', 'Semi', 'Supply'),
        'Table 25': ('Stock', 'PMC', None),
        'sheet80': ('Stock', 'Exchange', None),
        'Table 26': ('Price', None, None),
        'Mining_1': ('Flow', 'Mining', 'Supply'),
        'Mining_2': ('Flow', 'Mining', 'Supply'),
        'Smelting_1': ('Flow', 'Smelting', 'Supply'),
        'Smelting_2': ('Flow', 'Smelting', 'Supply'),
        'Refining_1': ('Flow', 'Refining', 'Supply'),
        'Refining_2': ('Flow', 'Refining', 'Supply'),
        'Refining_3': ('Flow', 'Refining', 'Supply'),
        'SXEW': ('Flow', 'SXEW', 'Supply'),
        'Semi_use_1': ('Flow', 'Semi', 'Use'),
        'Semi_use_2': ('Flow', 'Semi', 'Use'),
        
        }


    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Type',
        'Feed': 'Type',
        'Semis': 'Type',
        'Semis-': 'Type',
        'Source_x000d_\n1/': 'Type',
        'Feed Type': 'Type',
        'Feed Source  1/': 'Type',
        'Semis- Type': 'Type',
        'Semis Tye': 'Type',

        
    }

    flow_collect = []
    stock_collect = []
    price_collect = []

    for sheet, name in sheet_dict.items():

        logger.debug(f"Processing sheet: {sheet} with data type: {name[0]}, process: {name[1]}, flow type: {name[2]}")
        
        data_type = name[0]
        process = name[1]
        flow_type = name[2]


        if sheet == 'SXEW':
            header = 1  # skip the first row for SXEW
        else:
            header = 0

        df_init = pd.read_excel(path, sheet_name=sheet, header=header)
       
        if data_type =='Flow':
            flow = clean_flows(df_init, sheet, process, flow_type, rename_dict)            
            flow_collect.append(flow)


        elif data_type == 'Price' :
            df_price = clean_copper_price_table(df_init, id=id)
            price_collect.append(df_price)
       
        elif data_type == 'Stock'and process == 'PMC':

            stock = clean_pmc_stocks(df_init)
            checks_stocks(stock)
            stock_collect .append(stock)

        elif data_type == 'Stock' and process == 'Exchange':
            stock = clean_exchange_stocks(df_init, id=id)
            checks_stocks(stock)
            stock_collect.append(stock)
        else:
            raise ValueError(f"Unknown data type {data_type} in sheet {sheet}")
    

    df_flow = pd.concat(flow_collect, ignore_index=True)
    df_stock = pd.concat(stock_collect, ignore_index=True) if stock_collect else pd.DataFrame()
    df_price = pd.concat(price_collect, ignore_index=True) if price_collect else pd.DataFrame()

    os.makedirs(out_folder, exist_ok=True)
    flow_path = os.path.join(out_folder, 'icsg_2005_flows.csv')
    stock_path = os.path.join(out_folder, 'icsg_2005_stocks.csv')
    price_path = os.path.join(out_folder, 'icsg_2005_prices.csv')
    df_flow.to_csv(flow_path, index=False)
    df_stock.to_csv(stock_path, index=False)
    df_price.to_csv(price_path, index=False)

    return df_flow, df_stock, df_price

def check_flows(df):

    # per Region, Type Year Process is only one value
    duplicates = df[df.duplicated(subset=['Region', 'Type', 'Year', 'Process'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Duplicate entries found for the same Region, Type, Year, Process combination:\n{duplicates}")
    # there are NaN anywhere
    if df.isnull().any().any():
        logger.warning("NaN values found in the DataFrame")

    # Year col is int and there are no negative values in Value column
    if not pd.api.types.is_integer_dtype(df['Year']):
        logger.warning("Year column is not of integer type")
    if (df['Value'] < 0).any():
        logger.warning("Negative values found in Value column")


    pass

def checks_stocks(df):
    #1 there is per Type Region Stock_type Year a unique value
    duplicates = df[df.duplicated(subset=['Type', 'Region', 'Stock_type', 'Year'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Duplicate entries found for the same Type, Region, Stock_type, Year combination:\n{duplicates}")
        pass

    # there are NaN anywhere
    if df.isnull().any().any():
        logger.warning("NaN values found in the DataFrame")
    # Year col is int and there are no negative values in Value column
    if not pd.api.types.is_integer_dtype(df['Year']):
        logger.warning("Year column is not of integer type")

    if (df['Value'] < 0).any():
        logger.warning("Negative values found in Value column")
    
    pass

def clean_exchange_stocks(df_raw, id, start =3):

    df = df_raw.copy()

    col = df.iloc[1, :].to_list()
    df.columns = col

    df = df.iloc[start:, :].reset_index(drop=True)

  

    # --------------------------------------------------
    # 1️⃣ Set correct header (row 1)
    # --------------------------------------------------
    #df.columns = df.iloc[1]
    

    df = df.rename(columns={df.columns[0]: "Region"})

    # --------------------------------------------------
    # 2️⃣ Identify exchange blocks
    # --------------------------------------------------

    current_exchange = "LME"   # default (everything before COMEX)
    exchange_list = []

    for val in df["Region"]:

        val_str = str(val).strip().upper()

        if val_str == "COMEX":
            current_exchange = "COMEX"
            exchange_list.append(current_exchange)
            continue

        if val_str == "SHFE":
            current_exchange = "SHFE"
            exchange_list.append(current_exchange)
            continue

        exchange_list.append(current_exchange)

    df["Exchange_type"] = exchange_list

    # --------------------------------------------------
    # 3️⃣ Remove header and total rows
    # --------------------------------------------------
    df = df[
        ~df["Region"].str.contains(
            r"LONDON METAL EXCHANGE|TOTAL|Total",
            case=False,
            na=False
        )
    ]

    # --------------------------------------------------
    # 4️⃣ Identify year columns
    # --------------------------------------------------
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c))]

    # --------------------------------------------------
    # 5️⃣ Melt to long format
    # --------------------------------------------------
    df_long = df.melt(
        id_vars=["Region", "Exchange_type"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )

    # --------------------------------------------------
    # 6️⃣ Clean numeric values
    # --------------------------------------------------
    df_long["Value"] = (
        df_long["Value"]
        .astype(str)
        .str.replace(",", ".", regex=False)   # comma → decimal
    )

    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    df_long = df_long.dropna(subset=["Value"])

    # --------------------------------------------------
    # 7️⃣ Assign correct Region for COMEX / SHFE
    # --------------------------------------------------
    df_long["Region"] = df_long["Region"].str.strip()

    df_long.loc[df_long["Exchange_type"] == "COMEX", "Region"] = "United States"
    df_long.loc[df_long["Exchange_type"] == "SHFE", "Region"] = "China"

    # --------------------------------------------------
    # 8️⃣ Build final clean table
    # --------------------------------------------------
    df_clean = pd.DataFrame({
        "Stock_type": "exchanges",
        "Exchange_type": df_long["Exchange_type"],
        "Region": df_long["Region"],
        "Year": pd.to_numeric(df_long["Year"], errors="coerce"),
        "Value": df_long["Value"],
        "Unit": "kt"
    })

    df_clean = df_clean.reset_index(drop=True)

    # stock_type upper letter
    df_clean["Stock_type"] = df_clean["Stock_type"].str.title()
    #rename exchange type col in type
    df_clean = df_clean.rename(columns={"Exchange_type": "Type"})

    # filter NaN and 0
    df_clean = df_clean[df_clean['Value'].notna() & (df_clean['Value'] != 0)].reset_index(drop=True)
    return df_clean

def clean_copper_price_table(df_raw, id):

    df = df_raw.copy()

    # -------------------------------------------------
    # 1. Force proper column names
    # -------------------------------------------------

    # First column = year/period column
    df = df.rename(columns={df.columns[0]: "Year_raw"})

    # Rename remaining columns generically
    value_cols = []
    for i, col in enumerate(df.columns[1:], start=1):
        new_name = f"Price_{i}"
        df = df.rename(columns={col: new_name})
        value_cols.append(new_name)

    # Ensure Year_raw is a Series
    if isinstance(df["Year_raw"], pd.DataFrame):
        df["Year_raw"] = df["Year_raw"].iloc[:, 0]

    df["Year_raw"] = df["Year_raw"].astype(str).str.strip()

    # -------------------------------------------------
    # 2. Identify Annual blocks
    # -------------------------------------------------

    df["Type"] = np.where(
        df["Year_raw"].str.contains("ANNUAL", case=False, na=False),
        df["Year_raw"],
        np.nan
    )

    df["Type"] = df["Type"].ffill()

    # -------------------------------------------------
    # 3. Extract numeric years
    # -------------------------------------------------

    df["Year"] = pd.to_numeric(df["Year_raw"], errors="coerce")

    df = df[df["Year"].notna()]
    df["Year"] = df["Year"].astype(int)

    # -------------------------------------------------
    # 4. Melt to long
    # -------------------------------------------------

    df_long = df.melt(
        id_vars=["Year", "Type"],
        value_vars=value_cols,
        var_name="Column",
        value_name="Value"
    )

    # -------------------------------------------------
    # 5. Clean numeric values
    # -------------------------------------------------

    df_long["Value"] = (
        df_long["Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

    df_long = df_long[df_long["Value"].notna()]

    # -------------------------------------------------
    # 6. Map columns to exchanges
    # -------------------------------------------------

    exchange_map = {
        "Price_1": ("LME Cash", "$/metric tonne"),
        "Price_2": ("LME Cash", "Cents/pound"),
        "Price_3": ("COMEX", "$/metric tonne"),
        "Price_4": ("COMEX", "Cents/pound"),
        "Price_5": ("US Producer", "Cents/pound")
    }

    df_long["Exchange_type"] = df_long["Column"].map(
        lambda x: exchange_map.get(x, ("Unknown", None))[0]
    )

    df_long["Unit"] = df_long["Column"].map(
        lambda x: exchange_map.get(x, ("Unknown", None))[1]
    )

    # -------------------------------------------------
    # 7. Clean value type
    # -------------------------------------------------

    df_long["Value_type"] = (
        df_long["Type"]
        .str.replace("ANNUAL ", "", regex=False)
        .str.replace("AVERAGES", "AVERAGE", regex=False)
        .str.title()
    )

    # -------------------------------------------------
    # 8. Final output
    # -------------------------------------------------

    df_long = df_long[
        ["Year", "Value", "Exchange_type", "Unit", "Value_type"]
    ]

    df_long = df_long.sort_values(
        ["Year", "Exchange_type", "Value_type"]
    ).reset_index(drop=True)

    return df_long

def clean_pmc_stocks(df_raw):
    df = df_raw.copy()

    # --- 1. Promote first row as header and clean ---
    header = (
        df.iloc[0]
        .astype(str)
        .str.replace(r'_x000d_', ' ', regex=True)
        .str.replace('\n', ' ', regex=False)
        .str.strip()
    )
    df.columns = header

    # identify last row with 'TOTAL COUNTRY STOCKS'
    total_row = df[df[df.columns[0]].str.contains('TOTAL COUNTRY STOCKS', na=False)].index

    # if found, keep only rows before this row
    if not total_row.empty:
        df = df.iloc[1:total_row[0], :].reset_index(drop=True)
    
    # --- 2. Keep only rows with meaningful first column ---
    df = df[df[df.columns[0]].notna()]

    # --- 3. Identify year columns ---
    year_cols = [
        c for c in df.columns
        if str(c).replace('.', '', 1).isdigit()
        and 1900 <= int(float(c)) <= 2100
    ]

    # --- 4. Melt to long format ---
    df_long = df.melt(
        id_vars=[df.columns[0]],
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    df_long = df_long[df_long['Value'].notna()]

    # --- 5. Explode multi-value rows ---
    def explode_row(row):
        # Split multiple entries in first column
        parts = str(row[df.columns[0]]).split('_x000d_\n')
        # Split multiple values in Value column
        values = str(row['Value']).replace(',', '').split('_x000d_\n')
        # If numbers don't match parts, pad with NaN
        if len(values) < len(parts):
            values += [np.nan] * (len(parts) - len(values))
        elif len(values) > len(parts):
            # Sometimes the last part is "total", ignore it
            values = values[:len(parts)]
        # Create exploded rows
        exploded = []
        for p, v in zip(parts, values):
            exploded.append({'STOCK_TYPE_AND_COUNTRY': p.strip(), 'Year': row['Year'], 'Value': float(v)})
        return exploded

    exploded_rows = []
    for _, r in df_long.iterrows():
        exploded_rows.extend(explode_row(r))

    df_exp = pd.DataFrame(exploded_rows)

    # --- 6. Parse Region and Stock_type ---
    def parse_stock_region(x):
        # Skip totals or summary
        if re.search(r'total|summary|sources', x, re.IGNORECASE):
            return pd.Series([np.nan, np.nan])
        match = re.match(r'(.*?)[ ]*\((.*?)\)', x)
        if match:
            region = match.group(1).strip()
            stype = match.group(2).strip().lower()
            return pd.Series([region, stype])
        else:
            return pd.Series([x.strip(), np.nan])

    df_exp[['Region','Stock_type']] = df_exp['STOCK_TYPE_AND_COUNTRY'].apply(parse_stock_region)

    # --- 7. Drop rows with NaN Region or Value ---
    df_exp = df_exp.dropna(subset=['Region','Value'])

    # --- 8. Map known exchanges ---
    exchange_map = {'COMEX':'United States','SHFE':'China','LONDON METAL EXCHANGE':'Multiple','TOTAL EXCHANGES':'Multiple'}
    df_exp['Region'] = df_exp['Region'].replace(exchange_map)

    # --- 9. Fill unknown Stock_type if possible ---
    df_exp['Stock_type'] = df_exp['Stock_type'].fillna('unknown')

    # --- 10. Add Unit ---
    df_exp['Unit'] = 'kt'

    # --- 11. Keep only final columns ---
    df_clean = df_exp[['Stock_type','Region','Year','Value','Unit']]
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Year']).reset_index(drop=True)
    

    # replace unknown with total and make the first letter in Stock_type large
    df_clean['Stock_type'] = df_clean['Stock_type'].replace('unknown', 'Total')
    df_clean['Stock_type'] = df_clean['Stock_type'].str.title()
    df_clean['Region'] = df_clean['Region'].replace({'Others  2/': 'Others'}) 

    # rename stock stype to type
    df_clean = df_clean.rename(columns={'Stock_type': 'Type'})
    df_clean['Stock_type'] = 'PMC'

    # clean Type col
    tr ={'P(Prorodduucceersrs':'Producers'}

    rr = {'UU.SS.':'U.S.'}

    df_clean['Type'] = df_clean['Type'].replace(tr)
    df_clean['Region'] = df_clean['Region'].replace(rr)

    df_clean = df_clean[df_clean['Value'].notna() & (df_clean['Value'] != 0)].reset_index(drop=True)


    #make year an int col
    df_clean['Year'] = df_clean['Year'].astype(int)

    # if the value for U.S. producers is 1212.99 then change it to 12.9
    df_clean.loc[(df_clean['Region'] == 'U.S.') & (df_clean['Type'] == 'Producers') & (df_clean['Value'] == 1212.99), 'Value'] = 12.9

    return df_clean

def clean_flows(df_init, sheet, process, flow_type, rename_dict):
    """
    Parse and clean a raw ICSG flows sheet into tidy long-format DataFrame.

    Key rule:
    Drop 'Total' rows ONLY when another Type with a non-zero value exists
    for the same (Region, Year). Otherwise keep the Total.
    """

    import numpy as np
    import pandas as pd

    df = df_init.copy()

    # ── 1. Sheet-level type override ─────────────────────────────
    if sheet in ['Semi_use_1', 'Semi_use_2', 'SXEW']:
        df['Semis- Type'] = 'Cathode'

    # ── 2. Drop unnamed columns ──────────────────────────────────
    df.columns = df.columns.astype(str)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    # ── 3. Truncate at Source / ICSG footer row ─────────────────
    source_row_idx = df[df.iloc[:, 0].astype(str).str.contains('Source|ICSG', na=False)].index
    if len(source_row_idx):
        df = df.iloc[:source_row_idx[0], :].copy()

    
    # ── 5. Rename columns according to rename_dict ──────────────
    df.rename(columns=rename_dict, inplace=True)

    # check if a Type and Region column exist after renaming
    if 'Type' not in df.columns or 'Region' not in df.columns:
        logger.warning(f"After renaming columns in sheet {sheet}, 'Type' or 'Region' column is missing. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing 'Type' or 'Region' column after renaming in sheet {sheet}")

    # ── 6. Drop aggregated / regional totals ────────────────────
    AGG_PATTERN = r'\b(TOTALS?|WORLD|Other|OthOtheersrs|EUROPEAN|EU|By)\b'
    df = df[~df['Region'].astype(str).str.contains(AGG_PATTERN, case=False, na=False)]
    df = df[df['Region'].notna()]

    # ── 7. Forward-fill region names ────────────────────────────
    df['Region'] = df['Region'].bfill()
    df['Region'] = df['Region'].str.replace(r'(\d+/)+\s*', '', regex=True)

   
    rename_flow_dict = {
        'SSecondary': 'Secondary',
        'Primaryy': 'Primary',
        'Primary Secondary': 'Primary\nSecondary',
        'Electrowon Primary': 'Electrowon\nPrimary',
        'Electrowon Primary Secondary': 'Electrowon\nPrimary\nSecondary',
        'Electrowon\nPrimary Secondary': 'Electrowon\nPrimary\nSecondary',
        'Electrowon Primary\nSecondary': 'Electrowon\nPrimary\nSecondary',
        'Electrowon_x000d_\nPrimary Secondary': 'Electrowon\nPrimary\nSecondary',
        'Electrowon Primary_x000d_\nSecondary': 'Electrowon\nPrimary\nSecondary',
        'Copper_x000d_\nCu Alloy': 'Copper\nCu Alloy',
        'Copper_x000d_\nNot Specif.': 'Copper\nNot Specified',
        'Concentrates_x000d_\nSX-EW': 'Concentrates\nSX-EW',
        'Primary_x000d_': 'Primary',
        'Low-grade_x000d_': 'Low-grade',
        'Copper_x000d_\nCu Alloy': 'Copper\nCu Alloy',
        'Copper Cu Alloy_x000d_\nNot Specif.': 'Copper\nCu Alloy\nNot Specified',
        'Primary_x000d_\nSecondary': 'Primary\nSecondary',
        'Low-grade_x000d_\nElectrowon': 'Low-grade\nElectrowon',
        'Electrowon_x000d_\nPrimary': 'Electrowon\nPrimary',




    }
   

    df['Type'] = df['Type'].replace(rename_flow_dict)

    # ── 8. Explode multi-line Type/Value correctly ─────────────
    # Split multi-line Type first
    df['Type'] = df['Type'].astype(str).str.split(r'\n')

    value_cols = df.columns.difference(['Region', 'Type'])

    exploded_rows = []

    for _, row in df.iterrows():
        types = row['Type']  # list of types, e.g., ['Copper', 'Cu Alloy']
        if not isinstance(types, list):
            types = [types]  # wrap single element

        n_types = len(types)

        # Extract values per column and split by newline
        values_lists = []
        for col in value_cols:
            val = str(row[col])
            val_list = [v.strip() for v in val.replace('_x000d_\n', '\n').split('\n')]
            # Pad with NaN if fewer values than types
            val_list += [np.nan] * (n_types - len(val_list))
            values_lists.append(val_list)

        # Create one row per type
        for i, t in enumerate(types):
            new_row = {'Region': row['Region'], 'Type': t}
            for j, col in enumerate(value_cols):
                new_row[col] = values_lists[j][i]
            exploded_rows.append(new_row)

    # Convert back to DataFrame
    df_exploded = pd.DataFrame(exploded_rows)
    
    df = df_exploded.copy()


    # ── 9. Melt wide to long ───────────────────────────────────
    df = df.melt(id_vars=['Region', 'Type'], var_name='Year', value_name='Value')

    # ── 10. Clean numeric values ───────────────────────────────
    df['Value'] = df['Value'].replace('', np.nan)
    df['Value'] = df['Value'].str.replace(',', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # ── 11. Clean year labels ──────────────────────────────────
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    if process in ['Mining', 'Smelting', 'Refining']:
        df['Type'] = df['Type'].replace({'Electrowon': 'SX-EW'})

    # ── 13. Attach metadata ───────────────────────────────────
    df['Process'] = process
    df['Supply_Use'] = flow_type
    df['Unit'] = 'kt'

    # ── 14. Drop NaN/zero values ──────────────────────────────
    df = df[df['Value'].notna() & (df['Value'] != 0)]

    # ── 15. Trim whitespace ───────────────────────────────────
    df['Region'] = df['Region'].str.strip()
    df['Type'] = df['Type'].astype(str).str.strip()

    # ── 16. Fix known region typos ────────────────────────────
    
    df['Region'] = df['Region'].replace(REGION_RENAME)

    # ── 17. Drop redundant Total rows last ────────────────────
    total_mask = df['Type'].str.contains(r'\bTotal\b', case=False, na=False)
    has_specific_value = (~total_mask & df['Value'].notna() & (df['Value'] != 0)).groupby([df['Region'], df['Year']]).transform('any')
    df = df[~(total_mask & has_specific_value)]

    TYPE_RENAME = {'Not Specified': 'Not Specif.'}
    df['Type'] = df['Type'].replace(TYPE_RENAME)

    
    unique_types_after = df['Type'].unique()
    logger.debug(f"Unique types after cleaning in sheet {sheet}: {unique_types_after}")
    unique_regions_after = df['Region'].unique()
    logger.debug(f"Unique regions after cleaning in sheet {sheet}: {unique_regions_after}")
    df = df.reset_index(drop=True)
    return df



if __name__ == "__main__":
    icsg_2005()