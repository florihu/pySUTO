import pandas as pd
import numpy as np
import re
import os


from pysuto.util import get_path
from pysuto.logging import get_logger



logger = get_logger(__name__)
out_folder = r'data/proc/datafeed/icsg_clean'


def clean_flows(df_init, sheet, process, flow_type, rename_dict):

    df_init = df_init.copy()

    if sheet in ['Semi_use_1', 'Semi_use_2']:
        df_init['Type'] = 'Cathode'
    elif sheet == 'SXEW':
        df_init['Type'] = 'Cathode'

    df_init.columns = df_init.columns.astype(str)
    df_init = df_init.loc[:, ~df_init.columns.str.contains('^Unnamed')]

    # -------------------------------------------------
    # find row that contains Source
    # -------------------------------------------------
    source_rows = df_init[df_init.iloc[:,0].astype(str).str.contains('Source|ICSG', na=False)]

    if not source_rows.empty:
        source_row = source_rows.index[0]
        df = df_init.iloc[:source_row].copy()
    else:
        df = df_init.copy()

    # -------------------------------------------------
    # clean column names
    # -------------------------------------------------
    cols_old = df.columns
    cols_new = cols_old.str.split('_').str[0]

    rename_dict_dynamic = dict(zip(cols_old, cols_new))
    df.rename(columns=rename_dict_dynamic, inplace=True)

    df.rename(columns=rename_dict, inplace=True)

    # -------------------------------------------------
    # remove aggregated regions
    # -------------------------------------------------
    exclude_pattern = r'\b(TOTALS?|WORLD|Other|EUROPEAN|EU|By|TOTALS)\b'

    df = df[
        ~df['Region'].astype(str).str.contains(
            exclude_pattern,
            case=False,
            na=False
        )
    ]

    #df = df[df['Region'].notna()]

    # -------------------------------------------------
    # remove unwanted flow labels
    # -------------------------------------------------
    df = df[
        ~df['Type'].astype(str).str.contains(
            r'\b(Total|Not)\b',
            case=False,
            na=False
        )
    ]
    # remove rows where Type is a stray numeric value (e.g. world totals bleed-over)
    df = df[~df['Type'].astype(str).str.match(r'^\s*[\d,\.]+\s*$')]

    # filter out total
    
   # -------------------------------------------------
    # fill region
    # -------------------------------------------------
    df['Region'] = df['Region'].replace('', np.nan)
    df.Region = df.Region.bfill()

    # -------------------------------------------------
    # flow renaming
    # -------------------------------------------------
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
        'Totall': 'Total',
    }

    df['Type'] = df['Type'].replace(rename_flow_dict)

    if process in ['Mining','Smelting','Refining']:
        df['Type'] = df['Type'].replace({'Electrowon': 'SX-EW'})

    # -------------------------------------------------
    # reshape
    # -------------------------------------------------
    df = df.melt(
        id_vars=['Region','Type'],
        var_name='Year',
        value_name='Value'
    )

    # -------------------------------------------------
    # split multiline flows
    # -------------------------------------------------
    df_split = df.copy()

    df_split['Type'] = df_split['Type'].astype(str).str.split(r'_x000d_\n|\n')
    df_split['Value'] = df_split['Value'].astype(str).str.split(r'_x000d_\n|\n')

    def pad_value_list(val_list, target_len):
        if isinstance(val_list, list):
            return val_list + [np.nan]*(target_len-len(val_list))
        return [np.nan]*target_len

    df_split['Value'] = df_split.apply(
        lambda row: pad_value_list(row['Value'], len(row['Type'])),
        axis=1
    )

    df_split = df_split.explode(['Type','Value'], ignore_index=True)

    df = df_split

    # -------------------------------------------------
    # clean values
    # -------------------------------------------------
    df['Value'] = df['Value'].replace('', np.nan)
    df['Value'] = df['Value'].str.replace(',', '', regex=False)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # -------------------------------------------------
    # clean year
    # -------------------------------------------------
    df['Year'] = df['Year'].astype(str).str.split('\n').str[0]
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # -------------------------------------------------
    # metadata
    # -------------------------------------------------
    df['Process'] = process
    df['Supply_Use'] = flow_type
    df['Unit'] = 'kt'

    # -------------------------------------------------
    # remove empty values
    # -------------------------------------------------
    df = df[df['Value'].notna() & (df['Value'] != 0)]

    # -------------------------------------------------
    # string cleaning
    # -------------------------------------------------
    df['Region'] = df['Region'].str.strip()
    df['Type'] = df['Type'].str.strip()

    rr = {
        'Philippppines': 'Philippines',
        'Belgium-': 'Belgium-Luxembourg', 
        'Taiwan (China)': 'Taiwan',
        'Taipei, China': 'Taiwan',
        'OthOtheersrs': 'Others',
    }

    df['Region'] = df['Region'].replace(rr)

    df = df.reset_index(drop=True)

    return df



def main(id=2014):

    file_name = r'ICSG 2014 Statistical Yearbook.xlsx'
    path = get_path(file_name)
    

    # Define the list as [Process, Supply/Use]
    sheet_dict = {
        
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
        'Semi_supply_1': ('Flow', 'Semi', 'Supply'),
        'Semi_supply_2': ('Flow', 'Semi', 'Supply'),
        'Sheet28': ('Stock', 'PMC', None),
        'sheet83': ('Stock', 'Exchange', None),
        'Sheet29': ('Prices', None, None),
        
    }

    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Type',
        'Feed': 'Type',
        'Semis': 'Type',
        'Semis-': 'Type',
    }

    fcol = []
    scol = []
    pcol = []

    for sheet, name in sheet_dict.items():
        dtype = name[0]
        process = name[1]
        flow_type = name[2]
        

        if sheet == 'SXEW':
            header = 3  # skip the first row for SXEW
        else:
            header = 2

        df_init = pd.read_excel(path, sheet_name=sheet, header=header)

        if dtype == 'Flow':
            flow = clean_flows(df_init, sheet, process, flow_type, rename_dict)            
            fcol.append(flow)
        elif dtype == 'Prices' :
            df_price = clean_copper_price_table(df_init, id=id)
            pcol.append(df_price)
        elif dtype == 'Stock' and process == 'PMC':
            stock = clean_pmc_stocks(df_init)
            checks_stocks(stock)
            scol.append(stock)
        elif dtype == 'Stock' and process == 'Exchange':
            stock = clean_exchange_stocks(df_init, id=id)
            checks_stocks(stock)
            scol.append(stock)
        else:
            raise ValueError(f"Unknown data type {dtype} in sheet {sheet}")


    df_flow = pd.concat(fcol, ignore_index=True)
    df_stock = pd.concat(scol, ignore_index=True) if scol else pd.DataFrame()
    df_price = pd.concat(pcol, ignore_index=True) if pcol else pd.DataFrame()

    os.makedirs(out_folder, exist_ok=True)
    flow_path = os.path.join(out_folder, 'icsg_2014_flows.csv')
    stock_path = os.path.join(out_folder, 'icsg_2014_stocks.csv')
    price_path = os.path.join(out_folder, 'icsg_2014_prices.csv')
    df_flow.to_csv(flow_path, index=False)
    df_stock.to_csv(stock_path, index=False)
    df_price.to_csv(price_path, index=False)


    pass


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
    "Price_1": ("LME Cash",    "$/metric tonne"),
    "Price_2": ("LME Cash",    "Cents/pound"),
    "Price_3": ("COMEX",       "$/metric tonne"),
    "Price_4": ("COMEX",       "Cents/pound"),
    "Price_5": ("US Producer", "Cents/pound"),
    "Price_6": ("SHFE",        "Yuan/metric tonne"),
    "Price_7": ("SHFE",        "$/metric tonne"),
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

    # SHFE (and any exchange with no HIGHS/LOWS block) defaults to Average
    df_long["Value_type"] = df_long["Value_type"].fillna("Average")
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




if __name__ == "__main__":
    main()