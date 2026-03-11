
import pandas as pd
import numpy as np
import re
import os


from pysuto.util import get_path
from pysuto.logging import get_logger



logger = get_logger(__name__)
out_folder = r'data/proc/datafeed/icsg_clean'






GLOBAL_INDICATOR_SERIES = [
    'World Mine Production',
    'World Mine Capacity',
    'Mine Capacity Utilization (%)',
    'Primary Refined Production',
    'Secondary Refined Production',
    'Refined Production (Secondary+Primary)',
    'World Refinery Capacity',
    'Refineries Capacity Utilization (%)',
    'Secondary Refined as % in Total Refined Prod.',
]


def _norm_text(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def _to_numeric(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(',', '')
    text = text.replace('%', '')
    return pd.to_numeric(text, errors='coerce')


def _indicator_aliases():
    return {
        'World Mine Production': {
            'worldmineproduction',
        },
        'World Mine Capacity': {
            'worldminecapacity',
        },
        'Mine Capacity Utilization (%)': {
            'minecapacityutilization',
            'minecapacityutilisation',
        },
        'Primary Refined Production': {
            'primaryrefinedproduction',
        },
        'Secondary Refined Production': {
            'secondaryrefinedproduction',
        },
        'Refined Production (Secondary+Primary)': {
            'refinedproductionsecondaryprimary',
            'refinedproductionprimarysecondary',
            'refinedproduction',
        },
        'World Refinery Capacity': {
            'worldrefinerycapacity',
            'worldrefiningcapacity',
        },
        'Refineries Capacity Utilization (%)': {
            'refineriescapacityutilization',
            'refinerycapacityutilization',
            'refineriescapacityutilisation',
        },
        'Secondary Refined as % in Total Refined Prod.': {
            'secondaryrefinedasintotalrefinedprod',
            'secondaryrefinedasinrefinedproduction',
            'secondaryrefinedastotalrefinedproduction',
        },
    }


def extract_icsg_global_series(
    sheet_name,
    file_years=(2005, 2014, 2023, 2025),
    file_name_map=None,
    output_name='icsg_global_indicators.csv',
):
    """
    Extract world-level ICSG indicator time series from one sheet across multiple yearbooks.

    Parameters
    ----------
    sheet_name : str | dict
        Either one sheet name used for every yearbook or a dict {year: sheet_name}.
    file_years : tuple[int]
        ICSG yearbook editions to read.
    file_name_map : dict[int, str] | None
        Optional mapping of year to raw workbook file name.
    output_name : str
        CSV file name written to ``out_folder``.
    """

    if file_name_map is None:
        file_name_map = {
            2005: 'ICSG 2005 Statisctial Yearbook.xlsx',
            2014: 'ICSG 2014 Statistical Yearbook.xlsx',
            2023: 'ICSG 2023 Statistical Yearbook.xlsx',
            2025: 'ICSG 2025 Statistical Yearbook.xlsx',
        }

    aliases = _indicator_aliases()
    collect = []

    for year in file_years:
        file_name = file_name_map[year]
        file_path = get_path(file_name)
        selected_sheet = sheet_name[year] if isinstance(sheet_name, dict) else sheet_name

        df_raw = pd.read_excel(file_path, sheet_name=selected_sheet, header=None)

        year_row_idx = None
        year_cols = {}

        for idx in range(min(25, len(df_raw))):
            row = df_raw.iloc[idx]
            tmp = {}
            for col_idx, val in row.items():
                if pd.isna(val):
                    continue
                # Convert everything to float first
                try:
                    fval = float(str(val).strip().replace(',', '').replace('%',''))
                except ValueError:
                    continue

                # Consider it a year if it's between 1900 and 2100
                if 1900 <= fval <= 2100:
                    tmp[col_idx] = int(fval)  # store as integer year

            if len(tmp) > len(year_cols):
                year_cols = tmp
                year_row_idx = idx

        if not year_cols:
            raise ValueError(f'Could not identify year columns in {file_name} / {selected_sheet}')

        data_part = df_raw.iloc[(year_row_idx + 1):, :]

        for series in GLOBAL_INDICATOR_SERIES:
            alias_set = aliases[series]
            matched_idx = None

            for ridx, row in data_part.iterrows():
                probe_cells = row.iloc[:4].dropna().astype(str)
                probe_norm = [_norm_text(cell) for cell in probe_cells]

                if any(
                    (text in alias_set)
                    or any(text.startswith(alias) for alias in alias_set)
                    or any(alias in text for alias in alias_set)
                    for text in probe_norm
                ):
                    matched_idx = ridx
                    break

            if matched_idx is None:
                logger.warning(
                    'Series "%s" not found in %s / %s',
                    series,
                    file_name,
                    selected_sheet,
                )
                continue

            row = data_part.loc[matched_idx]
            for col_idx, obs_year in year_cols.items():
                value = _to_numeric(row.iloc[col_idx])
                if pd.isna(value):
                    continue
                collect.append({
                    'Source_yearbook': year,
                    'Sheet': selected_sheet,
                    'Series': series,
                    'Year': obs_year,
                    'Value': float(value),
                })

    out = pd.DataFrame(collect)
    out = out.sort_values(['Series', 'Year', 'Source_yearbook']).reset_index(drop=True)

    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, output_name)
    out.to_csv(out_path, index=False)
    logger.info('Saved global ICSG indicator series to %s', out_path)

    return out


















# Tranform the data feed into 
if __name__ == "__main__":

    sheet_name = {
            2005: 'Table 9',
            2014: 'Sheet9',
            2023: '1.Trend',
            2025: '1.Trend',
        }
    extract_icsg_global_series(sheet_name=sheet_name)