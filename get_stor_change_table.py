import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta,datetime


def get_storchange_table_for_reservoir(df_storage, df_curve, name):
    # Load your two dataframes

    # Calculate current storage
    latest_storage_row = df_storage.iloc[-1]
    current_storage = latest_storage_row["Value"]

    # Calculate 7-day change
    seven_days_ago = latest_storage_row["Timestamp"] - timedelta(days=7)
    past_storage_row = df_storage[df_storage["Timestamp"] == seven_days_ago]
    seven_day_change = current_storage - past_storage_row["Value"].iloc[0] if not past_storage_row.empty else None

    month = latest_storage_row["Timestamp"].month
    day = latest_storage_row["Timestamp"].day


    stor_curve_equivalent_date = datetime(2018 if month>=10 else 2019, month, day)

    # Calculate FIRO Storage Curve
    firo_curve_row = df_curve[df_curve["Date"] == stor_curve_equivalent_date]
    firo_storage_curve = firo_curve_row["Value"].iloc[0] if not firo_curve_row.empty else None

    # Calculate % of FIRO Storage Curve
    percent_firo_curve = (current_storage / firo_storage_curve * 100) if firo_storage_curve else None

    # Create the table
    fig = go.Figure(data=[
        go.Table(
            header=dict(values=[name, f"{month}/{day}"], align="center"),
            cells=dict(
                values=[
                    ["Current Storage (acre-feet)", "FIRO Storage Curve (acre-feet)", "% of FIRO Storage Curve", "7-day change (acre-feet)"],
                    [
                        f"{int(current_storage):,}",
                        f"{int(firo_storage_curve):,}" if firo_storage_curve else "N/A",
                        f"{int(percent_firo_curve):}%" if percent_firo_curve else "N/A",
                        f"{int(seven_day_change):,}" if seven_day_change else "N/A"
                    ]
                ],
                align="center"
            )
        )
    ])

    fig.update_layout(width=800, height=400, title=f"{name} Storage Summary")

    return fig
