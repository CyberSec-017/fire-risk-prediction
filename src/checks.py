import pandas as pd
from config import FIRE_PANEL, FIRES_DAILY

panel = pd.read_parquet(FIRE_PANEL)
fires = pd.read_parquet(FIRES_DAILY)
cell = ["lat_cell", "lon_cell"]
span = (panel["date"].max() - panel["date"].min()).days + 1

# 1. fire-days that never reached the panel
keys = panel[cell].drop_duplicates()
m = fires.merge(keys, on=cell, how="left", indicator=True)
in_range = fires["date"].between(panel["date"].min(), panel["date"].max())
print("panel rows:", len(panel), "| dates:", panel["date"].min().date(), "to", panel["date"].max().date())
print("fire-day rows:", len(fires), "| positives in panel:", int(panel["burned"].sum()))
print("fire-days outside the grid:", (m["_merge"] == "left_only").sum(),
      "| outside the date range:", (~in_range).sum())

# 2. one row per cell per day
print("rows per cell:", panel.groupby(cell).size().unique())
print("duplicate (cell, date) rows:", panel.duplicated(cell + ["date"]).sum())

# 3. label rates per year
print(panel.groupby(panel.date.dt.year)[["burned", "target"]].mean().round(4))

# 4. dry_days range
print(panel["dry_days"].describe())

# 5. cells with the most fire-days, as a share of all days
top = fires.groupby(cell).size().sort_values(ascending=False).head(5)
print((top / span).round(3))

# 6. year-round burners: wildfires are seasonal, industrial heat is not
f = fires[fires["date"] < "2022-01-01"].copy()      # pre-test years only
days = pd.date_range(f["date"].min(), "2021-12-31")
n_all, n_win = len(days), days.month.isin([12, 1, 2]).sum()
g = f.groupby(cell).agg(all_days=("date", "size"),
                        win_days=("date", lambda d: d.dt.month.isin([12, 1, 2]).sum()))
g["share_all"] = (g["all_days"] / n_all).round(3)
g["share_winter"] = (g["win_days"] / n_win).round(3)
print(g.sort_values("share_winter", ascending=False).head(12))