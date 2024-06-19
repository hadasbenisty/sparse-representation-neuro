"""
This file will contain analysis tools to handle the cluster_info (KiloSort) file we have for the data.
"""

import pandas as pd
from pathlib import Path

root_dir = Path("C:/Users/yoav.h/research/crsae_on_neuropixel")
data_root_dir = root_dir / "data"
experiment_root_dir = root_dir / "experiments"
code_root_dir = "/"


def load_cluster_info():
    cluster_info_file = Path(data_root_dir) / "cluster_info.csv"
    return pd.read_csv(cluster_info_file)


def filter_non_good(info_df):
    return info_df.loc[info_df.KSLabel == "good"]


if __name__ == "__main__":
    df = load_cluster_info()
    print(f"there are {len(df.cluster_id.unique())} distinct clusters")
