from calculate_speed import CalculateSpeed
import pandas as pd
from os.path import join, dirname, realpath

def speed_generator():
    time_stamps = [
        '0911_08t12',
        '0911_12t16',
        '0911_16t20',
        '0911_20t24',
        '0912_00t04',
        '0912_04t08',
    ]

    fns = [
        f'../data/cctv_hennessyRd_{ts}_000000_000000_carLoc.csv' for ts in time_stamps
    ]

    for ts, fn in zip(time_stamps, fns):
        ts_formatted = f"2018-{ts[:2]}-{ts[2:4]}T{ts[5:7]}:00"
        cs = CalculateSpeed(fn, start_time_stamp=ts_formatted)
        cs.get_speed()
        df = cs.get_df()

        yield df

def main():
    g_speed = speed_generator()
    dfs = []
    for df in g_speed:
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(f'../result/speeds_0911_s08_24h.csv', index=False)

    return df

if __name__ == '__main__':
    df = main()