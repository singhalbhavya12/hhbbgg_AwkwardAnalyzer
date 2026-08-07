import os
import pyarrow.parquet as pq

path = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2022/sim/preEE/GGJets_MGG-80/NOTAG_merged.parquet"
# "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2024/sim/GGJets_MGG-80/nominal/NOTAG_merged.parquet"
# "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2024/sim/DDQCCDGJets/GGJets_MGG-80_Rescaled.parquet"#
# "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2022/sim/postEE/GGJets_MGG-80/NOTAG_merged.parquet"


def check_file(p):
    try:
        table = pq.read_table(p, columns=["weight"])
        print(p, table.num_rows)
    except Exception as e:
        print("ERROR", p, e)

if os.path.isfile(path):
    check_file(path)
elif os.path.isdir(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".parquet"):
                check_file(os.path.join(root, f))
else:
    print("Path does not exist:", path)