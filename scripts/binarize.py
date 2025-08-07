# build_csv.py
import pathlib, csv
root = pathlib.Path("Dataset_BUSI_with_GT")
rows=[]
for cls,label in [("malignant",1),("benign",0),("normal",0)]:
    for p in (root/cls).glob("*.png"):
        if "_mask" in p.name:  # skip mask files
            continue
        rows.append((str(p), label))
with open("busi_labels.csv","w",newline='') as f:
    csv.writer(f).writerows(rows)
print(f"Wrote {len(rows)} rows")      # 780