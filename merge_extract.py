import pandas as pd
from pathlib import Path

MOUSE_CSV = Path("output/mouse_features.csv")
ACCESS_CSV = Path("output/log_features.csv")
OUT_CSV = Path("combine_dataset.csv")
REPORT_DIR = Path("output/merge_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["session_id"] = df["session_id"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates(subset=["session_id"], keep="last")
    after = len(df)
    print(f"[{source}] rows: {before} -> {after} after drop_duplicates on session_id")

    return df

def merge_dfs(mouse: pd.DataFrame, access: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    merged = mouse.merge(access, on="session_id", how=how, suffixes=("_mouse", "_access"))
    if "label_access" in merged.columns or "label_mouse" in merged.columns:
        merged["label"] = merged.get("label_access").fillna(merged.get("label_mouse"))
        # 라벨 충돌 감지
        if "label_access" in merged.columns and "label_mouse" in merged.columns:
            conflict = merged[
                merged["label_access"].notna()
                & merged["label_mouse"].notna()
                & (merged["label_access"] != merged["label_mouse"])
            ][["session_id", "label_access", "label_mouse"]]
            if not conflict.empty:
                conflict.to_csv(REPORT_DIR / "label_conflicts.csv", index=False, encoding="utf-8")
                print(f"Label conflicts: {len(conflict)} saved to output/merge_reports/label_conflicts.csv")

    return merged

def report_keys(mouse: pd.DataFrame, access: pd.DataFrame):
    m_keys = set(mouse["session_id"])
    a_keys = set(access["session_id"])
    inter = m_keys & a_keys
    only_m = m_keys - a_keys
    only_a = a_keys - m_keys

    print(f"mouse sessions: {len(m_keys)}, access sessions: {len(a_keys)}")
    print(f"intersection: {len(inter)}, only_mouse: {len(only_m)}, only_access: {len(only_a)}")

    pd.DataFrame(sorted(list(only_m)), columns=["session_id"]).to_csv(REPORT_DIR / "only_mouse.csv", index=False)
    pd.DataFrame(sorted(list(only_a)), columns=["session_id"]).to_csv(REPORT_DIR / "only_access.csv", index=False)

def main():
    mouse = load_and_clean(MOUSE_CSV, "mouse")
    access = load_and_clean(ACCESS_CSV, "access")

    report_keys(mouse, access)

    merged = merge_dfs(mouse, access, how="inner")

    cols = ["session_id", "label"]
    mouse_cols = [c for c in merged.columns if c.endswith("_mouse")]
    access_cols = [c for c in merged.columns if c.endswith("_access")]
    other_cols = [c for c in merged.columns if c not in cols + mouse_cols + access_cols]
    merged = merged[cols + mouse_cols + access_cols + other_cols]

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {OUT_CSV} ({len(merged)} rows)")

if __name__ == "__main__":
    main()