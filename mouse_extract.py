import json
from multiprocessing import Value
import re
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
log = logging.getLogger()

ROOT_DIR = Path("data") / "mouse_movements"
TARGET_FILE = "mouse_movements.json"
OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "mouse_features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_label(session_id: str) -> int:
    return 1 if "bot" in session_id.lower() else 0

MOUSE_PATTERN = re.compile(r"\[([mcs])\(\s*([^]]*?)\s*\)\]")

def parse_behaviour(behaviour: str) -> List[Dict]:
    """
    입력: "[m(597,562)][m(597,561)]...[c(l)]..."
    출력: [{'type': 'm', 'x': 597, 'y': 562, 't': 0.0}, ...]
    """
    events = []
    ts = 0.0
    interval_sec = 1 / 60 # 60fps
    
    if not behaviour or not behaviour.strip():
        return events
    
    for match in MOUSE_PATTERN.finditer(behaviour):
        try:
            typ = match.group(1)
            val = match.group(2).strip()

            if typ == "m":
                parts = [p.strip() for p in val.split(",") if p.strip()]
                if len(parts) == 2:
                    x, y = int(parts[0]), int(parts[1])
                    events.append({"type": "m", "x": x, "y": y, "t": ts})
            elif typ == "c":
                events.append({"type": "c", "t": ts})
            elif typ == "s":
                try:
                    scroll = int(val)
                    events.append({"type": "s", "scroll": scroll, "t": ts})
                except ValueError:
                    continue
            ts += interval_sec
            
        except (IndexError, ValueError, AttributeError) as e:
            continue
        
    return events

def extract_features(session_id: str, events: List[Dict], label: int) -> Optional[Dict]:
    if len(events) < 10:
        return None
    
    df = pd.DataFrame(events)
    duration = df["t"].iloc[-1] if len(df) > 0 else 0.0
    
    move = df[df["type"] == "m"]
    if len(move) < 5:
        return None
    
    x = move["x"].values.astype(float)
    y = move["y"].values.astype(float)
    mt = move["t"].values
    
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(mt) + 1e-9
    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / dt
    total_dist = dist.sum()
    avg_speed = total_dist / duration if duration > 0 else 0.0
    
    intervals_ms = np.diff(mt) * 1000
    interval_mean = intervals_ms.mean()
    interval_std = intervals_ms.std()
    regularity = 1 - (interval_std / interval_mean + 1e-6)
    
    if len(speed) > 1:
        accel = np.diff(speed) / (dt[1:] + 1e-9)
        accel_abs_mean = np.mean(np.abs(accel))
    else:
        accel_abs_mean = 0.0
        
    if len(dx) > 1:
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        curv_num = np.abs(dx[:-1] * ddy - dy[:-1] * ddx)
        curv_den = (dx[:-1]**2 + dy[:-1]**2) ** 1.5 + 1e-9
        curvature_mean = curv_num.mean() / curv_den.mean()
    else:
        curvature_mean = 0.0
        
    stop_ratio = (speed < 1.0).mean() # 1px/sec 이하
    
    click_count = int((df["type"] == "c").sum())
    scroll_amount = int(df[df["type"] == "s"]["scroll"].sum()) if "scroll" in df.columns else 0
    
    area = (x.max() - x.min()) * (y.max() - y.min())
    
    return {
        "session_id": session_id,
        "label": label,
        "event_count": len(events),
        "duration_sec": round(duration, 4),
        "mouse_move_count": len(move),
        "total_distance_px": round(total_dist, 2),
        "avg_speed_px_s": round(avg_speed, 4),
        "intevral_mean_ms": round(interval_mean, 4),
        "interval_std_ms": round(interval_std, 4),
        "regularity": round(regularity, 4),
        "accel_abs_mean": round(accel_abs_mean, 6),
        "curvature_mean": round(curvature_mean, 6),
        "stop_ratio": round(stop_ratio, 4),
        "click_count": click_count,
        "scroll_amount": scroll_amount,
        "covered_area_px2": round(area, 2),
    }

def process_session(session_dir: Path) -> Optional[Dict]:
    json_path = session_dir / TARGET_FILE
    if not json_path.exists():
        log.warning(f"JSON 파일이 없습니다: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        session_id = data.get("session_id", session_dir.name)
        behaviour = data.get("total_behaviour", "")
        if not behaviour or not behaviour.strip():
            log.info(f"빈 behaviour: {session_id}")
            return None
        
        events = parse_behaviour(behaviour)
        if not events:
            log.info(f"파싱된 이벤트 없음: {session_id}")
            return None
    
        return extract_features(session_id, events, get_label(session_id))
    
    except json.JSONDecodeError as e:
        log.error(f"JSON 파싱 실패: {session_id} - {e}")
        return None
    except Exception as e:
        log.error(f"{session_dir.name} 처리 실패: {e}")
        return None

def main():
    if not ROOT_DIR.exists():
        log.error(f"데이터 폴더가 없습니다: {ROOT_DIR}")
        return
    
    session_dirs = [p for p in ROOT_DIR.iterdir() if p.is_dir()]
    log.info(f"발견된 세션 폴더: {len(session_dirs)}개")
    
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_session, d): d for d in session_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="피처 추출"):
            result = future.result()
            if result:
                results.append(result)
                
    if not results:
        log.warning("피처 추출 결과 없음")
        return
    
    df = pd.DataFrame(results)
    df = df[sorted(df.columns)]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    human_cnt = df["label"].value_counts().get(0, 0)
    bot_cnt = df["label"].value_counts().get(1, 0)
    log.info(f"완료! -> {OUTPUT_CSV}")
    log.info(f"   총 {len(df)}개 세션 | 인간: {human_cnt}, 봇: {bot_cnt}")
    
if __name__ == "__main__":
    main()
    
