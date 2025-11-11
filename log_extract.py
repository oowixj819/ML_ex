import re
from token import OP
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Dict, Optional

# 확인용 로그
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  # ← :s 제거!
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
BOT_DIR = SCRIPT_DIR / "data" / "logs" / "bot"
HUMAN_DIR = SCRIPT_DIR / "data" / "logs" / "human"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_CSV = OUTPUT_DIR / "log_features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(f"봇 로그 폴더: {BOT_DIR}")
log.info(f"인간 로그 폴더: {HUMAN_DIR}")
log.info(f"결과 저장: {OUTPUT_CSV}")

# 정규표현식 패턴
LOG_PATTERN = re.compile(
    r'^- -\s+\[(?P<time>.+?)\]\s+'
    r'"(?P<request>.+?)"\s+'
    r'(?P<status>\d+)\s+'
    r'(?P<size>\d+|-)\s+'
    r'"(?P<referer>.+?)"\s+-'
)

# 라벨링: 폴더 기준 자동 지정
def get_label(file_path: Path) -> int:
    return 1 if "bot" in file_path.parts else 0

# 로그 라인 -> 이벤트 리스트
def process_access_log(log_path: Path) -> Optional[Dict]:
    if not log_path.exists():
        log.warning(f"파일 없음: {log_path}")
        return None
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip().startswith("- -")]
            
        if not lines:
            log.info(f"유효한 로그 없음: {log_path.name}")
            return None

        session_id = log_path.stem
        label = get_label(log_path)
        
        events = []
        for line in lines:
            m = LOG_PATTERN.match(line)
            if not m:
                continue
            d = m.groupdict()
            try:
                timestamp = datetime.strptime(d['time'], "%d/%b/%Y:%H:%M:%S %z")
            except ValueError:
                continue
                
            size = int(d['size']) if d['size'] != '-' else 0
            status = int(d['status'])
            request = d['request']
            
            parts = request.split()
            method = parts[0] if len(parts) > 0 else "UNKNOWN"
            path = parts[1] if len(parts) > 1 else "/"
            
            events.append({
                "t": timestamp,
                "status": status,
                "size": size,
                "method": method,
                "path": path,
            })
            
        if not events:
            return None
        
        df = pd.DataFrame(events)
        df = df.sort_values("t")
        
        session_start = df["t"].iloc[0]
        session_end = df["t"].iloc[-1]
        duration_sec = (session_end - session_start).total_seconds()
        
        total_requests = len(df)
        error_4xx = len(df[df["status"].between(400, 499)])
        error_5xx = len(df[df["status"].between(500, 599)])
        error_rate = (error_4xx + error_5xx) / total_requests
        
        avg_size = df["size"].mean()
        get_ratio = len(df[df["method"] == "GET"]) / total_requests
        unique_paths = df["path"].nunique()
        unique_status = df["status"].nunique()
        
        intervals = df["t"].diff().dt.total_seconds().dropna()
        interval_mean = intervals.mean() if len(intervals) > 0 else 0
        interval_std = intervals.std() if len(intervals) > 0 else 0
        
        return {
            "session_id": session_id,
            "label": label,
            "request_count": total_requests,
            "duration_sec": round(duration_sec, 2),
            "error_rate": round(error_rate, 4),
            "avg_response_size": round(avg_size, 2),
            "get_request_ratio": round(get_ratio, 4),
            "unique_paths": unique_paths,
            "unique_status_codes": unique_status,
            "request_interval_mean_sec": round(interval_mean, 4),
            "request_interval_std_sec": round(interval_std, 4),
        }
        
    except Exception as e:
        log.error(f"{log_path.name} 처리 실패: {e}")
        return None
    
def get_all_log_files() -> List[Path]:
    files =[]
    if BOT_DIR.exists():
        files.extend(p for p in BOT_DIR.iterdir() if p.suffix.lower() == ".log")
    if HUMAN_DIR.exists():
        files.extend(p for p in HUMAN_DIR.iterdir() if p.suffix.lower() == ".log")
    return files

def main():
    log_files = get_all_log_files()
    if not log_files:
        log.error("로그 파일 없음")
        return
    
    log.info(f"발견된 access log: {len(log_files)}개")
    
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_access_log, f): f for f in log_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="access log feature 추출"):
            r = future.result()  
            if r:
                results.append(r)
                
    if not results:
        log.warning("추출된 feature 없음")
        return
    
    df = pd.DataFrame(results)
    df = df[sorted(df.columns)]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    human = df["label"].value_counts().get(0, 0)
    bot = df["label"].value_counts().get(1, 0)
    log.info(f"완료! -> {OUTPUT_CSV}")
    log.info(f"   총 {len(df)}개 세션 | 인간: {human}, 봇: {bot}")
    
if __name__ == "__main__":
    main()
        
        