"""OpenDART API 클라이언트 – 재무제표 원시 데이터 수집."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

# ── 엔드포인트 ────────────────────────────────────────────────
BASE_URL = "https://opendart.fss.or.kr/api"
CORP_CODE_ENDPOINT = f"{BASE_URL}/corpCode.xml"
FIN_STMT_ALL_ENDPOINT = f"{BASE_URL}/fnlttSinglAcntAll.json"

# ── 보고서 코드 (분기별) ─────────────────────────────────────
REPORT_CODES = {
    "Q1": "11013",   # 1분기보고서
    "H1": "11012",   # 반기보고서
    "Q3": "11014",   # 3분기보고서
    "ANNUAL": "11011",  # 사업보고서
}

# ── 기본 경로 ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORP_XML_PATH = DATA_DIR / "corpCode.xml"


class DartApiError(Exception):
    """OpenDART API 호출 오류."""


# ── 환경 변수 / .env 파일 읽기 ────────────────────────────────
def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_api_key(explicit_key: str | None = None) -> str:
    """DART API 키를 반환. 우선순위: 인자 > 환경변수 > .env 파일."""
    if explicit_key:
        return explicit_key
    project_root = Path(__file__).resolve().parent.parent
    env = _read_env_file(project_root / ".env")
    key = os.getenv("DART_API_KEY") or env.get("DART_API_KEY")
    if not key:
        raise DartApiError(
            "API 키가 없습니다. DART_API_KEY 환경변수를 설정하거나 .env 파일에 추가하세요."
        )
    return key


# ── HTTP 유틸 ─────────────────────────────────────────────────
def _http_get(url: str, params: dict[str, str], timeout: int = 30) -> bytes:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(full_url, method="GET")
    context = None
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise DartApiError(f"네트워크 오류: {e}") from e


# ── 기업 코드 관련 ────────────────────────────────────────────
def download_corp_codes(api_key: str, out_path: Path = CORP_XML_PATH) -> Path:
    """OpenDART에서 기업코드 XML 다운로드."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _http_get(CORP_CODE_ENDPOINT, {"crtfc_key": api_key})
    with zipfile.ZipFile(BytesIO(data)) as zf:
        name = zf.namelist()[0]
        out_path.write_bytes(zf.read(name))
    return out_path


def load_corp_codes(xml_path: Path = CORP_XML_PATH) -> list[dict[str, str]]:
    """기업코드 XML → [{corp_code, corp_name, stock_code, ...}, ...]"""
    if not xml_path.exists():
        raise DartApiError(
            f"{xml_path}에 기업코드 XML이 없습니다. "
            "먼저 download_corp_codes()를 실행하세요."
        )
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows: list[dict[str, str]] = []
    for node in root.findall("list"):
        rows.append({
            "corp_code": (node.findtext("corp_code") or "").strip(),
            "corp_name": (node.findtext("corp_name") or "").strip(),
            "stock_code": (node.findtext("stock_code") or "").strip(),
            "modify_date": (node.findtext("modify_date") or "").strip(),
        })
    return rows


def find_corp(
    corp_name: str | None = None,
    stock_code: str | None = None,
    xml_path: Path = CORP_XML_PATH,
    limit: int = 20,
) -> list[dict[str, str]]:
    """기업명 또는 종목코드로 DART corp_code 검색."""
    rows = load_corp_codes(xml_path)
    results: list[dict[str, str]] = []
    name_q = (corp_name or "").lower()
    stock_q = (stock_code or "").strip()

    for row in rows:
        if name_q and name_q not in row["corp_name"].lower():
            continue
        if stock_q and stock_q != row["stock_code"]:
            continue
        results.append(row)
        if len(results) >= limit:
            break
    return results


def resolve_corp_code(
    api_key: str,
    corp_code: str | None = None,
    stock_code: str | None = None,
    corp_name: str | None = None,
) -> str:
    """corp_code / stock_code / corp_name 중 하나로 DART corp_code를 확정."""
    if corp_code:
        return corp_code

    xml_path = CORP_XML_PATH
    if not xml_path.exists():
        download_corp_codes(api_key, xml_path)

    if stock_code:
        results = find_corp(stock_code=stock_code, xml_path=xml_path, limit=1)
    elif corp_name:
        results = find_corp(corp_name=corp_name, xml_path=xml_path, limit=1)
    else:
        raise DartApiError("corp_code, stock_code, corp_name 중 하나를 제공하세요.")

    if not results:
        raise DartApiError(
            f"기업을 찾을 수 없습니다. (stock_code={stock_code}, corp_name={corp_name})"
        )
    return results[0]["corp_code"]


# ── 재무제표 조회 ─────────────────────────────────────────────
def fetch_financial_statements(
    api_key: str,
    corp_code: str,
    bsns_year: str,
    reprt_code: str,
    fs_div: str = "CFS",
) -> list[dict[str, Any]]:
    """
    OpenDART 전체 재무제표 단일회사 조회.

    Returns:
        list of financial statement items (각 계정과목 한 행).
        빈 리스트이면 데이터 없음.
    """
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
        "fs_div": fs_div,
    }
    data = _http_get(FIN_STMT_ALL_ENDPOINT, params)
    payload = json.loads(data.decode("utf-8"))
    status = payload.get("status")
    if status == "013":  # 조회된 데이터가 없음
        return []
    if status and status != "000":
        raise DartApiError(
            f"OpenDART 오류 [{status}]: {payload.get('message')}"
        )
    return payload.get("list", [])


def fetch_all_quarters(
    api_key: str,
    corp_code: str,
    year: str,
    fs_div: str = "CFS",
    quarters: list[str] | None = None,
    delay: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    """
    지정 연도의 분기별 재무제표를 모두 가져옴.

    Args:
        quarters: ["Q1","H1","Q3","ANNUAL"] 중 원하는 것만 지정. None이면 전부.
        delay: API 호출 간 대기 시간(초). OpenDART 분당 호출 제한 방지.

    Returns:
        {"Q1": [...], "H1": [...], ...}
    """
    if quarters is None:
        quarters = list(REPORT_CODES.keys())

    result: dict[str, list[dict[str, Any]]] = {}
    for q in quarters:
        code = REPORT_CODES.get(q)
        if not code:
            raise DartApiError(f"잘못된 분기 코드: {q}. 가능한 값: {list(REPORT_CODES.keys())}")
        items = fetch_financial_statements(api_key, corp_code, year, code, fs_div)
        result[q] = items
        if delay > 0:
            time.sleep(delay)
    return result
