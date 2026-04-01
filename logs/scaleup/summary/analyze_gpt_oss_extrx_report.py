#!/usr/bin/env python3
import json
import math
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
REGEX_LIST = ROOT / "datasets/scaleup/regex_list.json"
REPORT_MD = ROOT / "logs/scaleup/summary/oss_extrx_ce_std_analysis.md"
REPORT_XLSX = ROOT / "logs/scaleup/summary/oss_extrx_ce_std_analysis.xlsx"
REPORT_XLSX_ALT = ROOT / "logs/scaleup/summary/oss_extrx_ce_std_analysis.updated.xlsx"

CONFIGS = {
    "std": {
        "title": "STD",
        "dir": ROOT / "logs/scaleup/icl_gen_extrx/model=gpt-oss/std/reg",
        "split_token": "_totTrain=",
    },
    "ce_agentic": {
        "title": "CE / agentic_reflection",
        "dir": ROOT / "logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/agentic_reflection",
        "split_token": "_ceEpochs=",
    },
    "ce_single": {
        "title": "CE / single_inference",
        "dir": ROOT / "logs/scaleup/icl_gen_extrx/model=gpt-oss/ce/reg/single_inference",
        "split_token": "_ceEpochs=",
    },
}


def parse_regex(filename, split_token):
    prefix = "msgdict_regex="
    if not filename.startswith(prefix) or split_token not in filename:
        raise ValueError(f"Unexpected filename: {filename}")
    body = filename[len(prefix) :]
    return body.split(split_token, 1)[0]


def median_or_nan(values):
    return statistics.median(values) if values else float("nan")


def fmt_num(value):
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def bar(solved, total, width=10):
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * solved / total)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def summarize_log(path):
    payload = json.loads(path.read_text())
    run_keys = sorted(payload["summary"].keys(), key=lambda x: int(x.split("-")[1]))
    solved_runs = 0
    solved_samples = []
    solved_rounds = []

    for run_key in run_keys:
        if float(payload["summary"][run_key]["final_accuracy"]) >= 1.0:
            solved_runs += 1
        epoch_keys = sorted(payload[run_key].keys(), key=lambda x: int(x.split("-")[1]))
        for idx, epoch_key in enumerate(epoch_keys, start=1):
            epoch = payload[run_key][epoch_key]
            if float(epoch["Accuracy"]) >= 1.0:
                solved_samples.append(int(epoch["NumTrainingSamples"]))
                solved_rounds.append(idx)
                break

    total_runs = len(run_keys)
    return {
        "solved_runs": solved_runs,
        "runs": total_runs,
        "solve_rate": solved_runs / total_runs if total_runs else 0.0,
        "median_solved_samples": median_or_nan(solved_samples),
        "median_solved_rounds": median_or_nan(solved_rounds),
    }


def load_extrx_metadata():
    payload = json.loads(REGEX_LIST.read_text())["extrx"]

    mapping = {}
    states = []
    depths = set()
    valid_cells = set()
    template_cells = set()

    for row in payload:
        state = int(row["#States"])
        states.append(state)
        for item in row["regex_list"]:
            depth = int(item["Stardepth"])
            depths.add(depth)
            template_cells.add((state, depth))
            regexes = item.get("regex_list", [])
            if regexes:
                valid_cells.add((state, depth))
            for regex in regexes:
                mapping[regex] = (state, depth)

    return {
        "mapping": mapping,
        "states": sorted(states),
        "depths": sorted(depths),
        "valid_cells": valid_cells,
        "template_cells": template_cells,
    }


def collect_rows():
    meta = load_extrx_metadata()
    mapping = meta["mapping"]
    rows = {}

    for key, cfg in CONFIGS.items():
        for path in sorted(cfg["dir"].glob("*.json")):
            regex = parse_regex(path.name, cfg["split_token"])
            state_depth = mapping.get(regex)
            if state_depth is None:
                raise KeyError(f"Regex missing from regex_list.json: {regex}")
            rows.setdefault(
                regex,
                {"regex": regex, "states": state_depth[0], "stardepth": state_depth[1]},
            )
            rows[regex][key] = summarize_log(path)

    return sorted(rows.values(), key=lambda x: (x["states"], x["stardepth"], x["regex"])), meta


def overall_stats(rows, key):
    available = [row for row in rows if key in row]
    solved_runs = sum(row[key]["solved_runs"] for row in available)
    total_runs = sum(row[key]["runs"] for row in available)
    full_cells = sum(1 for row in available if row[key]["solved_runs"] == row[key]["runs"])
    return {
        "covered_cells": len(available),
        "solved_runs": solved_runs,
        "total_runs": total_runs,
        "solve_rate": solved_runs / total_runs if total_runs else 0.0,
        "full_cells": full_cells,
    }


def cell_text(cell):
    return (
        f"{cell['solved_runs']}/{cell['runs']} {bar(cell['solved_runs'], cell['runs'])}; "
        f"s={fmt_num(cell['median_solved_samples'])}; "
        f"r={fmt_num(cell['median_solved_rounds'])}"
    )


def matrix_md(rows, key, meta):
    lookup = {(row["states"], row["stardepth"]): row[key] for row in rows if key in row}
    header = ["#States"] + [f"StarDepth={d}" for d in meta["depths"]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for state in meta["states"]:
        cells = []
        for depth in meta["depths"]:
            state_depth = (state, depth)
            if state_depth not in meta["valid_cells"]:
                cells.append("NA")
            elif state_depth not in lookup:
                cells.append("MISSING")
            else:
                cells.append(cell_text(lookup[state_depth]))
        lines.append(f"| {state} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(rows, meta):
    valid_total = len(meta["valid_cells"])
    template_total = len(meta["template_cells"])
    stats = {key: overall_stats(rows, key) for key in CONFIGS}

    lines = []
    lines.append("# gpt-oss extrx: STD vs CE")
    lines.append("")
    lines.append("分析目录：`logs/scaleup/icl_gen_extrx/model=gpt-oss`")
    lines.append("")
    lines.append("说明：")
    lines.append("- 这份表按 `datasets/scaleup/regex_list.json` 里的 `extrx` 元信息整理，格式对齐 `simplyrx` report。")
    lines.append("- 单元格格式：`solved/total [progress]; s=<med #solved samples>; r=<med #solved rounds>`")
    lines.append("- `s`：只在 solved runs 上统计，取首次解出时 `NumTrainingSamples` 的中位数")
    lines.append("- `r`：只在 solved runs 上统计，取首次解出轮次的中位数，按 1-based 轮次计数")
    lines.append("- `NA` 表示该格子在 `regex_list.json` 中没有定义 regex；`MISSING` 表示定义了 regex，但当前日志里没有对应结果。")
    lines.append("")
    lines.append("## 总体观察")
    lines.append("")

    for key, cfg in CONFIGS.items():
        stat = stats[key]
        missing = valid_total - stat["covered_cells"]
        lines.append(
            f"- `{cfg['title']}`: solve rate = `{stat['solve_rate']:.3f}` "
            f"(`{stat['solved_runs']}/{stat['total_runs']}` runs), "
            f"`3/3` 格子数 = `{stat['full_cells']}/{stat['covered_cells']}`, "
            f"已覆盖 `{stat['covered_cells']}/{valid_total}` 个有效格子"
            + (f"，缺 `{missing}` 个。" if missing else "。")
        )

    lines.append(
        f"- `regex_list.json` 为 `extrx` 提供了 `{valid_total}` 个有效格子；"
        f"整个 `#States × Stardepth` 外框共有 `{template_total}` 个位置，其中 `{template_total - valid_total}` 个位置本身未定义 regex。"
    )
    lines.append("")

    for key, cfg in CONFIGS.items():
        lines.append(f"## {cfg['title']}")
        lines.append("")
        lines.append(matrix_md(rows, key, meta))
        lines.append("")

    return "\n".join(lines)


def col_name(idx):
    out = []
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        out.append(chr(65 + rem))
    return "".join(reversed(out))


def build_sheet_rows(rows, meta):
    data = [
        ["gpt-oss extrx: STD vs CE"],
        [""],
        ["分析目录", "logs/scaleup/icl_gen_extrx/model=gpt-oss"],
        [""],
    ]

    for key, cfg in CONFIGS.items():
        data.append([cfg["title"]])
        data.append(["#States"] + [f"StarDepth={d}" for d in meta["depths"]])
        lookup = {(row["states"], row["stardepth"]): row[key] for row in rows if key in row}
        for state in meta["states"]:
            values = []
            for depth in meta["depths"]:
                state_depth = (state, depth)
                if state_depth not in meta["valid_cells"]:
                    values.append("NA")
                elif state_depth not in lookup:
                    values.append("MISSING")
                else:
                    values.append(cell_text(lookup[state_depth]))
            data.append([state] + values)
        data.append([""])

    return data


def sheet_xml(rows, meta):
    data = build_sheet_rows(rows, meta)
    row_xml = []
    for r_idx, row in enumerate(data, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{col_name(c_idx)}{r_idx}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>'
        '<col min="1" max="1" width="14" customWidth="1"/>'
        '<col min="2" max="5" width="28" customWidth="1"/>'
        '</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(rows, meta, out_path):
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="extrx_ce_std" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml(rows, meta))
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)


def main():
    rows, meta = collect_rows()
    REPORT_MD.write_text(build_markdown(rows, meta), encoding="utf-8")

    try:
        write_xlsx(rows, meta, REPORT_XLSX)
        written_xlsx = REPORT_XLSX
    except PermissionError:
        write_xlsx(rows, meta, REPORT_XLSX_ALT)
        written_xlsx = REPORT_XLSX_ALT

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {written_xlsx}")


if __name__ == "__main__":
    main()
