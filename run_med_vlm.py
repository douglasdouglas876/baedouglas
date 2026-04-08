#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_med_vlm_all.py

기능
1) 이미지 + 프롬프트만 넣어서 MedGemma 예측 CSV 생성
2) 사용자가 가진 기존 sample.csv 와 uid 기준으로 merge해서 비교 CSV 생성

생성 CSV 컬럼:
uid,MeSH,Problems,image,indication,comparison,findings,impression

비교 CSV 컬럼:
uid,
gt_MeSH,pred_MeSH,
gt_Problems,pred_Problems,
gt_image,pred_image,
gt_indication,pred_indication,
gt_comparison,pred_comparison,
gt_findings,pred_findings,
gt_impression,pred_impression
"""

import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    HAS_BNB = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
REPORT_COLUMNS = ["uid", "MeSH", "Problems", "image", "indication", "comparison", "findings", "impression"]

PROMPT = """You are an expert chest radiology report assistant.

Analyze this single chest X-ray image and return exactly one valid JSON object.

Your goal is to produce a concise radiology-style report using ONLY the image.
Do not invent clinical history, prior exams, pathology, labs, symptoms, or test results.
If a field cannot be determined from the image alone, write "Not provided".

Field instructions:
- "MeSH": short semicolon-separated medical keywords based only on visible findings. If normal, use "normal".
- "Problems": short semicolon-separated diagnosis/problem summary. If no abnormality is seen, use "normal".
- "image": describe the study type briefly, e.g. "Xray Chest PA and Lateral" or "Chest radiograph frontal and lateral views". If exact view is uncertain, use a safe generic phrase.
- "indication": always "Not provided" unless explicitly visible in the image itself.
- "comparison": always "Not provided" unless comparison is explicitly visible in the image itself.
- "findings": concise radiology findings based only on visible image evidence.
- "impression": short final radiology impression.

Rules:
- Use only image evidence.
- If the image appears normal, say so clearly.
- Keep findings and impression consistent.
- Output JSON only.
- Do not output markdown.
- Do not output extra explanation.

Return exactly this schema:
{
  "MeSH": "",
  "Problems": "",
  "image": "",
  "indication": "",
  "comparison": "",
  "findings": "",
  "impression": ""
}
"""

SYSTEM_PROMPT = "You are an expert radiologist specialized in chest X-ray interpretation."


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {v}")


def normalize_text(v: Any, default: str = "Not provided") -> str:
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        items = [str(x).strip() for x in v if str(x).strip()]
        return "; ".join(items) if items else default
    s = str(v).strip()
    return s if s else default


def read_rows_from_csv(input_csv: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_image_rgb(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_json_str(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    raise ValueError("JSON object를 찾지 못했습니다.")


def sanitize_output(obj: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for k in REPORT_COLUMNS[1:]:
        out[k] = normalize_text(obj.get(k, "Not provided"))
    if not out["indication"] or out["indication"].lower() == "none":
        out["indication"] = "Not provided"
    if not out["comparison"] or out["comparison"].lower() in {"none", "none."}:
        out["comparison"] = "Not provided"
    return out


def load_model_and_processor(args):
    model_kwargs = {
        "device_map": "auto" if args.device_map_auto else {"": 0},
        "trust_remote_code": args.trust_remote_code,
    }

    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("4bit를 쓰려면 bitsandbytes가 설치되어 있어야 합니다.")
        torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    return model, processor


def get_model_device(model):
    model_device = getattr(model, "device", None)
    if model_device is None or str(model_device) == "cpu":
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    return model_device


def generate_once(model, processor, image: Image.Image, args, retry_mode: bool = False) -> str:
    prompt = PROMPT
    if retry_mode:
        prompt += "\n\nReminder: return ONLY one valid JSON object with the exact seven keys."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=args.do_pan_and_scan,
    )

    model_device = get_model_device(model)
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )
    if args.cache_implementation != "none":
        gen_kwargs["cache_implementation"] = args.cache_implementation

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
        gen_ids = outputs[0][input_len:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def infer_one_image(model, processor, image_path: Path, args) -> Dict[str, str]:
    image = read_image_rgb(image_path)
    raw_1 = generate_once(model, processor, image, args, retry_mode=False)
    try:
        parsed = sanitize_output(json.loads(extract_json_str(raw_1)))
    except Exception:
        raw_2 = generate_once(model, processor, image, args, retry_mode=True)
        try:
            parsed = sanitize_output(json.loads(extract_json_str(raw_2)))
        except Exception:
            parsed = {
                "MeSH": "Not provided",
                "Problems": "Not provided",
                "image": "Chest radiograph",
                "indication": "Not provided",
                "comparison": "Not provided",
                "findings": "Not provided",
                "impression": "Not provided",
            }
    return parsed


def resolve_image_file(row: Dict[str, Any], image_dir: Path, image_col: str) -> str:
    direct = str(row.get(image_col, "")).strip()
    if direct and (image_dir / direct).exists():
        return direct

    direct2 = str(row.get("image_file", "")).strip()
    if direct2 and (image_dir / direct2).exists():
        return direct2

    uid = str(row.get("uid", "")).strip()
    if uid:
        uid_prefix = uid.split(".")[0]
        for p in sorted(image_dir.glob(f"{uid_prefix}_*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                return p.name

    uid_digits = re.findall(r"\d+", uid)
    if uid_digits:
        prefix = uid_digits[0]
        for p in sorted(image_dir.glob(f"{prefix}_*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                return p.name
    return ""


def write_prediction_csv(rows: List[Dict[str, Any]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REPORT_COLUMNS})


def write_compare_csv(gt_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], compare_csv: Path):
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    gt_map = {str(r.get("uid", "")).strip(): r for r in gt_rows}
    pred_map = {str(r.get("uid", "")).strip(): r for r in pred_rows}

    uids = []
    seen = set()
    for source in (gt_rows, pred_rows):
        for r in source:
            uid = str(r.get("uid", "")).strip()
            if uid and uid not in seen:
                seen.add(uid)
                uids.append(uid)

    fieldnames = ["uid"]
    for col in REPORT_COLUMNS[1:]:
        fieldnames.extend([f"gt_{col}", f"pred_{col}"])

    with open(compare_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid in uids:
            gt = gt_map.get(uid, {})
            pred = pred_map.get(uid, {})
            row = {"uid": uid}
            for col in REPORT_COLUMNS[1:]:
                row[f"gt_{col}"] = normalize_text(gt.get(col, ""))
                row[f"pred_{col}"] = normalize_text(pred.get(col, ""))
            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    p.add_argument("--image_dir", type=str, default="/root/data/images")
    p.add_argument("--input_csv", type=str, default="/root/data/label/sample.csv")
    p.add_argument("--image_col", type=str, default="image")
    p.add_argument("--pred_csv", type=str, default="/root/data/label/medgemma_pred_report.csv")
    p.add_argument("--compare_csv", type=str, default="/root/data/label/medgemma_compare_report.csv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--load_in_4bit", type=str2bool, default=False)
    p.add_argument("--do_pan_and_scan", type=str2bool, default=False)
    p.add_argument("--device_map_auto", type=str2bool, default=True)
    p.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager", "none"])
    p.add_argument("--cache_implementation", type=str, default="none", choices=["none", "static", "dynamic"])
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--trust_remote_code", type=str2bool, default=False)
    return p.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    input_csv = Path(args.input_csv)
    pred_csv = Path(args.pred_csv)
    compare_csv = Path(args.compare_csv)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir가 없습니다: {image_dir}")
    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv가 없습니다: {input_csv}")

    gt_rows = read_rows_from_csv(input_csv)
    if args.limit > 0:
        gt_rows = gt_rows[:args.limit]
    if not gt_rows:
        raise ValueError("입력 CSV에 처리할 행이 없습니다.")

    print("=" * 80)
    print("run_med_vlm_all.py 시작")
    print(f"model_id       : {args.model_id}")
    print(f"image_dir      : {image_dir}")
    print(f"input_csv      : {input_csv}")
    print(f"pred_csv       : {pred_csv}")
    print(f"compare_csv    : {compare_csv}")
    print(f"limit          : {args.limit}")
    print(f"max_new_tokens : {args.max_new_tokens}")
    print(f"load_in_4bit   : {args.load_in_4bit}")
    print("=" * 80)

    model, processor = load_model_and_processor(args)

    pred_rows = []
    total = len(gt_rows)

    for idx, row in enumerate(gt_rows, start=1):
        uid = str(row.get("uid", idx)).strip()
        image_file = resolve_image_file(row, image_dir, args.image_col)
        image_path = image_dir / image_file if image_file else None

        print(f"[{idx}/{total}] uid={uid} -> image={image_file or 'NOT_FOUND'}")

        if not image_file or image_path is None or not image_path.exists():
            pred_rows.append({
                "uid": uid,
                "MeSH": "Not provided",
                "Problems": "Not provided",
                "image": "Chest radiograph",
                "indication": "Not provided",
                "comparison": "Not provided",
                "findings": "Image not found",
                "impression": "Image not found",
            })
            continue

        try:
            pred = infer_one_image(model, processor, image_path, args)
            pred_rows.append({
                "uid": uid,
                "MeSH": pred["MeSH"],
                "Problems": pred["Problems"],
                "image": pred["image"],
                "indication": pred["indication"],
                "comparison": pred["comparison"],
                "findings": pred["findings"],
                "impression": pred["impression"],
            })
        except Exception as e:
            pred_rows.append({
                "uid": uid,
                "MeSH": "Not provided",
                "Problems": "Not provided",
                "image": "Chest radiograph",
                "indication": "Not provided",
                "comparison": "Not provided",
                "findings": f"runtime_error: {type(e).__name__}",
                "impression": f"runtime_error: {e}",
            })

    write_prediction_csv(pred_rows, pred_csv)
    write_compare_csv(gt_rows, pred_rows, compare_csv)

    print(f"\n예측 CSV 저장 완료: {pred_csv}")
    print(f"비교 CSV 저장 완료: {compare_csv}")
    print("전체 완료")


if __name__ == "__main__":
    main()
