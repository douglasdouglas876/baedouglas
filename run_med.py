import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from huggingface_hub import login

# ==================================================
# 0. 사용자 설정 영역
# ==================================================
model_id = "google/medgemma-4b-it"

SYSTEM_PROMPT = """
너는 신중하고 정확하게 답하는 의학 AI 어시스턴트다.
반드시 다음 원칙을 따른다:
1. 응급 증상이 의심되면 즉시 응급 대응을 우선 안내한다.
2. 진단을 확정하지 말고, 가능한 질환과 이유를 설명한다.
3. 위험 신호(red flag)를 분명하게 알려준다.
4. 일반인이 이해하기 쉽게 한국어로 설명한다.
5. 생명을 위협할 수 있는 상황이면 지체 없이 119 또는 응급실 방문을 권고한다.
""".strip()

USER_PROMPT = """
갑자기 한쪽 팔다리에 힘이 빠지고 말이 어눌해지는 증상이 나타나면
어떤 질환을 가장 먼저 의심해야 하고, 즉시 어떻게 대처해야 해?
""".strip()

# ==================================================
# 1. HF 로그인
# ==================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        print("Hugging Face 로그인 성공")
    except Exception as e:
        print(f"Hugging Face 로그인 경고: {e}")
else:
    print("경고: HF_TOKEN 환경변수가 없거나 올바르지 않습니다. 이미 로그인되어 있으면 계속 진행합니다.")

# ==================================================
# 2. CUDA 확인
# ==================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 디바이스: {device}")

if device == "cuda":
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"총 GPU 메모리: {total_mem_gb:.2f} GB")

# ==================================================
# 3. 4bit 양자화 설정
# ==================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"\n--- 🚀 모델 로딩 시작: {model_id} ---")

# ==================================================
# 4. 토크나이저 / 모델 로드
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# 5. 메시지 구성
# ==================================================
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

# ==================================================
# 6. chat template 적용
# ==================================================
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

attention_mask = torch.ones_like(input_ids)

print(f"입력 토큰 수: {input_ids.shape[-1]}")
print("--- 🧠 모델이 답변 생성 중... ---")

# ==================================================
# 7. 생성
# ==================================================
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=192,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# ==================================================
# 8. 디코딩
# ==================================================
generated_tokens = outputs[0][input_ids.shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n" + "=" * 60)
print("🩺 응답:\n")
print(response)
print("=" * 60 + "\n")

# ==================================================
# 9. GPU 메모리 출력
# ==================================================
if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"현재 할당된 GPU 메모리: {allocated:.2f} GB")
    print(f"현재 예약된 GPU 메모리: {reserved:.2f} GB")