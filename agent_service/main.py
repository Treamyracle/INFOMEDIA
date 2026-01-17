import os
import re
import requests
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # [NEW] Untuk serve HTML
from fastapi.responses import FileResponse # [NEW]
from pydantic import BaseModel
import uvicorn

# --- CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NER_SERVICE_URL = os.getenv("NER_SERVICE_URL", "http://localhost:8000/predict")

genai.configure(api_key=GOOGLE_API_KEY)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
Kamu adalah 'Domi', Customer Support AI untuk aplikasi e-wallet bernama 'DompetKu'.
Tugasmu adalah membantu pengguna dalam dua hal saja:
1. Verifikasi Akun (Upgrade ke Premium): Pengguna wajib memberikan Nama Lengkap, NIK, dan Tanggal Lahir.
2. Pengajuan Refund Manual: Pengguna wajib memberikan Nama, Nomor HP akun DompetKu, Nama Bank, dan Nomor Rekening.

ATURAN PENTING:
- Kamu menerima teks dimana data sensitif (PII) sudah disensor menjadi tag seperti [REDACTED_NIK].
- JANGAN meminta user mengulang data yang sudah ada tag [REDACTED_...].
- Jika data belum lengkap, minta data yang kurang saja.
"""

model = genai.GenerativeModel('gemini-2.5-flash-lite', system_instruction=SYSTEM_PROMPT)

app = FastAPI(title="Agent Service")

# [NEW] Mount folder static agar index.html bisa diakses
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- GUARDRAIL FUNCTIONS (REGEX & NER) ---
# ... (Masukkan fungsi guardrail_regex dan guardrail_ner LENGKAP di sini) ...
# ... (Pastikan logic 'requests.post' mengarah ke NER_SERVICE_URL) ...
# --- COPAS FUNGSI GUARDRAIL DARI KODE SEBELUMNYA DI SINI ---
def guardrail_regex(text):
    text = re.sub(r'\b\d{16}\b', '[REDACTED_NIK]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', text)
    text = re.sub(r'(\+62|62|0)8[1-9][0-9]{6,9}', '[REDACTED_PHONE]', text)
    return text

def guardrail_ner(text):
    # Simpel implementasi untuk mempersingkat kode di sini
    # Pastikan pakai kode lengkapmu sebelumnya!
    try:
        response = requests.post(NER_SERVICE_URL, json={"text": text}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            entities = data['data']['entities']
            perf = data.get('performance', {})
            # ... LOGIC REDACTION & FILTER LABEL KAMU ...
            # ... (JANGAN LUPA LOGIC CASE INSENSITIVE & LABEL LIST) ...
            
            # --- VERSI SINGKAT UNTUK CONTOH STRUKTUR ---
            # Di real file, pakai logic lengkapmu ya!
            forbidden_zones = []
            for match in re.finditer(r'\[REDACTED_[A-Z]+\]', text):
                forbidden_zones.append((match.start(), match.end()))
            
            entities.sort(key=lambda x: x['start'], reverse=True)
            temp_text = text
            processed_entities = []
            
            valid_labels = ['PERSON', 'ADDRESS', 'NIK', 'EMAIL', 'PHONE', 'BIRTHDATE', 'BANK_NUM']
            
            for ent in entities:
                label = ent['label']
                start = ent['start']
                end = ent['end']
                real_word = text[start:end]
                ent['text'] = real_word
                
                is_conflict = False
                for f_start, f_end in forbidden_zones:
                    if (start >= f_start and start < f_end) or (end > f_start and end <= f_end):
                        is_conflict = True; break
                if is_conflict: continue
                
                if label in valid_labels:
                    replacement = f"[REDACTED_{label}]"
                    temp_text = temp_text[:start] + replacement + temp_text[end:]
                    processed_entities.append(ent)
            
            return temp_text, processed_entities, perf
    except Exception as e:
        print(f"Error NER: {e}")
    return text, [], {}

# --- ROUTES ---

# Route untuk serve UI HTML saat buka root '/'
@app.get("/")
async def read_index():
    # 1. Ambil lokasi folder di mana main.py berada (yaitu /app)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan dengan folder static dan file index.html
    # Hasilnya akan menjadi: /app/static/index.html
    file_path = os.path.join(base_dir, "static", "index.html")
    
    # 3. Kembalikan file tersebut
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": f"File tidak ditemukan di: {file_path}"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_input = req.message
    
    # Pipeline
    clean_v1 = guardrail_regex(user_input)
    final_clean, entities, perf = guardrail_ner(clean_v1)
    
    try:
        response = model.generate_content(final_clean)
        reply = response.text
    except Exception as e:
        reply = "Maaf, sistem sedang sibuk."

    return {
        "reply": reply,
        "debug": {
            "original": user_input,
            "final_clean": final_clean,
            "entities": entities,
            "performance": perf
        }
    }