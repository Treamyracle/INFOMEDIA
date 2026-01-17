import os
import re
import requests
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.ai.generativelanguage_v1beta.types import content, SafetySetting
import uvicorn

# --- CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NER_SERVICE_URL = os.getenv("NER_SERVICE_URL", "http://localhost:8000/predict")

genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. MOCK DATABASE ---
DATABASE_USER = {
    "1234567890123456": {
        "nama": "Arif Athaya",
        "email": "arif@example.com",
        "tgl_lahir": "04-10-2005",
        "phone": "08123456789",
        "alamat": "Jl. Discovery Terra B 59",
        "saldo": 5000000,
        "pin": "123456"
    },
    "3201123456789001": {
        "nama": "Budi Santoso",
        "email": "budi@test.com",
        "tgl_lahir": "17-08-1990",
        "phone": "089988776655",
        "alamat": "Jl. Sudirman No 1 Jakarta",
        "saldo": 150000,
        "pin": "654321"
    }
}

# --- 2. SESSION STORAGE ---
SESSION_DATA = {} 

# --- 3. DEFINISI TOOLS ---
def ganti_password(nik_tag: str, email_tag: str, birthdate_tag: str):
    """Mengganti password akun. Memverifikasi NIK, Email, dan Tanggal Lahir user."""
    real_nik = SESSION_DATA.get(nik_tag)
    real_email = SESSION_DATA.get(email_tag)
    real_birthdate = SESSION_DATA.get(birthdate_tag)

    if not all([real_nik, real_email, real_birthdate]):
        return "GAGAL: Data NIK, Email, atau Tanggal Lahir tidak lengkap."

    user = DATABASE_USER.get(real_nik)
    if not user:
        return f"GAGAL: NIK {real_nik} tidak terdaftar."

    if user['email'].lower() == real_email.lower() and user['tgl_lahir'] == real_birthdate:
        return f"BERHASIL: Link reset password telah dikirim ke email {real_email}."
    else:
        return "GAGAL: Email atau Tanggal Lahir tidak cocok."

def request_kartu_fisik(nama_tag: str, alamat_tag: str, phone_tag: str):
    """Request kartu debit fisik."""
    real_nama = SESSION_DATA.get(nama_tag)
    real_alamat = SESSION_DATA.get(alamat_tag)
    real_phone = SESSION_DATA.get(phone_tag)

    if not all([real_nama, real_alamat, real_phone]):
        return "GAGAL: Data Nama, Alamat, atau No HP tidak lengkap."

    return {
        "status": "DIPROSES",
        "message": f"Permintaan kartu fisik a.n '{real_nama}' diterima. Dikirim ke '{real_alamat}'."
    }

def withdraw_ke_bank(nik_tag: str, bank_num_tag: str, nama_pemilik_tag: str):
    """Pencairan saldo (Withdraw) ke rekening Bank."""
    real_nik = SESSION_DATA.get(nik_tag)
    real_bank_num = SESSION_DATA.get(bank_num_tag)
    real_nama_pemilik = SESSION_DATA.get(nama_pemilik_tag)

    if not all([real_nik, real_bank_num, real_nama_pemilik]):
        return "GAGAL: Data NIK, Nomor Rekening, atau Nama Pemilik tidak terdeteksi."

    user = DATABASE_USER.get(real_nik)
    if not user: 
        return f"GAGAL: NIK {real_nik} tidak ditemukan."

    if real_nama_pemilik.lower() not in user['nama'].lower():
        return f"GAGAL: Nama pemilik rekening ({real_nama_pemilik}) TIDAK SESUAI dengan pemilik akun ({user['nama']})."

    if user['saldo'] < 50000:
        return f"GAGAL: Saldo tidak mencukupi. Saldo saat ini: {user['saldo']}, Min Withdraw: 50.000."
    
    sisa_saldo = user['saldo'] - 50000
    DATABASE_USER[real_nik]['saldo'] = sisa_saldo
    
    return {
        "status": "BERHASIL",
        "message": f"Dana berhasil dicairkan ke rekening {real_bank_num} a.n {real_nama_pemilik}. Sisa saldo: {sisa_saldo}"
    }

my_tools = [ganti_password, request_kartu_fisik, withdraw_ke_bank]

# ... (kode import dan tools tetap sama) ...

# --- 4. SYSTEM PROMPT & MODEL SETUP ---
SYSTEM_PROMPT = """
Kamu adalah 'Domi', AI Customer Service E-Wallet.
Tugas: Membantu user menggunakan tools (Ganti Password, Request Kartu, Withdraw).

RULES:
1. Gunakan tag [REDACTED_...] apa adanya saat memanggil tools.
2. Jika tool return "GAGAL", katakan ke user permintaan DITOLAK beserta alasannya.
3. Jika tool return "BERHASIL", konfirmasi ke user dengan ramah.
"""

# [PERBAIKAN 1] Matikan Safety Filter (BLOCK_NONE)
# Ini WAJIB agar AI tidak memblokir respon soal "Transfer/Uang"
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
]

# [PERBAIKAN 2] Gunakan Nama Model yang Benar (PILIH SALAH SATU)
# Opsi A: Model Terbaru (Rekomendasi)
MODEL_NAME = 'gemini-2.5-flash' 
# Opsi B: Model Stabil (Jika Opsi A error)
# MODEL_NAME = 'gemini-1.5-flash' 

model = genai.GenerativeModel(
    MODEL_NAME, 
    tools=my_tools, 
    system_instruction=SYSTEM_PROMPT,
    safety_settings=safety_settings  # <-- Pastikan ini dimasukkan
)

# ... (sisa kode routes ke bawah tetap sama) ...

app = FastAPI(title="Agent Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 5. GUARDRAILS ---
def guardrail_regex(text):
    patterns = {
        r'\b\d{16}\b': '[REDACTED_NIK]',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '[REDACTED_EMAIL]',
        r'(\+62|62|0)8[1-9][0-9]{6,11}': '[REDACTED_PHONE]',
        r'\b\d{2}-\d{2}-\d{4}\b': '[REDACTED_BIRTHDATE]', 
        r'\b\d{10,12}\b': '[REDACTED_BANK_NUM]' 
    }
    for pattern, tag in patterns.items():
        for match in re.finditer(pattern, text):
            original = match.group(0)
            if tag == '[REDACTED_BANK_NUM]' and (original.startswith("08") or original.startswith("62")): continue
            SESSION_DATA[tag] = original
            text = text.replace(original, tag)
    return text

def guardrail_ner(text):
    try:
        response = requests.post(NER_SERVICE_URL, json={"text": text}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            entities = data['data']['entities']
            perf = data.get('performance', {})
            
            forbidden_zones = []
            for match in re.finditer(r'\[REDACTED_[A-Z]+\]', text):
                forbidden_zones.append((match.start(), match.end()))
            
            entities.sort(key=lambda x: x['start'], reverse=True)
            valid_labels = ['PERSON', 'ADDRESS', 'NIK', 'EMAIL', 'PHONE', 'BIRTHDATE', 'BANK_NUM']
            
            for ent in entities:
                label = ent['label']
                start, end = ent['start'], ent['end']
                is_conflict = False
                for f_start, f_end in forbidden_zones:
                    if (start >= f_start and start < f_end) or (end > f_start and end <= f_end):
                        is_conflict = True; break
                if is_conflict: continue
                
                if label in valid_labels:
                    real_word = text[start:end]
                    tag = f"[REDACTED_{label}]"
                    SESSION_DATA[tag] = real_word
                    text = text[:start] + tag + text[end:]
            return text, entities, perf
    except Exception as e:
        print(f"Error NER: {e}")
    return text, [], {}

# --- ROUTES ---

@app.get("/")
async def read_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "static", "index.html")
    if os.path.exists(file_path): return FileResponse(file_path)
    return {"error": "File not found"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_input = req.message
    
    clean_v1 = guardrail_regex(user_input)
    final_clean, entities, perf = guardrail_ner(clean_v1)
    
    try:
        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(final_clean)
        
        # [PERBAIKAN 3] Cek apakah response memiliki parts sebelum akses .text
        if response.parts:
            reply = response.text
        else:
            # Fallback jika model stuck
            if response.candidates and response.candidates[0].finish_reason == 1:
                 reply = "Permintaan telah diproses oleh sistem, namun AI tidak memberikan respon teks. Silakan cek saldo Anda di dashboard."
            else:
                 reply = "Maaf, respon AI terblokir oleh filter keamanan."

    except Exception as e:
        reply = f"Maaf, ada kesalahan sistem: {str(e)}"

    return {
        "reply": reply,
        "debug": {
            "original": user_input,
            "final_clean": final_clean,
            "session_data": SESSION_DATA,
            "entities": entities,
            "performance": perf,
            "database": DATABASE_USER
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)