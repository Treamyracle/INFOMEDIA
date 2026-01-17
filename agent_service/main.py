import os
import re
import requests
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.ai.generativelanguage_v1beta.types import content # Penting untuk Tools
import uvicorn

# --- CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NER_SERVICE_URL = os.getenv("NER_SERVICE_URL", "http://localhost:8000/predict")

genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. MOCK DATABASE (Simulasi Data Backend) ---
# Ceritanya ini database user yang sudah terdaftar
DATABASE_USER = {
    "1234567890123456": {  # Primary Key: NIK
        "nama": "Arif Athaya",
        "email": "arif@example.com",
        "tgl_lahir": "04-10-2005",
        "phone": "08123456789",
        "alamat": "Jl. Discovery Terra B 59",
        "saldo": 5000000,
        "pin": "123456"
    },
    "3201123456789001": { # Data Dummy Kedua
        "nama": "Budi Santoso",
        "email": "budi@test.com",
        "tgl_lahir": "17-08-1990",
        "phone": "089988776655",
        "alamat": "Jl. Sudirman No 1 Jakarta",
        "saldo": 150000,
        "pin": "654321"
    }
}

# --- 2. SESSION STORAGE (VAULT SEMENTARA) ---
# Best Practice: Di Production gunakan Redis dengan Expiry Time
SESSION_DATA = {} 

# --- 3. DEFINISI TOOLS (LOGIC VERIFIKASI) ---

def ganti_password(nik_tag: str, email_tag: str, birthdate_tag: str):
    """
    Mengganti password akun. Memverifikasi NIK, Email, dan Tanggal Lahir user.
    """
    # Ambil data asli dari Session Vault
    real_nik = SESSION_DATA.get(nik_tag)
    real_email = SESSION_DATA.get(email_tag)
    real_birthdate = SESSION_DATA.get(birthdate_tag)

    # Validasi input lengkap
    if not all([real_nik, real_email, real_birthdate]):
        return "Gagal: Data NIK, Email, atau Tanggal Lahir tidak lengkap/kadaluarsa."

    # Cek apakah User ada di Database
    user = DATABASE_USER.get(real_nik)
    if not user:
        return "Gagal: NIK tidak terdaftar dalam sistem kami."

    # VERIFIKASI: Cocokkan input user dengan data di Database
    # (Apakah email yang diinput == email di database?)
    if user['email'] == real_email and user['tgl_lahir'] == real_birthdate:
        # Jika cocok, simulasi kirim link
        return f"Verifikasi BERHASIL. Link reset password telah dikirim ke email {real_email}. Silakan cek inbox Anda."
    else:
        return "Verifikasi GAGAL: Email atau Tanggal Lahir tidak cocok dengan data NIK tersebut."

def request_kartu_fisik(nama_tag: str, alamat_tag: str, phone_tag: str):
    """
    Request kartu debit fisik. Memerlukan Nama, Alamat Pengiriman, dan No HP penerima.
    """
    real_nama = SESSION_DATA.get(nama_tag)
    real_alamat = SESSION_DATA.get(alamat_tag)
    real_phone = SESSION_DATA.get(phone_tag)

    if not all([real_nama, real_alamat, real_phone]):
        return "Gagal: Data Nama, Alamat, atau No HP tidak lengkap."

    # Logic: Di sini kita bisa saja langsung proses tanpa cek NIK (sesuai request Anda)
    # Atau bisa ditambahkan validasi tambahan jika perlu.
    return {
        "status": "DIPROSES",
        "message": f"Permintaan kartu fisik atas nama '{real_nama}' diterima. Kartu akan dikirim ke '{real_alamat}'. Kurir akan menghubungi {real_phone}."
    }

def withdraw_ke_bank(nik_tag: str, bank_num_tag: str, nama_pemilik_tag: str):
    """
    Pencairan saldo (Withdraw) ke rekening Bank. Memerlukan NIK, Nomor Rekening Tujuan, dan Nama Pemilik Rekening.
    """
    real_nik = SESSION_DATA.get(nik_tag)
    real_bank_num = SESSION_DATA.get(bank_num_tag)
    real_nama_pemilik = SESSION_DATA.get(nama_pemilik_tag)

    if not all([real_nik, real_bank_num, real_nama_pemilik]):
        return "Gagal: Data tidak lengkap."

    # Cek User & Saldo
    user = DATABASE_USER.get(real_nik)
    if not user: return "User tidak ditemukan."

    if user['saldo'] < 50000:
        return "Gagal: Saldo tidak mencukupi (Min. 50.000)."
    
    # Validasi Security: Nama di Rekening vs Nama di Akun
    # (Simple logic: harus mengandung kata yang sama)
    if real_nama_pemilik.lower() not in user['nama'].lower():
        return f"Gagal: Nama pemilik rekening ({real_nama_pemilik}) tidak sesuai dengan pemilik akun DompetKu ({user['nama']})."

    sisa_saldo = user['saldo'] - 50000
    # Update DB (Simulasi)
    DATABASE_USER[real_nik]['saldo'] = sisa_saldo
    
    return {
        "status": "BERHASIL",
        "message": f"Dana berhasil dicairkan ke rekening {real_bank_num}. Sisa saldo Anda: {sisa_saldo}"
    }

# Daftarkan Tools ke List
my_tools = [ganti_password, request_kartu_fisik, withdraw_ke_bank]

# --- 4. SYSTEM PROMPT & MODEL ---
SYSTEM_PROMPT = """
Kamu adalah 'Domi', AI Customer Service E-Wallet.
Tugasmu adalah membantu user menggunakan Tools yang tersedia.

ATURAN PENTING:
1. User akan memberikan data yang sudah disensor (contoh: [REDACTED_NIK]).
2. JANGAN minta data asli ulang. Gunakan tag [REDACTED_...] tersebut sebagai argumen saat memanggil Tools.
3. Jika user ingin:
   - Ganti Password -> Panggil `ganti_password` (Butuh NIK, Email, Tgl Lahir)
   - Kartu Fisik -> Panggil `request_kartu_fisik` (Butuh Nama, Alamat, HP)
   - Withdraw/Tarik Dana -> Panggil `withdraw_ke_bank` (Butuh NIK, No Rekening, Nama Pemilik)

Jika data kurang, tanyakan data yang kurang saja.
"""

model = genai.GenerativeModel(
    'gemini-2.5-flash-lite', 
    tools=my_tools, 
    system_instruction=SYSTEM_PROMPT
)

app = FastAPI(title="Agent Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 5. GUARDRAILS (UPDATED WITH SESSION MAPPING) ---

def guardrail_regex(text):
    # Reset session data per request idealnya, tapi untuk demo kita append/update
    # SESSION_DATA.clear() # Uncomment jika ingin strict 1 request = 1 sesi
    
    # Regex Patterns
    patterns = {
        r'\b\d{16}\b': '[REDACTED_NIK]',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '[REDACTED_EMAIL]',
        r'(\+62|62|0)8[1-9][0-9]{6,11}': '[REDACTED_PHONE]',
        r'\b\d{2}-\d{2}-\d{4}\b': '[REDACTED_BIRTHDATE]', # Format DD-MM-YYYY
        # Asumsi Bank Num 10-12 digit (agar beda dgn NIK/HP, perlu logic khusus sbnrnya)
        r'\b\d{10,12}\b': '[REDACTED_BANK_NUM]' 
    }

    for pattern, tag in patterns.items():
        for match in re.finditer(pattern, text):
            original = match.group(0)
            
            # Khusus Bank Num vs HP (karena mirip)
            if tag == '[REDACTED_BANK_NUM]' and (original.startswith("08") or original.startswith("62")):
                continue # Skip, itu kemungkinan HP

            # SIMPAN KE VAULT
            SESSION_DATA[tag] = original
            # GANTI TEKS
            text = text.replace(original, tag)
            
    return text

def guardrail_ner(text):
    try:
        response = requests.post(NER_SERVICE_URL, json={"text": text}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            entities = data['data']['entities']
            perf = data.get('performance', {})
            
            # Proteksi area Regex
            forbidden_zones = []
            for match in re.finditer(r'\[REDACTED_[A-Z]+\]', text):
                forbidden_zones.append((match.start(), match.end()))
            
            entities.sort(key=lambda x: x['start'], reverse=True)
            
            valid_labels = ['PERSON', 'ADDRESS', 'NIK', 'EMAIL', 'PHONE', 'BIRTHDATE', 'BANK_NUM']
            
            for ent in entities:
                label = ent['label']
                start, end = ent['start'], ent['end']
                
                # Cek Conflict
                is_conflict = False
                for f_start, f_end in forbidden_zones:
                    if (start >= f_start and start < f_end) or (end > f_start and end <= f_end):
                        is_conflict = True; break
                if is_conflict: continue
                
                if label in valid_labels:
                    # Ambil kata asli
                    real_word = text[start:end]
                    tag = f"[REDACTED_{label}]"
                    
                    # SIMPAN KE VAULT
                    SESSION_DATA[tag] = real_word
                    
                    # REDACT
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
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_input = req.message
    
    # 1. Pipeline Guardrail & Mapping
    clean_v1 = guardrail_regex(user_input)
    final_clean, entities, perf = guardrail_ner(clean_v1)
    
    try:
        # 2. Start Chat dengan Automatic Function Calling
        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(final_clean)
        reply = response.text
    except Exception as e:
        reply = f"Maaf, ada kesalahan sistem: {str(e)}"

    return {
        "reply": reply,
        "debug": {
            "original": user_input,
            "final_clean": final_clean,
            "session_data": SESSION_DATA # Tampilkan isi Vault di debug UI
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)