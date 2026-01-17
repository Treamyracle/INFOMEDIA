import os
import time  # Untuk hitung latency
import psutil # Untuk hitung CPU & RAM
from fastapi import FastAPI, Body, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="NER Service - PII Guardrail (Private)")

# --- 1. KONFIGURASI ---
HF_TOKEN = os.getenv("HF_TOKEN")
model_name = "treamyracle/indobert-ner-pii-guardrail"

# [OPTIONAL] Hard force agar PyTorch tidak melihat GPU sama sekali
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

print(f"🔐 Sedang memuat model PRIVATE: {model_name}...")
print("⚙️  Mode: CPU Only")

try:
    # --- 2. LOAD PIPELINE (CPU FORCED) ---
    nlp_pipeline = pipeline(
        "ner", 
        model=model_name, 
        tokenizer=model_name, 
        aggregation_strategy="simple",
        token=HF_TOKEN,
        device=-1  # <--- [PENTING] -1 artinya CPU. 0 artinya GPU pertama.
    )
    print("✅ Model berhasil dimuat di CPU!")
except Exception as e:
    print(f"❌ Gagal memuat model! {e}")
    raise e

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_entities(request: TextRequest):
    text = request.text
    
    # [NEW] 1. Mulai Timer
    start_time = time.time()
    
    try:
        # Jalankan model
        results = nlp_pipeline(text)
        
        # [NEW] 2. Stop Timer & Hitung Latency
        end_time = time.time()
        
        # --- [TAMBAHAN: PRINT OUTPUT MODEL] ---
        print("\n" + "="*20 + " RAW MODEL OUTPUT " + "="*20)
        print(results)
        print("="*60 + "\n")
        # --------------------------------------
        
        latency_ms = (end_time - start_time) * 1000  # Konversi ke milisecond
        
        # [NEW] 3. Ambil Info Resource (CPU & RAM)
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024  # Convert Bytes to MB
        cpu_usage_percent = process.cpu_percent(interval=None) # CPU % sejak call terakhir
        
        # Filter hasil
        entities = []
        for item in results:
            entities.append({
                "text": item['word'],
                "label": item['entity_group'],
                "score": float(item['score']),
                "start": item['start'], 
                "end": item['end']
            })
            
        return {
            "status": "success",
            "data": {
                "original_text": text,
                "entities": entities
            },
            # [NEW] 4. Kirim Data Performance ke Client
            "performance": {
                "latency_ms": round(latency_ms, 2),
                "memory_mb": round(memory_usage_mb, 2),
                "cpu_percent": cpu_usage_percent
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)