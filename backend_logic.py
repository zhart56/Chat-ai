import requests
import json
import base64
import os
import re
import logging
from PIL import Image
from io import BytesIO
import gradio as gr  # TAMBAHKAN IMPORT INI
from duckduckgo_search import DDGS
from config import OPENROUTER_API_URL, DEEPSEEK_API_URL,  BLACKBOX_API_URL, global_settings, TTS_CACHE_DIR, DATA_DIR
from file_utils import add_to_memory, load_memory
import threading
import time
import multiprocessing
import easyocr
import cv2
import numpy as np
import uuid
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ... (kode lainnya tetap sama)

logger = logging.getLogger(__name__)

# Import library opsional dengan penanganan error

try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
    logger.info("Library google-genai (dengan Client) berhasil diimpor.")
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    logger.warning("Library 'google-genai' tidak terinstal. Fitur Google Gemini tidak akan berfungsi.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLOv8 (ultralytics) berhasil diimpor.")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics tidak terinstal. Fitur deteksi objek live tidak akan berfungsi.")

# Variabel global untuk model, agar tidak perlu dimuat berulang kali
yolo_model = None

try:
    import mss
    import numpy as np
    MSS_AVAILABLE = True
    logger.info("MSS (untuk tangkapan layar) berhasil diimpor.")
except ImportError:
    MSS_AVAILABLE = False
    logger.warning("MSS tidak terinstal. Fitur tangkapan layar tidak akan berfungsi.")

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV (untuk kamera) berhasil diimpor.")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV tidak terinstal. Fitur kamera tidak akan berfungsi.")

try:
    import pyvts
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    PYVTS_AVAILABLE = True
    logger.info("pyvts dan pydub berhasil diimpor.")
except ImportError:
    logger.warning("pyvts atau pydub tidak terinstal. Fitur VTube Studio tidak akan berfungsi.")
    pyvts, AudioSegment = None, None
    PYVTS_AVAILABLE = False

try:
    import pytchat
    PYTCHAT_AVAILABLE = True
    logger.info("pytchat berhasil diimpor.")
except ImportError:
    logger.warning("pytchat tidak terinstal. Fitur VTuber Live Chat tidak akan berfungsi.")
    PYTCHAT_AVAILABLE = False
    pytchat = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("EasyOCR berhasil diimpor")
except ImportError:
    logger.warning("EasyOCR tidak terinstal. Fitur OCR tidak akan berfungsi.")
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import pypdf
    PYPDF_AVAILABLE = True
    logger.info("PyPDF berhasil diimpor")
except ImportError:
    logger.warning("PyPDF tidak terinstal. Fitur PDF tidak akan berfungsi.")
    PYPDF_AVAILABLE = False
    pypdf = None

try:
    import docx
    DOCX_AVAILABLE = True
    logger.info("docx berhasil diimpor")
except ImportError:
    logger.warning("docx tidak terinstal. Fitur DOCX tidak akan berfungsi.")
    DOCX_AVAILABLE = False
    docx = None

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
    logger.info("DDGS berhasil diimpor")
except ImportError:
    logger.warning("DDGS tidak terinstal. Pencarian DuckDuckGo tidak akan berfungsi.")
    DDGS_AVAILABLE = False
    DDGS = None

# -- MODIFIKASI UNTUK DEEPSEEK & LM STUDIO --
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library berhasil diimpor")
except ImportError:
    logger.warning("Library 'openai' tidak terinstal. Fitur DeepSeek dan LM Studio tidak akan berfungsi.")
    OPENAI_AVAILABLE = False
    OpenAI = None

# -- AKHIR MODIFIKASI --

try:
    import speech_recognition as sr
    from gtts import gTTS
    TTS_AVAILABLE = True
    logger.info("SpeechRecognition dan gTTS berhasil diimpor")
except ImportError:
    logger.warning("Library 'gTTS' atau 'SpeechRecognition' tidak terinstal.")
    TTS_AVAILABLE = False
    sr, gTTS = None, None

try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
    logger.info("Google search berhasil diimpor")
except ImportError:
    logger.warning("Google search tidak terinstal.")
    GOOGLE_SEARCH_AVAILABLE = False
    google_search = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
    logger.info("YouTubeTranscriptApi berhasil diimpor")
except ImportError:
    logger.warning("YouTubeTranscriptApi tidak terinstal. Transkrip bawaan tidak akan berfungsi.")
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    # Kita definisikan class palsu agar kode tidak error saat memeriksa exception
    class YouTubeTranscriptApi: pass
    class TranscriptsDisabled: pass
    class NoTranscriptFound: pass

try:
    import whisper
    import yt_dlp
    WHISPER_AVAILABLE = True
    logger.info("Whisper dan yt-dlp berhasil diimpor")
except ImportError:
    logger.warning("Whisper atau yt-dlp tidak terinstal.")
    WHISPER_AVAILABLE = False
    whisper, yt_dlp = None, None

# PENINGKATAN: Tambahkan trafilatura untuk ekstraksi konten web yang lebih baik
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
    logger.info("Trafilatura berhasil diimpor")
except ImportError:
    logger.warning("Library 'trafilatura' tidak terinstal. Kualitas pembacaan web akan terbatas.")
    TRAFILATURA_AVAILABLE = False
    trafilatura = None

# Inisialisasi EasyOCR Reader
ocr_reader = None
if EASYOCR_AVAILABLE:
    try:
        logger.info("Menginisialisasi EasyOCR Reader...")
        ocr_reader = easyocr.Reader(['en', 'id'])
        logger.info("EasyOCR Reader siap.")
    except Exception as e:
        logger.error(f"Gagal menginisialisasi EasyOCR: {e}")
        ocr_reader = None

# Inisialisasi model Whisper
whisper_model = None
whisper_loaded = False

def load_whisper_model():
    """Memuat model Whisper jika belum ada."""
    global whisper_model, whisper_loaded
    
    if not WHISPER_AVAILABLE:
        logger.warning("Whisper tidak tersedia")
        return None
        
    if whisper_model is None and not whisper_loaded:
        try:
            logger.info("Mencoba memuat model Whisper (base)... Ini mungkin memakan waktu saat pertama kali.")
            whisper_model = whisper.load_model("base")
            whisper_loaded = True
            logger.info("Model Whisper berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat model Whisper: {e}")
            whisper_model = None
            
    return whisper_model

def get_ai_response(history, system_prompt):
    """
    Fungsi LAMA untuk mendapatkan respons penuh (non-streaming).
    Digunakan untuk fungsi internal seperti auto-memory atau ringkasan pencarian.
    """
    full_response = ""
    try:
        for chunk in get_ai_response_stream(history, system_prompt):
            if isinstance(chunk, str) and chunk.startswith("__ERROR__:"):
                # Jika terjadi error di fungsi internal, kembalikan sebagai string
                return chunk
            full_response += chunk
    except Exception as e:
        error_message = f"__ERROR__:Error selama pemrosesan internal non-streaming: {e}"
        logger.error(error_message)
        return error_message
    return full_response


def get_ai_response_stream(history, system_prompt):
    """
    Fungsi BARU yang menghasilkan respons AI secara streaming dengan struktur yang bersih.
    """
    provider = global_settings.get("api_provider", "OpenRouter")
    error_prefix = "__ERROR__:"
    
    MAKSIMAL_HISTORY = 20 

    if provider == "Google Gemini":
        if not GOOGLE_GENAI_AVAILABLE:
            yield f"{error_prefix}Library 'google-genai' belum terinstal. Jalankan: pip install -q -U google-genai"
            return
        
        api_key = global_settings.get("gemini_api_key", "")
        model_name = global_settings.get("model_name", "models/gemini-1.5-flash-latest")
        
        if not api_key:
            yield f"{error_prefix}API Key untuk Google Gemini belum diatur di Konfigurasi."
            return
        
        try:
            client = genai.Client(api_key=api_key)
            history_terbaru = history[-MAKSIMAL_HISTORY:]

            if 'gemma' in model_name.lower():
                contents = []
                is_first_user_message = True
                for msg in history_terbaru:
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    if role == "assistant" and not msg.get("content", "").strip():
                        continue
                    content_text = msg.get("content", "")
                    if role == "user" and is_first_user_message and system_prompt:
                        content_text = f"{system_prompt}\n\n---\n\n{content_text}"
                        is_first_user_message = False
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content_text)]))
                
                # --- PERBAIKAN: Tangani kasus history kosong untuk Gemma ---
                if not contents and system_prompt:
                    contents.append(types.Content(role='user', parts=[types.Part.from_text(text=system_prompt)]))
                # --- AKHIR PERBAIKAN ---
                
                response_stream = client.models.generate_content_stream(model=model_name, contents=contents)

            else:
                contents = []
                for msg in history_terbaru:
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    if role == "assistant" and not msg.get("content", "").strip():
                        continue
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.get("content", ""))]))

                config = types.GenerateContentConfig(system_instruction=system_prompt)
                
                # --- PERBAIKAN: Tangani kasus history kosong untuk Gemini ---
                # Jika history kosong, AI butuh pesan user untuk direspons, bahkan dengan system_instruction.
                if not contents and system_prompt:
                    contents.append(types.Content(role='user', parts=[types.Part.from_text(text=system_prompt)]))
                # --- AKHIR PERBAIKAN ---

                response_stream = client.models.generate_content_stream(model=model_name, contents=contents, config=config)

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_message = f"{error_prefix}Terjadi Error API (Google Gemini): {e}"
            logger.error(error_message, exc_info=True)
            yield error_message
        return

    # --- Sisa kode untuk provider lain tidak berubah ---
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    
    if history:
        messages.extend(history[-MAKSIMAL_HISTORY:])
        
    # --- PERBAIKAN KECIL: Pastikan messages tidak pernah kosong untuk provider lain ---
    if len(messages) <= 1 and not any(msg['role'] == 'user' for msg in messages):
        messages.append({"role": "user", "content": " "}) # Tambahkan dummy user message jika hanya ada system prompt
    # --- AKHIR PERBAIKAN KECIL ---

    try:
        if provider == "DeepSeek":
            if not OPENAI_AVAILABLE:
                raise ImportError("Library OpenAI (diperlukan untuk DeepSeek) tidak terinstal.")
            api_key = global_settings.get("deepseek_api_key", "")
            if not api_key: raise ValueError("API Key untuk DeepSeek belum diatur.")
            
            client = OpenAI(api_key=api_key, base_url=DEEPSEEK_API_URL)
            response_stream = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True, timeout=30)
            for chunk in response_stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "LM Studio":
            if not OPENAI_AVAILABLE:
                raise ImportError("Library OpenAI (diperlukan untuk LM Studio) tidak terinstal.")
            base_url = global_settings.get("lmstudio_api_url", "")
            if not base_url: raise ValueError("URL API untuk LM Studio belum diatur.")
            
            client = OpenAI(api_key="lm-studio", base_url=base_url)
            response_stream = client.chat.completions.create(model="local-model", messages=messages, stream=True, timeout=30)
            for chunk in response_stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "Blackbox AI":
            api_key = global_settings.get("blackbox_api_key", "")
            if not api_key: raise ValueError("API Key untuk Blackbox AI belum diatur.")
            model = global_settings.get("blackbox_model_name", "blackboxai/openai/gpt-4")
            
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "stream": True}

            with requests.post(BLACKBOX_API_URL, headers=headers, json=payload, stream=True, timeout=180) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line and line.decode('utf-8').startswith("data: "):
                        json_str = line.decode('utf-8')[6:]
                        if json_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(json_str)
                            content = data['choices'][0].get('delta', {}).get('content')
                            if content: yield content
                        except (json.JSONDecodeError, IndexError, KeyError): continue
        
        else:  # Default ke OpenRouter
            api_key = global_settings.get("openrouter_api_key", "")
            if not api_key: raise ValueError("API Key untuk OpenRouter belum diatur.")
            model = global_settings.get("model_name", "google/gemini-flash-1.5")
            
            headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
            payload = { "model": model, "messages": messages, "stream": True }
            if history and isinstance(history[-1].get("content"), list):
                payload["max_tokens"] = 4096
            
            with requests.post(OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=180) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line and line.decode('utf-8').startswith("data: "):
                        json_str = line.decode('utf-8')[6:]
                        if json_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(json_str)
                            content = data['choices'][0].get('delta', {}).get('content')
                            if content: yield content
                        except (json.JSONDecodeError, IndexError, KeyError): continue

    except Exception as e:
        error_message = f"{error_prefix}Terjadi Error API ({provider}): {e}"
        logger.error(error_message, exc_info=True)
        yield error_message

def speech_to_text(audio_filepath):
    """Mengubah file audio menjadi teks menggunakan Google Speech Recognition."""
    if not TTS_AVAILABLE:
        return "Error: Library SpeechRecognition tidak terinstal."
    
    if not audio_filepath:
        return ""
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_filepath) as source:
            audio_data = recognizer.record(source)
            logger.info("Menerjemahkan audio ke teks...")
            text = recognizer.recognize_google(audio_data, language='id-ID')
            logger.info(f"Teks yang dikenali: {text}")
            return text
    except sr.UnknownValueError:
        return "Maaf, saya tidak bisa memahami audio tersebut."
    except sr.RequestError as e:
        return f"Error koneksi ke layanan Google Speech; {e}"
    except Exception as e:
        logger.error(f"Error speech_to_text: {e}")
        return f"Terjadi error saat pemrosesan audio: {e}"

def clean_text_for_tts(text):
    """Membersihkan teks dari emoji dan simbol yang tidak diinginkan sebelum diubah menjadi suara."""
    if not text:
        return ""
    
    # Hapus emoji
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Hapus simbol khusus
    symbol_pattern = r'[\*\`~@#$%\^&()_+=\|\\{}\[\];:\"<>/]'
    text = re.sub(symbol_pattern, '', text)
    
    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def text_to_speech(text):
    """Mengubah teks menjadi file audio MP3 menggunakan gTTS."""
    if not TTS_AVAILABLE:
        logger.error("Library gTTS tidak terinstal.")
        return None
    
    cleaned_text = clean_text_for_tts(text)
    if not cleaned_text:
        logger.warning("Teks kosong setelah dibersihkan, tidak ada audio yang dibuat.")
        return None
        
    try:
        logger.info(f"Mengubah teks bersih menjadi suara: '{cleaned_text[:50]}...'")
        tts = gTTS(cleaned_text, lang='id')
        filepath = os.path.join(TTS_CACHE_DIR, "response.mp3")
        tts.save(filepath)
        logger.info(f"File audio disimpan di: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Gagal membuat file audio: {e}")
        return None

def build_prompt_with_memory(system_prompt, memory_content):
    """Menggabungkan system prompt dengan konten dari memori."""
    if memory_content and memory_content.strip():
        return f"{system_prompt}\n\n--- FAKTA PENTING DARI MEMORI (INGAT INI SELALU) ---\n{memory_content}\n--- AKHIR DARI MEMORI ---"
    return system_prompt

def build_roleplay_prompt(ai_persona, user_persona, pov_mode, memory_content):
    """Membangun system prompt untuk mode roleplay."""
    pov_instructions = {
        # ... (isi pov_instructions tetap sama)
    }
    
    # DAFTAR EKSPRESI YANG SESUAI DENGAN HOTKEY DI VTS
    EMOTION_LIST = ["senang", "sedih", "marah", "kaget", "netral"]

    json_instruction = f"""
---
ATURAN OUTPUT PENTING (UNTUK AI):
- Anda HARUS selalu merespons dalam format JSON yang valid.
- JSON harus berisi dua kunci (key): "emotion" dan "response".
- Untuk kunci "emotion", pilih SATU emosi yang paling sesuai dari daftar berikut: {str(EMOTION_LIST)}. Emosi ini harus mencerminkan isi dari respons Anda.
- Untuk kunci "response", isi dengan teks balasan Anda seperti biasa, mengikuti aturan sudut pandang di atas.

CONTOH FORMAT JSON:
{{
    "emotion": "senang",
    "response": "Tentu saja aku mau! Ayo kita pergi berpetualang bersama!"
}}
---
"""

    full_persona = f"--- DESKRIPSI KARAKTER AI (PERAN ANDA) ---\n{ai_persona}\n\n--- DESKRIPSI KARAKTER PENGGUNA (PERAN SAYA) ---\n{user_persona}"
    pov_instruction = pov_instructions.get(pov_mode, "")
    
    # Gabungkan semua instruksi
    final_prompt = json_instruction + "\n" + full_persona + "\n" + pov_instruction

    return build_prompt_with_memory(final_prompt, memory_content), full_persona

# Di dalam backend_logic.py

def run_ocr_in_process(image_bytes, queue):
    """
    Fungsi ini berjalan di proses terpisah yang bersih.
    Ia menerima DATA GAMBAR (bytes) dan menjalankan OCR.
    """
    import easyocr 
    
    try:
        # --- MULAI BLOK PRE-PROCESSING ---
        # Ubah data bytes menjadi gambar OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Ubah ke grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 2. Terapkan thresholding untuk mendapatkan gambar hitam-putih yang tajam
        _, processed_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Opsi: Jika ingin mengirim kembali gambar yang diproses (untuk debug)
        # _, buffer = cv2.imencode('.png', processed_img)
        # image_bytes_processed = buffer.tobytes()
        # --- AKHIR BLOK PRE-PROCESSING ---

        reader = easyocr.Reader(['en', 'id'], gpu=False)
        
        # easyocr sekarang membaca gambar yang sudah bersih
        results = reader.readtext(processed_img, detail=1, paragraph=False)
        
        queue.put(results)
    except Exception as e:
        queue.put(f"__ERROR__:{e}")

def extract_text_from_file(file_obj):
    """Mengekstrak teks dari berbagai tipe file (PDF, DOCX, TXT, Gambar/OCR)."""
    if file_obj is None:
        return None, "Tidak ada file yang diunggah."
    
    file_path = file_obj.name
    filename = os.path.basename(file_path)
    
    try:
        if filename.lower().endswith('.pdf'):
            reader = pypdf.PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages])
            return text, f"Teks dari '{filename}' (PDF)"
            
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text, f"Teks dari '{filename}' (DOCX)"
            
        elif filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                return text, f"Teks dari '{filename}' (TXT)"
                
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            # --- PERUBAHAN UTAMA: BACA FILE KE MEMORI DULU ---
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            # --- AKHIR PERUBAHAN ---

            q = multiprocessing.Queue()
            
            # Berikan image_bytes, bukan file_path
            p = multiprocessing.Process(target=run_ocr_in_process, args=(image_bytes, q))
            p.start()
            
            results = q.get(timeout=60)
            p.join()

            if isinstance(results, str) and results.startswith("__ERROR__:"):
                raise Exception(results.replace("__ERROR__:", ""))

            if not results:
                return "", f"Tidak ada teks yang terdeteksi dari gambar '{filename}' (OCR)"

            # Proses hasil dengan logika stabil dari file lama Anda
            avg_char_height = sum([res[0][2][1] - res[0][0][1] for res in results]) / len(results) if results else 0
            results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            
            output_text = []
            current_line = []
            last_y = -1
            
            for bbox, text, conf in results:
                top_y = bbox[0][1]
                if last_y == -1:
                    last_y = top_y
                
                if top_y > last_y + (avg_char_height * 0.7):
                    if current_line:
                        current_line.sort(key=lambda item: item[0])
                        output_text.append(" ".join([item[1] for item in current_line]))
                    current_line = []
                    last_y = top_y
                
                left_x = bbox[0][0]
                current_line.append((left_x, text))
            
            if current_line:
                current_line.sort(key=lambda item: item[0])
                output_text.append(" ".join([item[1] for item in current_line]))
            
            return "\n".join(output_text), f"Teks dari gambar '{filename}' (OCR dengan Tata Letak)"

        else:
            return None, "Tipe file tidak didukung."
            
    except Exception as e:
        # logger.error(f"Error extract_text_from_file: {e}")
        return None, f"Gagal membaca file: {e}"

def image_to_base64(pil_image):
    """Mengubah objek gambar PIL menjadi string base64."""
    try:
        # --- PERBAIKAN DI SINI ---
        # Periksa apakah mode gambar adalah RGBA (memiliki transparansi)
        if pil_image.mode == 'RGBA':
            # Ubah menjadi RGB, ini akan menghapus channel Alpha (transparansi)
            pil_image = pil_image.convert('RGB')
        # --- AKHIR PERBAIKAN ---

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG") # Sekarang ini akan selalu berhasil
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error image_to_base64: {e}")
        return None

def process_chat(user_input, history, system_prompt, memory_content=""):
    """Generator untuk memproses chat standar (NON-STREAMING)."""
    if not user_input or not user_input.strip():
        yield history, ""
        return
        
    # Tambahkan input pengguna ke history
    new_history = history + [{"role": "user", "content": user_input}]
    yield new_history, ""
    
    # Dapatkan respons AI
    final_system_prompt = build_prompt_with_memory(system_prompt, memory_content)
    ai_response = get_ai_response(new_history, final_system_prompt)
    
    # Tambahkan respons AI ke history
    final_history = new_history + [{"role": "assistant", "content": ai_response}]
    yield final_history, ""

def get_button_updates_for_reset():
    """Mengembalikan update Gradio untuk mereset tombol file."""
    return (
        gr.update(visible=True), 
        gr.update(visible=False), 
        gr.update(visible=False), 
        gr.update(visible=False)
    )

def process_chat_with_file(user_input, history, file_obj, mode, system_prompt, memory_content=""):
    """
    Generator untuk memproses chat dengan file (Vision, OCR, Doc).
    Versi ini membersihkan komponen File setelah selesai untuk mencegah error.
    """
    if (not user_input or not user_input.strip()) and not file_obj:
        yield history, "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    final_system_prompt = build_prompt_with_memory(system_prompt, memory_content)
    btn_reset_updates = get_button_updates_for_reset()
    
    # --- BLOK LOGIKA UNTUK MODE VISION ---
    if mode == "vision":
        try:
            user_message = {"role": "user", "content": (user_input or "Jelaskan gambar ini", file_obj.name)}
            ai_placeholder = {"role": "assistant", "content": ""}
            history_for_display = history + [user_message, ai_placeholder]

            pil_image = Image.open(file_obj.name)
            base64_image = image_to_base64(pil_image)
            if not base64_image:
                raise Exception("Gagal mengonversi gambar ke base64.")

            multimodal_content = [{"type": "text", "text": user_input or "Jelaskan gambar ini"}, {"type": "image_url", "image_url": {"url": base64_image}}]
            history_for_api = history + [{"role": "user", "content": multimodal_content}]

            yield (history_for_display, "", None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))

            vision_prompt = final_system_prompt or "Anda adalah AI Vision yang ahli menganalisis gambar."
            ai_response = get_ai_response(history_for_api, vision_prompt)

            history_for_display[-1]["content"] = ai_response
            
            # --- PERBAIKAN DI SINI ---
            # Kirim gr.update(value=None) untuk membersihkan komponen file
            yield history_for_display, "", gr.update(value=None), *btn_reset_updates
            return

        except Exception as e:
            logger.error(f"Error processing vision: {e}")
            yield history, f"Error memproses gambar: {e}", gr.update(value=None), *btn_reset_updates
            return

    # --- BLOK LOGIKA UNTUK MODE OCR & DOC ---
    elif mode in ["ocr", "doc"]:
        try:
            extracted_text, status_msg = extract_text_from_file(file_obj)
            if extracted_text is None:
                raise Exception(status_msg)

            user_prompt_with_context = (f"Berdasarkan konteks dari file '{os.path.basename(file_obj.name)}' ini:\n--- AWAL TEKS ---\n{extracted_text}\n--- AKHIR TEKS ---\n\n" f"Jawab pertanyaan berikut: '{user_input}'")
            new_history = history + [{"role": "user", "content": user_prompt_with_context}, {"role": "assistant", "content": ""}]
            
            yield (new_history, "", None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
            
            doc_prompt = final_system_prompt or "Anda adalah asisten AI cerdas. Jawab berdasarkan teks dari file."
            ai_response = get_ai_response(new_history[:-1], doc_prompt)

            new_history[-1]["content"] = ai_response
            
            # --- PERBAIKAN DI SINI ---
            # Kirim gr.update(value=None) untuk membersihkan komponen file
            yield new_history, "", gr.update(value=None), *btn_reset_updates
            return
            
        except Exception as e:
            logger.error(f"Error processing doc/ocr: {e}")
            yield history, f"Gagal memproses file: {e}", gr.update(value=None), *btn_reset_updates
            return

    # --- BLOK JIKA MODE TIDAK VALID ---
    else:
        yield history, "Error: Mode tidak valid.", gr.update(value=None), *btn_reset_updates
        return

def get_youtube_transcript_whisper(url):
    """Mendapatkan transkrip dari audio YouTube menggunakan Whisper dengan penanganan file yang lebih andal."""
    model = load_whisper_model()
    if not model:
        return "Error: Model Whisper tidak berhasil dimuat.", None

    # Buat nama file sementara, ini hanya sebagai template
    unique_id = uuid.uuid4()
    audio_template = os.path.join(TTS_CACHE_DIR, f"youtube_audio_{unique_id}.%(ext)s")
    
    # Path file audio final akan kita dapatkan nanti
    final_audio_filepath = None

    try:
        logger.info(f"Mencoba mengunduh audio dari {url}...")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192', # <-- KUALITAS AUDIO DITINGKATKAN
            }],
            'outtmpl': audio_template, # Gunakan template, bukan nama file statis
            'noplaylist': True, 'quiet': True, 'no_warnings': True, 'retries': 5,
            'fragment_retries': 5,
            'extractor_args': {'youtube': {'player_client': 'android'}},
            'http_headers': {'User-Agent': 'Mozilla/5.0'}
        }

        # --- PERBAIKAN UTAMA DI SINI ---
        # Gunakan extract_info untuk mengunduh sekaligus mendapatkan nama file yang benar
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            # Dapatkan path file yang sebenarnya setelah diunduh dan diproses
            final_audio_filepath = info_dict.get('requested_downloads', [{}])[0].get('filepath')

        if not final_audio_filepath or not os.path.exists(final_audio_filepath):
            # Pesan error ini sekarang lebih akurat
            return "Gagal mendapatkan path file audio setelah diunduh.", None
        # --- AKHIR PERBAIKAN ---

        logger.info(f"Mentranskripsi audio dari file: {os.path.basename(final_audio_filepath)}...")
        result = model.transcribe(final_audio_filepath)
        logger.info("Transkripsi selesai.")
        
        return "Berhasil mentranskripsi audio dari video.", result.get('text', '')

    except Exception as e:
        logger.error(f"Error get_youtube_transcript_whisper: {e}")
        return f"Gagal memproses audio YouTube: {e}", None
        
    finally:
        # Selalu bersihkan file yang sebenarnya diunduh
        if final_audio_filepath and os.path.exists(final_audio_filepath):
            try:
                os.remove(final_audio_filepath)
                logger.info(f"File audio temporer {os.path.basename(final_audio_filepath)} dihapus.")
            except Exception as e:
                logger.error(f"Gagal menghapus file audio temporer: {e}")

def get_youtube_transcript(url):
    """Mendapatkan transkrip dari URL YouTube (mencoba API dulu, lalu Whisper)."""
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return "URL YouTube tidak valid.", None

    video_id = video_id_match.group(1)

    # Coba dapatkan transkrip bawaan terlebih dahulu
    if YOUTUBE_TRANSCRIPT_AVAILABLE:
        try:
            logger.info(f"Mencari transkrip bawaan untuk video ID: {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['id', 'en'])
            text_chunks = [chunk['text'] for chunk in transcript_list]
            return "Berhasil mendapatkan transkrip bawaan.", " ".join(text_chunks)
        except (TranscriptsDisabled, NoTranscriptFound):
            logger.warning(f"Video ID: {video_id} tidak memiliki transkrip bawaan. Beralih ke Whisper.")
        except Exception as e:
            logger.warning(f"Terjadi error saat mengambil transkrip bawaan: {e}. Beralih ke Whisper.")

    # Fallback ke Whisper jika transkrip bawaan gagal atau tidak tersedia
    logger.info("Tidak ada transkrip bawaan, menggunakan Whisper sebagai fallback...")
    return get_youtube_transcript_whisper(url)

def get_web_content(url):
    """
    Mengambil konten utama dari URL web dengan penanganan error yang lebih baik.
    """
    logger.info(f"Memulai get_web_content untuk URL: {url}")
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
        'Sec-Ch-Ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    try:

        if url.lower().endswith('.pdf'):
            logger.info(f"Mendeteksi URL PDF: {url}. Memulai unduhan...")
            response = requests.get(url, headers=headers, timeout=60, verify=False)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            reader = pypdf.PdfReader(pdf_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if text:
                logger.info(f"Berhasil mengekstrak teks dari PDF: {url}")
                return text[:20000]
            else:
                logger.warning(f"Gagal mengekstrak teks dari PDF, meskipun berhasil diunduh: {url}")
                return None

        response = requests.get(url, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        downloaded = response.text

        if TRAFILATURA_AVAILABLE and downloaded:
            main_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, output_format='txt')
            if main_text and len(main_text) > 150:
                logger.info(f"Berhasil mengekstrak konten dari {url} menggunakan trafilatura.")
                return main_text[:20000]

        logger.info(f"Trafilatura gagal, menggunakan fallback BeautifulSoup untuk {url}.")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(downloaded, 'lxml')
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text[:8000]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error get_web_content untuk {url} : {e}")
        return None

def perform_web_search(query, num_results, engine="DuckDuckGo"):
    """Melakukan pencarian web menggunakan DuckDuckGo atau Google."""
    try:
        if engine == "Google":
            api_key = global_settings.get("google_api_key")
            cse_id = global_settings.get("google_cse_id")

            if not api_key or not cse_id:
                return "Error: Google API Key atau Search Engine ID belum diatur di Konfigurasi.", []

            logger.info(f"Mencari di Google API untuk: '{query}'...")
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': api_key,
                    'cx': cse_id,
                    'q': query,
                    'num': num_results
                }
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                search_results = response.json()
            
                formatted_results = []
                if 'items' in search_results:
                    for item in search_results['items']:
                        formatted_results.append({
                            "href": item.get('link'),
                            "title": item.get('title'),
                            "body": item.get('snippet')
                        })
                return "Hasil pencarian dari Google API.", formatted_results
            except Exception as e:
                logger.error(f"Gagal melakukan pencarian Google API: {e}")
                return f"Gagal melakukan pencarian Google API: {e}", []
            
        else:  # DuckDuckGo
            if not DDGS_AVAILABLE:
                return "Error: Library 'duckduckgo-search' tidak terinstal.", []
                
            logger.info(f"Mencari di DuckDuckGo untuk: '{query}'...")
            text_results = DDGS().text(keywords=query, max_results=num_results)
            
            # Format hasil dengan URL lengkap
            formatted_results = []
            for res in text_results:
                if 'href' in res and res['href']:
                    # Pastikan URL memiliki scheme
                    url = res['href']
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    formatted_results.append({
                        "href": url,
                        "title": res.get('title', 'Tanpa Judul'),
                        "body": res.get('body', 'Konten akan dibaca dari link.')
                    })
            
            return "Hasil pencarian dari DuckDuckGo.", formatted_results if formatted_results else []
            
    except Exception as e:
        if "HTTP Error 429" in str(e):
            return "Error: Terlalu banyak permintaan. Silakan coba lagi nanti.", []
        logger.error(f"Gagal melakukan pencarian: {e}")
        return f"Gagal melakukan pencarian: {e}", []

def process_user_web_search(user_input, history, search_engine, memory_content=""):
    """
    Generator untuk memproses alur pencarian web, dari kueri hingga ringkasan.
    Menggunakan logika multi-kueri dari versi lama dengan perbaikan streaming dan fitur modern.
    """
    if not user_input or not user_input.strip():
        yield history, ""
        return

    new_history = history + [{"role": "user", "content": user_input}]

    new_history.append({"role": "assistant", "content": "🤔 Menganalisis permintaan..."})
    yield new_history, ""

    intent_prompt = f"""
Apakah permintaan pengguna berikut secara spesifik meminta untuk dicarikan sebuah VIDEO?
Jawab HANYA dengan 'YA' atau 'TIDAK'.

Permintaan Pengguna: "{user_input}"
Jawaban:
"""
    intent = get_ai_response([], intent_prompt).strip().upper()
    
    if "YA" in intent or "YOUTUBE.COM" in user_input.upper() or "YOUTU.BE" in user_input.upper():
        
        video_url = None
        # Jika pengguna sudah memberi URL, langsung pakai
        if "YOUTUBE.COM" in user_input.upper() or "YOUTU.BE" in user_input.upper():
            video_url = user_input
        # Jika belum, cari videonya
        else:
            new_history[-1] = {"role": "assistant", "content": f"🔍 Baik, saya carikan video tentang '{user_input}' di YouTube..."}
            yield new_history, ""
            
            # Buat kueri pencarian khusus untuk YouTube
            youtube_query = f'"{user_input}" site:youtube.com'
            _, search_results = perform_web_search(youtube_query, 1, engine=search_engine) # Cari 1 hasil teratas
            
            if search_results and "youtube.com" in search_results[0].get('href', ''):
                video_url = search_results[0]['href']
                new_history.append({"role": "assistant", "content": f"✅ Video ditemukan: {search_results[0]['title']}"})
                yield new_history, ""
            else:
                new_history.pop()
                new_history.append({"role": "assistant", "content": "Maaf, saya tidak berhasil menemukan video yang relevan di YouTube."})
                yield new_history, ""
                return
            
        new_history.append({"role": "assistant", "content": "🔄 Memproses transkrip video..."})
        yield new_history, ""

        status, transcript = get_youtube_transcript(video_url)
        if transcript:
            prompt = f"Anda adalah AI pereview. Buat ringkasan dari transkrip video ini:\n\n---\n{transcript}\n---\n\nSertakan juga sumbernya: {video_url}"
            new_history.pop() # Hapus status
            new_history.append({"role": "assistant", "content": ""}) # Bubble kosong untuk diisi
            stream = get_ai_response_stream([], prompt)
            full_response = ""
            for chunk in stream:
                full_response += chunk
                new_history[-1]['content'] = full_response
                yield new_history, ""
            return
        else:
            new_history.pop()
            new_history.append({"role": "assistant", "content": f"Gagal mendapatkan transkrip: {status}"})
            yield new_history, ""
            return
    else:
        new_history[-1] = {"role": "assistant", "content": "1. 🔄 Memformulasikan kueri pencarian artikel..."}
        yield new_history, ""

    query_expansion_prompt = f"""
Anda adalah seorang Research Assistant AI yang sangat cerdas. Tugas Anda adalah menganalisis permintaan pengguna dan menghasilkan 3 kueri pencarian web yang beragam dan efektif.
Permintaan Pengguna: "{user_input}"
Instruksi:
1.  **Kueri 1 (Spesifik):** Reformulasi langsung permintaan pengguna.
2.  **Kueri 2 (Konseptual):** Perluas topik dengan mencari konsep yang lebih luas.
3.  **Kueri 3 (Alternatif):** Ambil sudut pandang yang berbeda.
Format output Anda HANYA berupa daftar kueri, satu kueri per baris. JANGAN tambahkan nomor atau bullet point.
"""
    expanded_queries_str = get_ai_response([], query_expansion_prompt)
    if expanded_queries_str.startswith("__ERROR__:"):
        new_history.pop()
        new_history.append({"role": "assistant", "content": f"Gagal membuat kueri: {expanded_queries_str}"})
        yield new_history, ""
        return

    queries = [q.strip() for q in expanded_queries_str.split('\n') if q.strip()]
    if not queries: queries = [user_input]
    
    new_history[-1] = {"role": "assistant", "content": f"2. ✅ Selesai membuat {len(queries)} kueri pencarian."}
    yield new_history, ""

    # 2. MELAKUKAN PENCARIAN DAN MEMBACA ARTIKEL
    all_articles_content = []
    source_list = []
    
    # MENGGUNAKAN PENGATURAN DARI KONFIGURASI (FITUR BARU)
    num_results_per_query = global_settings.get("search_results_count", 3)

    new_history.append({"role": "assistant", "content": "3. 🔍 Memulai pencarian di internet..."})
    yield new_history, ""

    for i, query in enumerate(queries):
        new_history[-1] = {"role": "assistant", "content": f"   - Mencari kueri {i+1}/{len(queries)}: '{query}'"}
        yield new_history, ""
        
        _, search_results = perform_web_search(query, num_results_per_query, engine=search_engine)
        if not search_results:
            continue

        for j, res in enumerate(search_results):
            url = res.get('href', '')
            title = res.get('title', 'Tanpa Judul')
            if not url or url in [s.split(' - ')[-1] for s in source_list]: continue

            new_history.append({"role": "assistant", "content": f"      - 📰 Membaca '{title}'..."})
            yield new_history, ""
            
            content = get_web_content(url)
            if content and len(content) > 150:
                all_articles_content.append(f"--- SUMBER: {url} ---\n{content}\n--- AKHIR SUMBER ---")
                source_list.append(f"{len(source_list)+1}. {title} - {url}")

    if not all_articles_content:
        new_history.pop()
        new_history.append({"role": "assistant", "content": "Maaf, saya tidak berhasil mendapatkan konten yang relevan dari hasil pencarian."})
        yield new_history, ""
        return

    # 3. MEMBUAT RANGKUMAN
    new_history.append({"role": "assistant", "content": f"4. 👍 Selesai membaca {len(source_list)} artikel. Menyiapkan rangkuman..."})
    yield new_history, ""
    
    context = "\n\n".join(all_articles_content)
    search_summary = f"Berikut adalah sintesis informasi dari {len(source_list)} sumber yang berhasil dibaca:"
    system_prompt_for_summary = (
        f"Anda adalah seorang Asisten AI ahli yang informatif. Tugas Anda adalah memberikan jawaban terbaik dan paling komprehensif atas pertanyaan pengguna.\n\n"
        f"Sebagai referensi utama, saya telah mengumpulkan beberapa teks dari internet di bawah ini. Gunakan informasi dari teks ini sebagai dasar jawaban Anda.\n\n"
        f"Anda juga **Boleh** menggunakan pengetahuan umum Anda sendiri untuk melengkapi atau memperkaya jawaban jika relevan dan akurat.\n\n"
        f"Gabungkan informasi dari pencarian web dan pengetahuan Anda untuk menyusun jawaban yang lengkap dan mudah dipahami.\n\n"
        f"Pertanyaan Pengguna: '{user_input}'\n\n"
        f"--- KONTEKS DARI PENCARIAN WEB ---\n"
        f"{context}\n"
        f"--- AKHIR KONTEKS ---\n\n"
        f"Jawaban Komprehensif Anda:"
    )
    final_system_prompt = build_prompt_with_memory(system_prompt_for_summary, memory_content)

    # MENGGUNAKAN STREAMING UNTUK MENCEGAH MACET (PERBAIKAN PENTING)
    new_history.pop()
    new_history.append({"role": "assistant", "content": ""})
    stream = get_ai_response_stream([], final_system_prompt)
    full_response = ""
    for chunk in stream:
        if isinstance(chunk, str) and not chunk.startswith("__ERROR__:"):
            full_response += chunk
            new_history[-1]['content'] = full_response
            yield new_history, ""

    if source_list:
        formatted_sources = "\n\n\n**Sumber Informasi:**\n" + "\n".join(source_list)
        full_response += formatted_sources
        new_history[-1]['content'] = full_response
        yield new_history, ""

def update_memory_automatically(history, directory, memory_file_name):
    """
    Menganalisis percakapan terakhir dan secara otomatis menyimpannya ke memori.
    Versi ini sudah sinkron dengan fungsi memori yang baru.
    """
    # Jangan lakukan apa-apa jika tidak ada file memori yang dipilih
    if not memory_file_name:
        # Kembalikan konten memori yang ada (jika ada) atau string kosong
        return load_memory(directory, memory_file_name)

    logger.info(f"Memulai pembaruan memori otomatis untuk: {memory_file_name}")
    if len(history) < 2:
        return load_memory(directory, memory_file_name)

    last_user_turn = history[-2]['content']
    last_ai_turn = history[-1]['content']

    # Menangani input multimodal (gambar + teks)
    if isinstance(last_user_turn, tuple):
        last_user_turn = last_user_turn[0] # Ambil bagian teksnya saja
    elif isinstance(last_user_turn, list):
        text_parts = [item.get('text', '') for item in last_user_turn if isinstance(item, dict) and item.get('type') == 'text']
        last_user_turn = " ".join(text_parts)

    memory_prompt = f"""
Anda adalah asisten pencatat yang cerdas. Tugas Anda adalah membaca percakapan berikut dan mengekstrak FAKTA KUNCI yang baru dan penting untuk diingat di masa depan.

CONTOH:
- Jika pengguna menyebutkan nama atau preferensi mereka ("nama saya Budi", "saya suka kopi").
- Jika sebuah keputusan penting dibuat dalam roleplay.
- Jika sebuah informasi baru yang signifikan terungkap.

Tulis fakta-fakta tersebut sebagai daftar poin singkat. Jika tidak ada fakta baru yang penting, balas dengan kata 'None'.

PERCAKAPAN TERAKHIR:
Pengguna: "{last_user_turn}"
Asisten: "{last_ai_turn}"

FAKTA BARU UNTUK DIINGAT:
"""
    facts_to_remember = get_ai_response([], memory_prompt)

    if facts_to_remember and "none" not in facts_to_remember.lower():
        logger.info(f"Fakta baru ditemukan: \n{facts_to_remember}")
        # Panggil add_to_memory dengan 3 argumen yang benar
        add_to_memory(directory, memory_file_name, facts_to_remember)
    else:
        logger.info("Tidak ada fakta baru yang signifikan untuk diingat.")

    # Panggil load_memory dengan 2 argumen yang benar
    return load_memory(directory, memory_file_name)


def live_chat_listener(video_id, stop_event, message_queue):
    """
    Fungsi ini berjalan di thread terpisah untuk mengambil pesan live chat.
    'stop_event' adalah objek threading.Event untuk menghentikan loop.
    'message_queue' adalah list sederhana untuk menampung pesan baru.
    """
    if not PYTCHAT_AVAILABLE:
        logger.error("Pytchat tidak tersedia.")
        return

    try:
        logger.info(f"Memulai listener untuk video ID: {video_id}")
        chat = pytchat.create(video_id=video_id)
        while chat.is_alive() and not stop_event.is_set():
            for c in chat.get().items:
                # Tambahkan pesan ke antrian dalam format yang kita inginkan
                message_queue.append(f"{c.author.name}: {c.message}")
                logger.info(f"Pesan baru: {c.author.name}: {c.message}")
            # Beri jeda sedikit agar tidak membebani CPU
            time.sleep(2)
        logger.info("Listener dihentikan.")
    except Exception as e:
        logger.error(f"Error di live_chat_listener: {e}")
        message_queue.append(f"__ERROR__: Gagal terhubung atau stream berakhir. Error: {e}")

def process_chat_for_vtuber(new_messages, history, ai_persona, user_persona, memory_content=""):
    """
    Menganalisis pesan baru dan memutuskan apakah AI harus merespons.
    """
    # Gabungkan semua pesan baru menjadi satu konteks
    context = "\n".join(new_messages)

    # Buat prompt khusus untuk AI
    prompt = f"""
    Anda adalah seorang VTuber AI. Persona Anda adalah:
    --- PERSONA ANDA (AI) ---
    {ai_persona}
    ---

    Peran streamer/moderator (saya) adalah:
    --- PERAN SAYA (STREAMER) ---
    {user_persona}
    ---

    Berikut adalah beberapa pesan terbaru dari penonton di live chat.
    Tugas Anda adalah membaca pesan-pesan ini dan memberikan satu respons yang menarik, relevan, dan sesuai dengan persona Anda. Anda bisa menyapa beberapa penonton atau merespons satu topik yang menarik.
    JANGAN merespons setiap pesan satu per satu. Buatlah respons yang terasa alami seolah-olah Anda sedang live streaming.

    PESAN-PESAN DARI PENONTON:
    {context}

    RESPONS ANDA SEBAGAI VTUBER:
    """

    # Dapatkan respons AI (non-streaming untuk kasus ini lebih mudah)
    response_text = get_ai_response(history, prompt)

    # Jika ada error, kembalikan error tersebut
    if response_text.startswith("__ERROR__:"):
        return response_text, None
    
    filtered_response_text = apply_content_filter(response_text, memory_content)

    # Ubah teks respons menjadi suara
    audio_path = text_to_speech(response_text)

    return filtered_response_text, audio_path

# --- FUNGSI BARU UNTUK VTube Studio ---

# Variabel global untuk menyimpan koneksi VTS
vts_plugin = None
vts_connected = False

async def initialize_vts_plugin():
    """
    Menginisialisasi dan mengautentikasi plugin VTube Studio dengan alur
    API level tinggi dan nama fungsi pengecekan status yang benar.
    """
    global vts_plugin, vts_connected
    await disconnect_vts_plugin()

    if not PYVTS_AVAILABLE:
        return "Error: Library pyvts tidak terinstal.", None

    try:
        logger.info("Langkah 1: Menginisialisasi objek plugin pyvts...")
        token_path = os.path.join(DATA_DIR, "vts_token.txt")
        plugin_info = {
            "plugin_name": "AI VTuber Companion",
            "developer": "YourName",
            "authentication_token_path": token_path
        }
        vts_plugin = pyvts.vts(plugin_info=plugin_info)

        logger.info("Langkah 2: Menghubungkan ke WebSocket VTS...")
        await vts_plugin.connect()
        logger.info("Koneksi WebSocket berhasil.")

        # Cek dulu apakah sudah terautentikasi menggunakan fungsi yang benar
        # Status 2 berarti sudah terautentikasi
        if vts_plugin.get_authentic_status() != 2: # <-- PERBAIKAN DI SINI
            logger.info("Belum terautentikasi. Meminta token (Cek pop-up di VTube Studio!)...")
            await vts_plugin.request_authenticate_token()
            await vts_plugin.write_token()

        logger.info("Menggunakan token untuk autentikasi sesi...")
        await vts_plugin.request_authenticate()

        # Cek status akhir menggunakan fungsi yang benar
        if vts_plugin.get_authentic_status() == 2: # <-- PERBAIKAN DI SINI JUGA
            logger.info("Autentikasi BERHASIL.")
            vts_connected = True
            return "Berhasil terhubung dan terautentikasi dengan VTube Studio!", vts_plugin
        else:
            logger.warning("Autentikasi GAGAL setelah mencoba mendapatkan token.")
            await disconnect_vts_plugin()
            return "Gagal autentikasi. Pastikan Anda menerima permintaan di VTube Studio.", None

    except AssertionError as e:
        logger.error(f"AssertionError: {e}. Kemungkinan Anda menolak izin plugin di VTube Studio.")
        await disconnect_vts_plugin()
        return f"Gagal: {e}. Pastikan izin diberikan di VTS.", None
    except Exception as e:
        logger.error(f"Terjadi error tak terduga: {e}", exc_info=True)
        await disconnect_vts_plugin()
        return f"Gagal terhubung ke VTS. Pastikan API aktif. Error: {e}", None

async def disconnect_vts_plugin():
    """Memutuskan koneksi dari VTube Studio."""
    global vts_plugin, vts_connected
    if vts_plugin and vts_connected:
        try:
            await vts_plugin.close()
            logger.info("Koneksi VTube Studio ditutup.")
        except Exception as e:
            logger.error(f"Error saat menutup koneksi VTS: {e}")
    vts_plugin = None
    vts_connected = False
    return "Koneksi VTube Studio diputus."

async def control_mouth_with_audio(audio_path):
    """Menganalisis file audio dan mengirimkan data volume ke VTS untuk menggerakkan mulut."""
    global vts_plugin
    if not vts_plugin or not vts_connected or not audio_path:
        return

    try:
        logger.info(f"Memproses audio {audio_path} untuk gerakan mulut VTS.")
        # Muat file audio MP3
        audio = AudioSegment.from_mp3(audio_path)
        # Dapatkan parameter MouthOpen
        param_data = await vts_plugin.get_parameter("MouthOpen")
        if not param_data:
            logger.warning("Tidak dapat menemukan parameter 'MouthOpen' di model VTS.")
            return

        # Kirim data volume per chunk untuk animasi yang lebih halus
        chunk_length_ms = 100 # 100ms per chunk
        chunks = make_chunks(audio, chunk_length_ms)

        for chunk in chunks:
            # RMS adalah cara sederhana untuk mengukur volume
            volume = chunk.rms
            # Normalisasi volume ke rentang parameter (biasanya 0-1)
            # Nilai 10000 ini mungkin perlu disesuaikan tergantung audio Anda
            normalized_volume = min(volume / 10000.0, 1.0)

            # Kirim nilai parameter ke VTube Studio
            await param_data.set_value(normalized_volume)
            # Jeda singkat agar VTS tidak kewalahan
            await asyncio.sleep(chunk_length_ms / 1000.0)

        # Tutup mulut setelah selesai berbicara
        await param_data.set_value(0)
        logger.info("Animasi mulut VTS selesai.")
    except Exception as e:
        logger.error(f"Error saat mengontrol mulut VTS: {e}")
        # Pastikan mulut tertutup jika terjadi error
        if vts_plugin and vts_connected:
            try:
                param_data = await vts_plugin.get_parameter("MouthOpen")
                if param_data:
                    await param_data.set_value(0)
            except:
                pass

# --- FUNGSI BARU UNTUK MEMICU HOTKEY ---
async def trigger_vts_hotkey(hotkey_id):
    """Mengirim permintaan untuk memicu hotkey berdasarkan ID/namanya."""
    global vts_plugin, vts_connected
    if not vts_plugin or not vts_connected:
        logger.warning("VTS tidak terhubung, tidak bisa memicu hotkey.")
        return

    try:
        logger.info(f"Mencoba memicu hotkey: {hotkey_id}")
        # Membuat request menggunakan metode yang sudah disediakan pyvts
        hotkey_trigger_request = vts_plugin.vts_request.requestTriggerHotKey(hotkey_id=hotkey_id)

        # Mengirim request
        response = await vts_plugin.request(hotkey_trigger_request)

        # Periksa jika ada error dari VTS
        if response and response.get("data", {}).get("errorID", 0) != 0:
            logger.error(f"Gagal memicu hotkey '{hotkey_id}': {response.get('data').get('message')}")
        else:
            logger.info(f"Hotkey '{hotkey_id}' berhasil dipicu.")

    except Exception as e:
        logger.error(f"Terjadi exception saat memicu hotkey '{hotkey_id}': {e}")

async def process_vtuber_response(history, system_prompt):
    """
    Mendapatkan respons dari AI, mem-parse JSON, memicu hotkey, 
    dan menyiapkan audio untuk lip-sync.
    """
    logger.info("Mendapatkan respons AI untuk VTuber...")
    # Kita gunakan fungsi non-streaming untuk memastikan kita mendapat JSON lengkap
    full_response_str = get_ai_response(history, system_prompt)

    if full_response_str.startswith("__ERROR__:"):
        logger.error(f"Gagal mendapatkan respons AI: {full_response_str}")
        return "Maaf, terjadi error saat memproses respons.", None, None

    try:
        # Coba membersihkan dan mem-parse JSON dari respons AI
        # Terkadang AI menambahkan markdown ```json ... ```
        cleaned_str = re.sub(r'^```json\s*|```\s*$', '', full_response_str, flags=re.MULTILINE).strip()
        response_data = json.loads(cleaned_str)
        
        emotion = response_data.get("emotion", "netral")
        text_response = response_data.get("response", "Aku tidak tahu harus berkata apa.")
        
        logger.info(f"AI memilih emosi: '{emotion}' dan respons: '{text_response[:50]}...'")

        # Memicu hotkey berdasarkan emosi yang dipilih AI
        await trigger_vts_hotkey(emotion)

        # Membuat file audio dari respons teks
        audio_path = text_to_speech(text_response)
        if audio_path:
            # Menggerakkan mulut sesuai audio yang baru dibuat
            await control_mouth_with_audio(audio_path)

        return text_response, audio_path, emotion

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Gagal mem-parse JSON dari AI: {e}. Menggunakan respons sebagai teks biasa.")
        # Fallback: Jika AI gagal memberi JSON, anggap semua sebagai teks biasa
        await trigger_vts_hotkey("netral") # Set ke ekspresi netral
        audio_path = text_to_speech(full_response_str)
        if audio_path:
            await control_mouth_with_audio(audio_path)
        return full_response_str, audio_path, "netral"
    
    # Di dalam backend_logic.py

def apply_content_filter(text, memory_content):
    """Menyensor kata-kata yang tidak diinginkan dari teks berdasarkan memori."""
    if not memory_content or not text:
        return text
    
    banned_words = [line.strip().lower() for line in memory_content.split('\n') if line.strip()]
    if not banned_words:
        return text
        
    # Buat pola regex yang cocok dengan kata-kata yang diblokir (case-insensitive)
    # \b memastikan kita hanya mencocokkan seluruh kata
    pattern = r'\b(' + '|'.join(re.escape(word) for word in banned_words) + r')\b'
    
    # Ganti kata yang cocok dengan '[disensor]'
    censored_text = re.sub(pattern, '[disensor]', text, flags=re.IGNORECASE)
    
    return censored_text

def capture_screen_to_base64():
    """Mengambil tangkapan layar dan mengubahnya menjadi base64."""
    if not MSS_AVAILABLE:
        return None, "Error: Library 'mss' tidak terinstal."
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            base64_img = image_to_base64(pil_img)
            return base64_img, "Tangkapan layar berhasil diambil."
    except Exception as e:
        logger.error(f"Gagal mengambil tangkapan layar: {e}")
        return None, f"Error: {e}"

def capture_camera_to_base64():
    """Mengambil satu frame dari kamera dan mengubahnya menjadi base64."""
    if not CV2_AVAILABLE:
        return None, "Error: Library 'opencv-python' tidak terinstal."
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Error: Tidak dapat membuka kamera."

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Error: Gagal mengambil gambar dari kamera."

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        base64_img = image_to_base64(pil_img)
        return base64_img, "Gambar dari kamera berhasil diambil."
    except Exception as e:
        logger.error(f"Gagal mengakses kamera: {e}")
        return None, f"Error: {e}"

def process_vision_for_vtuber(image_base64, history, ai_persona, user_persona, memory_content=""):
    """Memproses gambar untuk VTuber, mendapatkan respons, dan memfilternya."""
    if not image_base64:
        return "Gagal memproses gambar.", None

    system_prompt, _ = build_roleplay_prompt(ai_persona, user_persona, "Sudut Pandang Kedua (Interaktif)", memory_content) 

    multimodal_content = [
        {"type": "text", "text": "Kamu adalah seorang VTuber. Lihat gambar ini dan berikan komentarmu secara singkat dan menarik seolah-olah kamu sedang live streaming."},
        {"type": "image_url", "image_url": {"url": image_base64}}
    ]

    vision_history = history + [{"role": "user", "content": multimodal_content}]

    response_text = get_ai_response(vision_history, system_prompt)
    if response_text.startswith("__ERROR__:"):
        return response_text, None

    # Kita gunakan lagi filter kata yang sudah dibuat
    filtered_response = apply_content_filter(response_text, memory_content)
    audio_path = text_to_speech(filtered_response) 

    return filtered_response, audio_path

def load_yolo_model():
    """Memuat model YOLO jika belum ada."""
    global yolo_model
    if not YOLO_AVAILABLE:
        return False
    if yolo_model is None:
        try:
            logger.info("Memuat model YOLOv8n (nano)... Ini mungkin perlu mengunduh saat pertama kali.")
            # 'yolov8n.pt' adalah model yang kecil dan cepat, cocok untuk permulaan
            yolo_model = YOLO('yolov8n.pt')
            logger.info("Model YOLO berhasil dimuat.")
            return True
        except Exception as e:
            logger.error(f"Gagal memuat model YOLO: {e}")
            return False
    return True

def live_screen_observer(stop_event, message_queue, target_object="person"):
    """
    Fungsi yang berjalan di thread untuk mengamati layar dan mendeteksi objek.
    'stop_event' digunakan untuk menghentikan loop dari luar.
    'message_queue' digunakan untuk mengirim pesan kembali ke UI.
    'target_object' adalah apa yang ingin kita cari.
    """
    if not load_yolo_model():
        message_queue.append("__ERROR__: Model YOLO tidak dapat dimuat.")
        return

    logger.info(f"Memulai pengamatan layar untuk objek: '{target_object}'")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while not stop_event.is_set():
            try:
                # 1. Ambil gambar layar
                sct_img = sct.grab(monitor)
                frame = np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))

                # 2. Lakukan deteksi objek dengan YOLO
                results = yolo_model(frame, verbose=False) # verbose=False agar tidak print log ke konsol

                # 3. Proses hasilnya
                for result in results:
                    # Dapatkan nama-nama objek yang terdeteksi
                    detected_names = [yolo_model.names[int(cls)] for cls in result.boxes.cls]
                    
                    # 4. Jika objek yang kita cari ditemukan
                    if target_object.lower() in [name.lower() for name in detected_names]:
                        confidence_scores = result.boxes.conf.tolist()
                        highest_confidence = max(confidence_scores) if confidence_scores else 0
                        
                        # Kirim pesan ke UI dan berhenti sejenak agar tidak spam
                        message = f"Objek '{target_object}' terdeteksi dengan keyakinan {highest_confidence:.2f}!"
                        logger.info(message)
                        message_queue.append(message)
                        time.sleep(5) # Jeda 5 detik setelah menemukan objek
                        break # Hentikan proses deteksi untuk frame ini

                # Beri jeda singkat agar tidak membebani CPU
                time.sleep(0.5)

            except Exception as e:
                error_msg = f"__ERROR__: Terjadi error di loop pengamatan: {e}"
                logger.error(error_msg)
                message_queue.append(error_msg)
                break
    
    logger.info("Pengamatan layar dihentikan.")
    message_queue.append("__INFO__: Pengamatan dihentikan.")

def classify_news_hoax(article_text):
    """
    Menggunakan AI untuk menganalisis teks artikel dan mengklasifikasikannya
    sebagai berita asli, potensi hoax, atau tidak dapat dipastikan.
    """
    if not article_text or len(article_text) < 150:
        return {"klasifikasi": "Tidak Cukup Informasi", "alasan": "Teks artikel terlalu pendek untuk dianalisis."}

    # Prompt khusus untuk AI agar bertindak sebagai fact-checker
    fact_checker_prompt = f"""
Anda adalah seorang AI fact-checker yang sangat teliti dan objektif. Tugas Anda adalah menganalisis TEKS ARTIKEL berikut dan menentukan apakah artikel tersebut merupakan berita asli, berpotensi hoax, atau tidak dapat dipastikan.

Berikan jawaban Anda dalam format JSON yang valid dengan dua kunci: "klasifikasi" dan "alasan".

- Untuk "klasifikasi", pilih SATU dari tiga nilai berikut: "Berita Asli", "Potensi Hoax", "Tidak Dapat Dipastikan".
- Untuk "alasan", berikan penjelasan singkat (1-2 kalimat) mengenai keputusan Anda. Perhatikan ciri-ciri seperti: judul yang sensasional, ketiadaan sumber yang jelas, nada tulisan yang provokatif, atau informasi yang tidak masuk akal.

--- TEKS ARTIKEL UNTUK DIANALISIS ---
{article_text[:4000]}
--- AKHIR TEKS ARTIKEL ---

CONTOH JAWABAN JSON:
{{
  "klasifikasi": "Potensi Hoax",
  "alasan": "Judulnya sangat provokatif dan isinya tidak menyertakan kutipan dari sumber atau pakar yang dapat diverifikasi."
}}

JAWABAN JSON ANDA:
"""
    try:
        response_str = get_ai_response([], fact_checker_prompt)
        # Membersihkan respons dari markdown code block jika ada
        cleaned_str = re.sub(r'^```json\s*|```\s*$', '', response_str, flags=re.MULTILINE).strip()
        response_json = json.loads(cleaned_str)
        
        # Validasi dasar
        if "klasifikasi" in response_json and "alasan" in response_json:
            return response_json
        else:
            return {"klasifikasi": "Error Parsing", "alasan": "AI tidak mengembalikan format JSON yang benar."}
            
    except Exception as e:
        logger.error(f"Gagal saat analisis hoax: {e}")
        return {"klasifikasi": "Error Analisis", "alasan": str(e)}