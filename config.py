import os
from pathlib import Path

# Konfigurasi path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VTUBER_CHATS_DIR = DATA_DIR / "vtuber_chats"
# Buat direktori data jika belum ada
DATA_DIR.mkdir(exist_ok=True)

# Konfigurasi URL dan Direktori
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
BLACKBOX_API_URL = "https://api.blackbox.ai/chat/completions" 
LMSTUDIO_API_URL_DEFAULT = "http://localhost:1234/v1"

# Path untuk berbagai direktori
SETTINGS_FILE = DATA_DIR / "settings.json"
WEB_CHATS_DIR = DATA_DIR / "web_chats"
RP_CHATS_DIR = DATA_DIR / "rp_chats"
CHARACTERS_DIR = DATA_DIR / "characters"
MEMORY_WEB_DIR = DATA_DIR / "memory_web"
MEMORY_RP_DIR = DATA_DIR / "memory_rp"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"

# Buat semua direktori yang dibutuhkan
for directory in [WEB_CHATS_DIR, RP_CHATS_DIR, CHARACTERS_DIR, MEMORY_WEB_DIR, MEMORY_RP_DIR, TTS_CACHE_DIR, VTUBER_CHATS_DIR]:
    directory.mkdir(exist_ok=True)

# Variabel global untuk menyimpan pengaturan sesi
global_settings = {
    "api_provider": "OpenRouter",  # Default provider
    "openrouter_api_key": "",
    "deepseek_api_key": "",
    "blackbox_api_key": "",
    "blackbox_model_name": "blackboxai/openai/gpt-4",
    "lmstudio_api_url": LMSTUDIO_API_URL_DEFAULT,
    "model_name": "google/gemini-flash-1.5",
    "search_results_count": 5,
    "google_api_key": "",      
    "google_cse_id": "",
    "gemini_api_key": ""
}

# System prompts default
DEFAULT_SYSTEM_PROMPT = """Anda adalah asisten AI yang cerdas dan ramah. Berikan jawaban yang informatif dan jelas. Gunakan bahasa Indonesia yang baik dan benar. Jika Anda tidak yakin dengan jawabannya, katakan saja bahwa Anda tidak tahu. Jangan membuat-buat informasi."""

DEFAULT_RP_SYSTEM_PROMPT = """Anda adalah karakter dalam roleplay. Tetap dalam karakter, jadikan interaksi ini menarik dan imersif. Gunakan bahasa Indonesia yang baik dan benar."""

