import os
import json
import re
import logging
import gradio as gr
from config import WEB_CHATS_DIR, RP_CHATS_DIR, CHARACTERS_DIR, SETTINGS_FILE, MEMORY_WEB_DIR, MEMORY_RP_DIR, global_settings

logger = logging.getLogger(__name__)

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding='utf-8') as f:
                global_settings.update(json.load(f))
            logger.info("Pengaturan berhasil dimuat dari file.")
    except Exception as e:
        logger.error(f"Gagal memuat pengaturan: {e}")
    return global_settings

def save_settings(provider, or_key, ds_key, lm_url, model_name, search_count, bb_key, bb_model, google_key, google_cx, gemini_key):
    try:
        settings_to_save = {
            "api_provider": provider, "openrouter_api_key": or_key, "deepseek_api_key": ds_key,
            "blackbox_api_key": bb_key, "blackbox_model_name": bb_model, "lmstudio_api_url": lm_url,
            "model_name": model_name, "search_results_count": int(search_count),
            "search_results_count": int(search_count),
            "google_api_key": google_key,   
            "google_cse_id": google_cx,
            "gemini_api_key": gemini_key      
        }
        global_settings.update(settings_to_save)
        with open(SETTINGS_FILE, "w", encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4, ensure_ascii=False)
        logger.info("Pengaturan berhasil disimpan.")
        return "Pengaturan berhasil disimpan!"
    except Exception as e:
        return f"Error menyimpan pengaturan: {e}"

def list_files_in_dir(directory):
    if not os.path.isdir(directory): return []
    try:
        return sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    except Exception as e:
        logger.error(f"Gagal membaca direktori {directory}: {e}")
        return []


def get_next_chat_filename(directory, prefix="chat"):
    """Mendapatkan nama file chat berikutnya berdasarkan nomor."""
    try:
        files = list_files_in_dir(directory)
        numeric_files = []
        
        for f in files:
            match = re.search(r'_(\d+)\.json$', f)
            if match:
                numeric_files.append(int(match.group(1)))
        
        next_num = max(numeric_files) + 1 if numeric_files else 1
        return f"{prefix}_{next_num}.json"
    except Exception as e:
        logger.error(f"Gagal menghasilkan nama file: {e}")
        return f"{prefix}_1.json"

def save_chat_history(history, chat_name, directory, prefix):
    """Menyimpan riwayat obrolan ke file JSON."""
    try:
        if not chat_name or not chat_name.strip():
            chat_name = get_next_chat_filename(directory, prefix)
        
        file_name = f"{chat_name}.json" if not chat_name.endswith(".json") else chat_name
        file_path = os.path.join(directory, file_name)
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Chat '{file_name}' disimpan di {directory}.")
        return f"Chat '{file_name}' disimpan!", gr.update(choices=list_files_in_dir(directory), value=file_name)
    except Exception as e:
        logger.error(f"Gagal menyimpan chat: {e}")
        return f"Error menyimpan chat: {e}", gr.update()

def load_chat_history(chat_name, directory):
    """Memuat riwayat obrolan dari file JSON."""
    try:
        if not chat_name or not chat_name.strip():
            return [], "Pilih chat untuk dimuat."
        
        file_path = os.path.join(directory, chat_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                history = json.load(f)
            
            logger.info(f"Chat '{chat_name}' dimuat dari {directory}.")
            return history, f"Chat '{chat_name}' dimuat!"
        else:
            logger.warning(f"File '{chat_name}' tidak ditemukan di {directory}.")
            return [], f"Error: File '{chat_name}' tidak ditemukan."
    except Exception as e:
        logger.error(f"Gagal memuat chat: {e}")
        return [], f"Error memuat chat: {e}"

def delete_chat_history(chat_name, directory):
    """Menghapus file riwayat obrolan."""
    try:
        if not chat_name or not chat_name.strip():
            return "Pilih chat untuk dihapus.", gr.update()
        
        file_path = os.path.join(directory, chat_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Chat '{chat_name}' dihapus dari {directory}.")
            return f"Chat '{chat_name}' dihapus.", gr.update(choices=list_files_in_dir(directory), value=None)
        else:
            logger.warning(f"File '{chat_name}' tidak ditemukan di {directory}.")
            return f"Error: File '{chat_name}' tidak ditemukan.", gr.update()
    except Exception as e:
        logger.error(f"Gagal menghapus chat: {e}")
        return f"Error menghapus chat: {e}", gr.update()

def save_character(ai_persona, user_persona, name):
    """Menyimpan deskripsi karakter AI dan pengguna ke file JSON."""
    try:
        if not name or not name.strip():
            return "Nama karakter tidak boleh kosong.", gr.update(), name
        
        file_name = f"{name}.json" if not name.endswith(".json") else name
        file_path = os.path.join(CHARACTERS_DIR, file_name)
        
        char_data = {
            "ai_persona": ai_persona, 
            "user_persona": user_persona
        }
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(char_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Karakter '{file_name}' disimpan.")
        return f"Karakter '{file_name}' disimpan!", gr.update(choices=list_files_in_dir(CHARACTERS_DIR), value=file_name), name
    except Exception as e:
        logger.error(f"Gagal menyimpan karakter: {e}")
        return f"Error menyimpan karakter: {e}", gr.update(), name

def load_character(name):
    """Memuat deskripsi karakter dari file JSON."""
    try:
        if not name or not name.strip():
            return "", "", "Pilih karakter untuk dimuat.", name
        
        file_path = os.path.join(CHARACTERS_DIR, name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                char_data = json.load(f)
            
            ai_persona = char_data.get("ai_persona", "")
            user_persona = char_data.get("user_persona", "")
            file_name_without_ext = os.path.splitext(name)[0]
            
            logger.info(f"Karakter '{name}' dimuat.")
            return ai_persona, user_persona, f"Karakter '{name}' dimuat!", file_name_without_ext
        else:
            logger.warning(f"File karakter '{name}' tidak ditemukan.")
            return "", "", f"Error: Karakter '{name}' tidak ditemukan.", name
    except Exception as e:
        logger.error(f"Gagal memuat karakter: {e}")
        return "", "", f"Error memuat karakter: {e}", name

def delete_character(name):
    """Menghapus file karakter."""
    try:
        if not name or not name.strip():
            return "Pilih karakter untuk dihapus.", gr.update(), "", "", ""
        
        file_path = os.path.join(CHARACTERS_DIR, name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Karakter '{name}' dihapus.")
            return f"Karakter '{name}' dihapus.", gr.update(choices=list_files_in_dir(CHARACTERS_DIR), value=None), "", "", ""
        else:
            logger.warning(f"File karakter '{name}' tidak ditemukan.")
            return f"Error: Karakter '{name}' tidak ditemukan.", gr.update(), "", "", ""
    except Exception as e:
        logger.error(f"Gagal menghapus karakter: {e}")
        return f"Error menghapus karakter: {e}", gr.update(), "", "", ""

def get_memory_filepath(memory_id):
    """Mendapatkan path file memori berdasarkan ID-nya."""
    try:
        if memory_id.startswith("rp_"):
            return os.path.join(MEMORY_RP_DIR, memory_id)
        else:
            return os.path.join(MEMORY_WEB_DIR, memory_id)
    except Exception as e:
        logger.error(f"Gagal mendapatkan path memori: {e}")
        return ""

def load_memory(memory_id):
    """Memuat konten memori dari file."""
    try:
        filepath = get_memory_filepath(memory_id)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding='utf-8') as f:
                content = f.read()
            return content
        return ""
    except Exception as e:
        logger.error(f"Gagal memuat memori: {e}")
        return ""

def add_to_memory(memory_id, new_fact):
    """Menambahkan fakta baru ke file memori."""
    try:
        if not new_fact or not new_fact.strip():
            return "Fakta tidak boleh kosong."
        
        filepath = get_memory_filepath(memory_id)
        with open(filepath, "a", encoding='utf-8') as f:
            f.write(f"- {new_fact}\n")
        
        logger.info(f"Fakta ditambahkan ke memori '{memory_id}'.")
        return f"Fakta '{new_fact[:30]}...' ditambahkan ke memori."
    except Exception as e:
        logger.error(f"Gagal menambahkan ke memori: {e}")
        return f"Error menambahkan ke memori: {e}"

def clear_memory(memory_id):
    """Menghapus semua konten dari file memori."""
    try:
        filepath = get_memory_filepath(memory_id)
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.info(f"Memori '{memory_id}' dibersihkan.")
        return ""
    except Exception as e:
        logger.error(f"Gagal membersihkan memori: {e}")
        return f"Error membersihkan memori: {e}"

def get_memory_list(directory):
    if not os.path.isdir(directory): return []
    return sorted([f for f in os.listdir(directory) if f.endswith('.txt')])

def load_memory(directory, filename):
    if not filename or not directory: return ""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding='utf-8') as f: return f.read()
    return ""

def add_to_memory(directory, filename, new_fact):
    if not filename or not new_fact or not new_fact.strip(): return "Nama file atau fakta tidak boleh kosong."
    filepath = os.path.join(directory, filename)
    with open(filepath, "a", encoding='utf-8') as f: f.write(f"- {new_fact}\n")
    return f"Fakta ditambahkan ke memori '{filename}'."

def clear_memory(directory, filename):
    if not filename: return "Pilih file memori untuk dibersihkan."
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        with open(filepath, "w", encoding='utf-8') as f: f.write("")
        return f"Memori '{filename}' dibersihkan."
    return f"File '{filename}' tidak ditemukan."

def delete_memory_file(directory, filename):
    if not filename: return "Pilih file memori untuk dihapus.", gr.update()
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return f"File memori '{filename}' dihapus.", gr.update(choices=get_memory_list(directory), value=None)
    return f"File '{filename}' tidak ditemukan.", gr.update()