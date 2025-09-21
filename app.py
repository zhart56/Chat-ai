from ui.main_ui import create_ui
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Aplikasi Chat AI dimulai")
    
    # Membuat antarmuka dari file ui/main_ui.py
    try:
        app_ui = create_ui()
        # Menjalankan aplikasi
        app_ui.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Error menjalankan aplikasi: {e}")
        print(f"Error: {e}. Lihat app.log untuk detail.")
