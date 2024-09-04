import os
import subprocess
import shutil
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.utils import executor
from aiogram.types.message import ContentType

# Initialize bot and dispatcher
bot = Bot(token=os.getenv('DOTCODE_BOT_TOKEN'))
dp = Dispatcher(bot)

# Directory to store the uploaded files temporarily
TEMP_DIR = "temp_uploads"

# Set up logging to check for potential errors
logging.basicConfig(level=logging.INFO)


# Command handler for /start command
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Send me a photo, and I'll process it using the script!")


# Handler for photo upload
@dp.message_handler(content_types=ContentType.PHOTO)
async def handle_photo(message: types.Message):
    # Create a temporary directory to store the uploaded photo
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Download the photo and save it as "photo.jpg"
    photo = message.photo[-1]  # Get the highest resolution photo
    photo_file = await bot.download_file_by_id(photo.file_id)
    photo_path = f"{TEMP_DIR}/photo.jpg"  # Always save as "photo.jpg"

    with open(photo_path, "wb") as f:
        f.write(photo_file.getvalue())

    # Output directory for the processed files
    output_dir = f"{TEMP_DIR}/output"

    try:
        # Call the processing script with the downloaded image
        logging.info(f"Running script on {photo_path}, output will be in {output_dir}")
        result = subprocess.run(["python3", "main.py", photo_path, "-o", output_dir], check=True)

        # Check if script ran successfully
        if result.returncode == 0:
            logging.info(f"Script completed successfully. Sending files from {output_dir}.")
            # After processing, gather the output files
            await send_output_files(message.chat.id, f"{TEMP_DIR}/output/photo")
        else:
            logging.error(f"Script failed with return code {result.returncode}")
            await message.reply("Script failed to process the image.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing the image: {str(e)}")
        await message.reply(f"Error processing the image: {str(e)}")

    # Cleanup the temporary files
    cleanup_temp_files(photo_path, f"{TEMP_DIR}/output/photo")


# Function to send processed files back to the user
async def send_output_files(chat_id, output_dir):
    files_to_send = ['grid.txt', 'grid_checker.txt', 'points.png', 'warped.png', 'thresh.png']

    for file_name in files_to_send:
        file_path = f"{output_dir}/{file_name}"
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logging.info(f"Sending file {file_path} to user.")
            await bot.send_document(chat_id, InputFile(file_path))
        else:
            logging.warning(f"File {file_path} is missing or empty. Skipping.")


# Function to clean up temporary files
def cleanup_temp_files(photo_path, output_dir):
    # Delete the uploaded photo
    if os.path.exists(photo_path):
        os.remove(photo_path)

    # Delete the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Check if TEMP_DIR is empty, if so, remove it
    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)


if __name__ == '__main__':
    logging.info("Bot started")
    executor.start_polling(dp, skip_updates=True)
