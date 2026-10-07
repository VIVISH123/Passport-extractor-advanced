import shutil
import pytesseract

path = shutil.which("tesseract")

if not path:
    print("Tesseract was not found on PATH.")
    print("Install Tesseract OCR, then run this script again.")
    raise SystemExit(1)

print(f"Tesseract found: {path}")
print("Version:")
print(pytesseract.get_tesseract_version())
