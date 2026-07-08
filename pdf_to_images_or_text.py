import os
import sys

pdf_path = "/home/andre-marques/Desktop/Estudo/PowerRangersCode/apresentação 2.0.pdf"
output_dir = "/home/andre-marques/Desktop/Estudo/PowerRangersCode"

print(f"Reading PDF from: {pdf_path}")

# Try to extract text first using pypdf
extracted_text = []
try:
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    print(f"PyPDF read successfully. Total pages: {len(reader.pages)}")
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        extracted_text.append(f"--- PAGE {idx+1} ---\n{text}\n")
    with open(os.path.join(output_dir, "extracted_text.txt"), "w") as f:
        f.writelines(extracted_text)
    print("Extracted text saved to extracted_text.txt")
except Exception as e:
    print(f"Failed to extract text with pypdf: {e}")

# Try to render pages as images using PyMuPDF (fitz)
try:
    import fitz
    doc = fitz.open(pdf_path)
    print(f"PyMuPDF open successfully. Total pages: {doc.page_count}")
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img_path = os.path.join(output_dir, f"slide_{i+1}.png")
        pix.save(img_path)
        print(f"Saved slide_{i+1}.png")
except Exception as e:
    print(f"Failed to convert to images using fitz: {e}")
