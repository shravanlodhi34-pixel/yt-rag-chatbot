from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF location


pdf_path = Path(__file__).parent.parent / "DOCUMENTS" / "AI_RAG_Sample_Document.pdf"

# Read PDF
reader = PdfReader(str(pdf_path))

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print("PDF text extracted successfully!")
print("Total characters:", len(text))

# Create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

# Show first chunk
print("\nFirst chunk:")
print(chunks[0])