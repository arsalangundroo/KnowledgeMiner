from knowledgeminer.rag.loaders import PDFReader

if __name__ == "__main__":
    documents = PDFReader.load_pdf(filename='/Users/gar1syv/Desktop/books/Quran/quran-in-modern-english.pdf')
    print(len(documents))
    print(documents)