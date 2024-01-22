from llama_index import download_loader


class PDFReader(object):
    Loader = download_loader("PDFReader")
    loader = Loader()

    #TODO: Make sure it works without initialization so that downloader is run only once
    @staticmethod
    def load_pdf(filename: str):
        documents = PDFReader.loader.load_data(file=filename)
        return documents
