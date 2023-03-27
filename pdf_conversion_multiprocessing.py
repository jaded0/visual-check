import glob
import time
import subprocess as sub
import multiprocessing

def process_pdf(pdf_dpi_tuple):
    pdf, dpi = pdf_dpi_tuple
    i = pdfs.index(pdf)
    sub.run(f"pdftoppm -png -rx {dpi} -ry {dpi} {pdf} pdfs/bigmages/{i}", shell=True, capture_output=True)
    return i

def update_progress(results):
    processed_count = 0
    for result in results:
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"Processed {processed_count} PDF files.")
    print("PDF conversion completed.")

pdfs = glob.glob('pdfs/2302/*')
dpi = 150

# You can adjust the number of processes based on your machine's resources
num_processes = multiprocessing.cpu_count()

with multiprocessing.Pool(num_processes) as pool:
    results = pool.imap_unordered(process_pdf, [(pdf, dpi) for pdf in pdfs])
    update_progress(results)
