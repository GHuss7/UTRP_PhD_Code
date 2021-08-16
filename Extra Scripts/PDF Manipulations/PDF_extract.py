from PyPDF2 import PdfFileReader, PdfFileWriter
import argparse

# Arguments from command line
parser = argparse.ArgumentParser()

#-dir DIRECTORY
parser.add_argument("-s", "--start", dest = "start_page", default = 1, help="Start_page", type=int)
parser.add_argument("-e", "--end", dest = "end_page", default = -1, help="End_page", type=int)
parser.add_argument("-d", "--doc", dest = "doc_name", default = "main", help="Doc_name", type=str)


args = parser.parse_args()
arg_start = args.start_page
arg_end = args.end_page
doc_name = args.doc_name

# Note: index starts at 1 and is inclusive of the end. 
# The following will extract page 3 of the pdf file.
pdfs = {doc_name+'.pdf': ({'start': arg_start, 'end': arg_end},)}  

for pdf, segments in pdfs.items():
    pdf_reader = PdfFileReader(open(pdf, 'rb'))
    for segment in segments:
        pdf_writer = PdfFileWriter()
        start_page = segment['start']
        if segment['end'] == -1:
            segment['end'] = pdf_reader.getNumPages()
        end_page = segment['end']
        for page_num in range(start_page - 1, end_page):
            pdf_writer.addPage(pdf_reader.getPage(page_num))
        output_filename = f'{pdf.replace(".pdf", "")}_{start_page}_to_{end_page}.pdf'
        with open(output_filename,'wb') as out:
            pdf_writer.write(out)