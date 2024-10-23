import argparse
import io
import lzma
from pathlib import Path
import time


DATA_PATH = Path(__file__).parent.parent / 'TRANSLATION_MATRICES'
COMPRESSED_DATA_PATH = Path(__file__).parent.parent / 'TRANSLATION_MATRICES/compressed'

def compress_t_xy():
    """function that compress the Txy translation pre-calculated
     matrices, and save them in multiple files"""
    name = 't_xy'

    CHUNK_SIZE = 5 * 1024 * 1024

    tic = time.time()

    print('Compression and splitting of Txy translation matrices - it could take up to 15 minutes!')

    comp = lzma.LZMACompressor(preset=9)
    out = b''

    with open(DATA_PATH / f'{name}.pkl', 'rb') as fp:
        chunk = fp.read(CHUNK_SIZE)
        while len(chunk) > 0:
            out = out + comp.compress(chunk)
            chunk = fp.read(CHUNK_SIZE)
    out = out + comp.flush()
    toc = time.time()
    print(f'     -> compression finished in {(toc - tic)/60:.2f} min')
    print('     -> saving the compressed data in multiple files')

    toc = time.time()

    file_number = 0
    out_buff = io.BytesIO(out)

    VOLUME_SIZE = 45 * 1024 * 1024
    if len(out) > VOLUME_SIZE:
        partial_output = out_buff.read(VOLUME_SIZE)
        while len(partial_output) > 0:
            with open(COMPRESSED_DATA_PATH / f'{name}.{file_number}.xz', 'wb') as fp:
                fp.write(partial_output)
            file_number += 1
            partial_output = out_buff.read(VOLUME_SIZE)
    else:
        with open(COMPRESSED_DATA_PATH / f'{name}.xz', 'wb') as fp:
            fp.write(out)

    print(f'     -> finished saving the {file_number} files  `{name}.?.xz` in {(toc - tic)/60:.2f} min')


def compress_t_z():
    """function that compress the Tz translation pre-calculated"""
    
    name = 't_z'
    CHUNK_SIZE = 5 * 1024 * 1024

    tic = time.time()
    print('Compression and splitting of Tz translation matrices - it could take up a few minutes!')
    comp = lzma.LZMACompressor(preset=9)
    out = b''

    with open(DATA_PATH / f'{name}.pkl', 'rb') as fp:
        while chunk := fp.read(CHUNK_SIZE):
            out = out + comp.compress(chunk)
    out = out + comp.flush()
    toc = time.time()
    print(f'     -> compression finished in {(toc - tic)/60:.2f} min')

    tic = time.time()

    with open(COMPRESSED_DATA_PATH / f'{name}.xz', 'wb') as fp:
        fp.write(out)
    toc = time.time()

    print(f'     -> finished saving the file {name}.xz files in {(toc - tic)/60:.2f} min')
  

def decompress_t_xy():
    """post install function that joins the splitted lzma compressed file with the Txy translation pre-calculated
     matrices, and uncompress that file"""
    name = 't_xy'
    print('Decompression of Txy translation matrices - it could take a few minutes!')
    number_of_files = 13
    tic = time.time()
    out = b''
    for k in range(number_of_files):
        with open(COMPRESSED_DATA_PATH / f'{name}.{k}.xz', 'rb') as fp:
            out = out + fp.read()
    content = lzma.decompress(out)
    toc = time.time()
    print(f'     -> decompression finished in {(toc - tic)/60:.2f} min, saving file {name}.pkl')
    
    tic = time.time()
    with open(DATA_PATH / f'{name}.pkl', 'wb') as fp:
         fp.write(content)
    toc = time.time()
    print(f'     -> file {name}.pkl saved in {(toc - tic)/60:.2f} min')
    


def decompress_t_z():
    """post install function uncompress the lzma compressed file with the Tz translation pre-calculated
     matrices"""
    name = 't_z'
    print('Decompression of Tz translation matrices - it could take a few minutes!')
        
    tic = time.time()
    with lzma.open(COMPRESSED_DATA_PATH / f'{name}.xz', 'rb') as fp:
        content = fp.read()
    toc = time.time()
    print(f'     -> decompression finished in {(toc - tic)/60:.2f} min, saving file {name}.pkl')
    
    tic = time.time()
    with open(DATA_PATH / f'{name}.pkl', 'wb') as fp:
        fp.write(content)
    toc = time.time()
    print(f'     -> file {name}.pkl in {toc - tic:.2f} sec')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='precalc_matrices_compression',
                description='It handles the compression and decompression of the precalculated translation matrices.\nSelect either `compress` or `decompress`',
                epilog='',
                usage='%(prog)s [-c | -d]')
    
    parser.add_argument('-c', '--compress',
                action='store_true', help='Compress the precalc matrices') 

    
    parser.add_argument('-d', '--decompress',
                action='store_true', help='Decompress the precalc matrices')



    args = parser.parse_args()
    if args.compress and (not args.decompress):
        print('Executed with the --compress option')
        compress_t_xy()
        compress_t_z()
    elif args.decompress and (not args.compress):
        print('Defaulting to decompressing the matrices')
        decompress_t_xy()
        decompress_t_z()
    else:

        parser.print_help()
        