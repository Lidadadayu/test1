import logging
import os
import urllib
import urllib.request
from os.path import join as join
import numpy as np
from utils import is_int, cleanup_file

def download_progress(count, block_size, total_size):
    """
    回调函数，用于显示下载进度
    :param count: 当前已下载的块数
    :param block_size: 每个块的大小
    :param total_size: 文件的总大小
    """
    percent = int(count * block_size * 100 / total_size)
    if percent > 100:
        percent = 100
    print(f"\rDownload progress: {percent}%", end="")

def download_dataset_qm9(datadir, dataname):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    data_dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(data_dir, exist_ok=True)

    logging.info(
        'Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(data_dir))

    logging.info('Beginning download of GDB9 dataset!')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(data_dir, 'dsgdb9nsd.xyz.tar.bz2')

    try:
        urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data, reporthook=download_progress)
        print()  # 换行，使输出更美观
        logging.info('GDB9 dataset downloaded successfully!')
    except Exception as e:
        logging.error(f'Error downloading GDB9 dataset: {e}')
if __name__ == '__main__':
    datadir = 'data/raw_data'
    dataname = 'qm9'
    download_dataset_qm9(datadir, dataname)