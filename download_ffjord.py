from io import BytesIO
from urllib.request import urlopen
import tarfile
def download_targz(dl_path = "https://zenodo.org/record/1161203/files/data.tar.gz?download=1", path = "./"):
    resp = urlopen(dl_path)
    data = BytesIO(resp.read())
    tar = tarfile.open(fileobj=data, mode='r:gz')
    tar.extractall(path)
download_targz()
print('Done')