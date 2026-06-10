
import os
import hashlib
import requests
import tarfile
from tqdm.auto import tqdm

DATASET_MAP = {
    "ASVspoof5_protocols.tar.gz": "865d0e894ea9f686f0f37e5ae3ae3616",
    
    "flac_T_aa.tar": "b0cc86b14826a7701b52aad4f53daf9c",
    "flac_T_ab.tar": "d05be3f4be7a343fdbdd0ed29fdff2e1",
    "flac_T_ac.tar": "70d1ba4ad75a20aef3dae541fbd321e3",
    "flac_T_ad.tar": "c9bb56af2cc410d98338d74babcb95c9",
    "flac_T_ae.tar": "a3969816982f3e52300d6147ad796df7",
    
    "flac_D_aa.tar": "df0be44957623991028cce59792beb17",
    "flac_D_ab.tar": "1e8cd685d89b64502692f1bcf1a13db3",
    "flac_D_ac.tar": "5e0031f08c30e4bdbf0c59f91b2d662b",
    
    "flac_E_aa.tar": "a8c800766f3d4ef87971e2b4f29663e2",
    "flac_E_ab.tar": "c35064188f54f07c87aba58de22534a0",
    "flac_E_ac.tar": "ed7cbcd1b2847998b72472ddf6b445e3",
    "flac_E_ad.tar": "626a080e994b05df49c577d7b3dede8d",
    "flac_E_ae.tar": "05f19a5e64fa556714ae1b38eb2ea70b",
    "flac_E_af.tar": "2ca2f52a3bbf827f7ec155ecb47e85d6",
    "flac_E_ag.tar": "671200b5de2fc1e74a563b07b057af8f",
    "flac_E_ah.tar": "eaf885b1299a61eb07f71e82e96dba37",
    "flac_E_ai.tar": "b2625683bf440abae2af901a45542ad9",
    "flac_E_aj.tar": "c50281c8233af6f3b0899604d21ade44",
}

def verify_md5(fname, expected_md5):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def download_and_extract(base_url, target_dir="ml/data/asvspoof5"):
    os.makedirs(target_dir, exist_ok=True)
    
    for filename, expected_hash in DATASET_MAP.items():
        file_path = os.path.join(target_dir, filename)
        url = f"{base_url}/{filename}" 
        
        if not os.path.exists(file_path):
            print(f"\n--- Downloading {filename} ---")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Проверка на 404/500 ошибки
                total = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            except Exception as e:
                print(f"❌ Ошибка скачивания {filename}: {e}")
                continue
        
        print(f"Verifying {filename}...")
        if not verify_md5(file_path, expected_hash):
            print(f"❌ MD5 mismatch for {filename}! Файл поврежден, удаляю...")
            os.remove(file_path)
            continue
        print(f"✅ MD5 OK")

        print(f"Extracting {filename}...")
        try:
            mode = "r:gz" if filename.endswith("gz") else "r:"
            with tarfile.open(file_path, mode) as tar:
                tar.extractall(target_dir)
            
            os.remove(file_path)
            print(f"Cleaned up {filename} (Archive deleted)")
        except Exception as e:
            print(f"❌ Ошибка распаковки {filename}: {e}")

BASE_URL = "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/main/" 
download_and_extract(BASE_URL)  