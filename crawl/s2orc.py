import json
import os
import re
import requests
import wget
from tqdm import tqdm
import time


API_KEY = "mQFznbO2Tz2IdDlgGK1pMao0FxIvqkNb3S9gepaV"
DATASET_NAME = "s2orc"
LOCAL_PATH = r"G:\txt_datasets\s2orc"
os.makedirs(LOCAL_PATH, exist_ok=True)


for i in range(100):
    response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
    RELEASE_ID = response["release_id"]
    print(f"最新版本ID: {RELEASE_ID}")


    have_loaded = []
    try:
        with open(r"./have_load_s2orc.json", 'r', encoding='utf-8') as f:
            have_loaded = json.load(f)
    except:
        print("no file")


    response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": API_KEY}).json()
    pths = response["files"]

    for url in tqdm(pths):

        match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
        assert match.group(1) == RELEASE_ID
        SHARD_ID = match.group(2)
        if os.path.exists(os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz")):
            print("url exists")
            continue
        print(url)
        try:
            wget.download(url, out=os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz"))
            print(os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz"))
            have_loaded.append(url)
            time.sleep(10)
        except:
            print("failed" + url)
            try:
                os.remove(os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz"))
            except:
                pass
            time.sleep(10)
            break
    print("下载了所有分片。")
    time.sleep(600)

