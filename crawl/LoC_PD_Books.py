import requests
import os
import time
from tqdm import tqdm


for i in tqdm(range(1, 137)):
    # 定义URL
    url = 'https://hf-mirror.com/datasets/storytracer/LoC-PD-Books/resolve/main/data/train_00{:03d}.parquet?download=true'.format(i)

    # 定义本地保存的文件路径
    local_filename = r'G:\txt_datasets\LoC_PD_Books\train_00{:03d}.parquet'.format(i)
    if os.path.exists(local_filename):
        print(local_filename, 'exists')
        continue

    try:
        # 发送GET请求获取文件内容
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # 如果请求返回的不是2xx状态码，抛出HTTPError异常
            with open(local_filename, 'wb') as f:  # 使用二进制写模式打开文件
                for chunk in r.iter_content(chunk_size=8192):  # 分块读取文件内容
                    if chunk:  # 过滤掉空chunk
                        f.write(chunk)  # 写入文件

        print(f"文件已成功保存到：{local_filename}")
        time.sleep(10)
    except:
        print(url, 'failed')
        time.sleep(30)
