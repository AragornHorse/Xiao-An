import json


def json_to_jsonl(json_pth, jsonl_pth):
    # pth = r"D:\baby\sft_dataset\HealthCareMagic\HealthCareMagic-100k.json"

    with open(json_pth, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # with open(r"D:\baby\sft_dataset\HealthCareMagic\HealthCareMagic-100k.jsonl", 'a', encoding='utf-8') as f:
    with open(jsonl_pth, 'a', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


# json_to_jsonl()

def check_jsonl(jsonl_pth):
    with open(jsonl_pth, 'r', encoding='utf-8') as f:
        for line in f:
            print(line)
            print(json.loads(line))
            input(">>>")
        # break


def find_newline_offsets(filepath):
    newline_offsets = []
    with open(filepath, 'rb') as file:
        offset = 0
        while True:
            byte = file.read(1)
            if not byte:
                break

            if byte == b'\n':
                newline_offsets.append(offset)
            elif byte == b'\r':
                next_byte = file.read(1)
                if next_byte == b'\n':
                    newline_offsets.append(offset)
                    offset += 1
                else:
                    # file.write(next_byte)
                    pass

            offset += 1

    return newline_offsets


def get_menu(jsonl_pth, menu_pth):
    # filepath = r"D:\baby\sft_dataset\HealthCareMagic\HealthCareMagic-100k.jsonl"
    filepath = jsonl_pth
    offsets = find_newline_offsets(filepath)

    # with open(r"D:\baby\sft_dataset\HealthCareMagic\menu.json", 'w', encoding='utf-8') as f:
    with open(menu_pth, 'w', encoding='utf-8') as f:
        json.dump(offsets, f)


def dolly_format():
    pth = r"D:\baby\sft_dataset\databricks_dolly_15k\databricks-dolly-15k.jsonl"

    with open(pth, 'r', encoding='utf-8') as f:
        with open(r"D:\baby\sft_dataset\databricks_dolly_15k\format\databricks-dolly-15k.jsonl", 'a', encoding='utf-8') as tf:
            for line in f:
                line = json.loads(line)
                if len(line['context']) > 10:
                    tl = {
                        "input": f"\"{line['context']}\" {line['instruction']}",
                        "output": line['response']
                    }
                else:
                    tl = {
                        "input": line['instruction'],
                        "output": line['response']
                    }
                json.dump(tl, tf)
                tf.write('\n')


# dolly_format()
# check_jsonl(r"D:\baby\sft_dataset\databricks_dolly_15k\format\databricks-dolly-15k.jsonl")

# get_menu(r"D:\baby\sft_dataset\databricks_dolly_15k\format\databricks-dolly-15k.jsonl", r"D:\baby\sft_dataset\databricks_dolly_15k\format\menu.json")


def instruct_data_format():
    pth = r"D:\baby\sft_dataset\open_instruct\instruct_data.json"

    with open(pth, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(r"D:\baby\sft_dataset\open_instruct\format\train.jsonl", 'a', encoding='utf-8') as f:
        for line in data:
            # if len(line['input']) > 10:
            #     print("-" * 50)
            #     print(line['instruction'])
            #     print("-" * 10)
            #     print(line['input'])
            #     print("-" * 10)
            #     print(line['output'])
            if len(line['input']) > 10:
                tl = {
                    "input": f"\"{line['input']}\" {line['instruction']}",
                    "output": f"{line['output']}"
                }
            else:
                tl = {
                    "input": f"{line['instruction']}",
                    "output": f"{line['output']}"
                }
            json.dump(tl, f)
            f.write("\n")


# check_jsonl(r"D:\baby\sft_dataset\open_instruct\format\train.jsonl")
# get_menu(r"D:\baby\sft_dataset\open_instruct\format\train.jsonl", r"D:\baby\sft_dataset\open_instruct\format\menu.json")

def format_open_hermes():
    import pyarrow.parquet as pq
    pth = r"G:\txt_datasets\sft_dataset\OpenHermes2.5\train-00003-of-00004.parquet"

    now_role = "user"
    now_txt = ""

    rst = {}

    with open(r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00003\train.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read(['messages'])[0]
            for chat in data:
                for line in chat:
                    content = str(line[0])
                    role = str(line[1])

                    if role == "system":
                        role = "user"

                    if role == now_role:
                        now_txt += " " + content
                    else:
                        if now_role == "user":
                            rst['input'] = now_txt
                        else:
                            rst['output'] = now_txt
                            json.dump(rst, tf)
                            tf.write("\n")
                            rst = {}
                        now_role = role
                        now_txt = content

    get_menu(r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00003\train.jsonl", r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00003\menu.json")


def financial_instruct_format():
    import pyarrow.parquet as pq
    pth = r"G:\txt_datasets\sft_dataset\financial_instruction_aq22\train-00000-of-00001.parquet"


    with open(r"G:\txt_datasets\sft_dataset\financial_instruction_aq22\format\train.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read()
            for i in range(len(data)):
                instruct = str(data[0][i])
                output = str(data[1][i])

                line = {
                    "input": instruct,
                    "output": output
                }

                json.dump(line, tf)
                tf.write("\n")

    get_menu(
        r"G:\txt_datasets\sft_dataset\financial_instruction_aq22\format\train.jsonl",
        r"G:\txt_datasets\sft_dataset\financial_instruction_aq22\format\menu.json"
    )


def deita10k_format():
    pth = r"G:\txt_datasets\sft_dataset\deita10k\test_gen-00000-of-00001.parquet"

    import pyarrow.parquet as pq

    now_role = "user"
    now_txt = ""

    rst = {}

    with open(r"G:\txt_datasets\sft_dataset\deita10k\test_1\test.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read()
            data = data[2]

            for chat in data:
                for line in chat:
                    content = str(line[0])
                    role = str(line[1])

                    if role == "system":
                        role = "user"

                    if role == now_role:
                        now_txt += " " + content
                    else:
                        if now_role == "user":
                            rst['input'] = now_txt
                        else:
                            rst['output'] = now_txt
                            json.dump(rst, tf)
                            tf.write("\n")
                            rst = {}
                        now_role = role
                        now_txt = content

    get_menu(r"G:\txt_datasets\sft_dataset\deita10k\test_1\test.jsonl", r"G:\txt_datasets\sft_dataset\deita10k\test_1\menu.json")


def auto_cot_format():
    import pyarrow.parquet as pq

    pth = r"G:\txt_datasets\sft_dataset\auto_CoT\validation-00000-of-00001-4ad407f38924987c.parquet"

    with open(r"G:\txt_datasets\sft_dataset\auto_CoT\val\val.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read()

            instruction = data[0]
            ipt = data[1]
            opt = data[2]

            for i in range(len(data)):

                ip = f"{str(instruction[i])} {str(ipt[i])}"
                op = str(opt[i])

                line = {
                    "input": ip,
                    "output": op
                }

                json.dump(line, tf)
                tf.write('\n')

    get_menu(r"G:\txt_datasets\sft_dataset\auto_CoT\val\val.jsonl", r"G:\txt_datasets\sft_dataset\auto_CoT\val\menu.json")


def chatbot_format():
    pth = r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\test-00000-of-00001-8d5da67d5c6856ed.parquet"
    import pyarrow.parquet as pq


    with open(r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\test\test.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read()
            ipt = data[1]
            opt = data[0]

            for i in range(len(data)):
                line = {
                    'input': str(ipt[i]),
                    'output': str(opt[i])
                }

                json.dump(line, tf)
                tf.write('\n')

    get_menu(r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\test\test.jsonl", r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\test\menu.json")


def cot_reformatted_format():
    pth = r"G:\txt_datasets\sft_dataset\CoT_reformatted\train-00004-of-00005-68e17edb2f8829d4.parquet"

    import pyarrow.parquet as pq


    with open(r"G:\txt_datasets\sft_dataset\CoT_reformatted\00004\train.jsonl", 'a', encoding='utf-8') as tf:
        with pq.ParquetFile(pth) as pf:
            data = pf.read()

            instruction = data[0]
            ipt = data[1]
            opt = data[2]

            for i in range(len(data)):
                line = {
                    "input": f"{str(instruction[i])} {str(ipt[i])}",
                    "output": f"{str(opt[i])}"
                }
                json.dump(line, tf)
                tf.write('\n')

    get_menu(r"G:\txt_datasets\sft_dataset\CoT_reformatted\00004\train.jsonl", r"G:\txt_datasets\sft_dataset\CoT_reformatted\00004\menu.json")


def tofu_format():
    from glob import glob

    pths = glob(r"G:\txt_datasets\sft_dataset\TOFU\*.json")


    with open(r"G:\txt_datasets\sft_dataset\TOFU\format\train.jsonl", 'a', encoding='utf-8') as tf:
        for pth in pths:
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    line = {
                        "input": line['question'],
                        "output": line['answer']
                    }
                    json.dump(line, tf)
                    tf.write('\n')

    get_menu(r"G:\txt_datasets\sft_dataset\TOFU\format\train.jsonl", r"G:\txt_datasets\sft_dataset\TOFU\format\menu.json")


# get_menu(r"G:\txt_datasets\sft_dataset\chat\train.jsonl", r"G:\txt_datasets\sft_dataset\chat\menu.json")
