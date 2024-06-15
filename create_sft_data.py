import pymysql

connect = pymysql.connect(user='root', password='David2002', db='baby_sft_data')
cur = connect.cursor()
cur.execute("use baby_sft_data;")


def insert_question(question):
    sql = f'insert into questions(text) values("{question}");'
    cur.execute(sql)
    connect.commit()


def insert_answer(qid, answer):
    sql = f'insert into answers(qid, text) values({qid}, "{answer}");'
    cur.execute(sql)
    connect.commit()


# insert_question("what's your name?")
# insert_answer(2, "My name is Xiao An.")
# insert_question("Which one is more difficult to train, YOLO or GPT?")

questions = [
    "How did your dad meet your mom?",
    "Do you like dogs or cats?",
    "I'm a bit sad, can you be with me?",
    "I love you.",
    "I don't feel well.",
    "When did you born?",
    "Who built you?",
    "I quit my course today, was I doing right?",
    "I think I fall in wove with a girl. What do you think I can do?",
]


