from datetime import datetime


def print_message(msg, newline=True):
    now = datetime.now()

    if newline:
        msg= msg + '\n'

    curr = '\r[{}] '.format(now.strftime("%H:%M:%S"))
    print(curr + msg, end="")


