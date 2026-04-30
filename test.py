import time


def sendFile(fileName: str) -> str:
    time.sleep(1)
    print(f"{fileName}: finish")


def main():
    file = [f"file[{i}]" for i in range(10)]
    for i in range(10):
        sendFile(file[i])
    print("all finish")


main()
