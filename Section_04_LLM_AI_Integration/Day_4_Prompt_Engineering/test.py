from pathlib import Path


def read_file(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


if __name__ == "__main__":
    file_path = input("Enter the file path: ").strip()
    print(read_file(file_path))
