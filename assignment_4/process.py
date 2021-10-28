from constants import START, END


class ProcessData:
    def __init__(self, filename: str, N: int) -> None:
        self.file = filename
        self.N = N

    def __preprocess(self, line: str) -> str:
        line = line.replace("\n", "")
        if self.N != 1:
            line = START + " " + line + " " + END
        return line

    def modify_data(self) -> None:
        with open(self.file, "r") as file:
            all_lines = file.readlines()

        self.return_lines = []
        for line in all_lines:
            self.return_lines.append(self.__preprocess(line))

        with open("processed_data.txt", mode="w") as out_file:
            empty_line = START + "  " + END
            for line in self.return_lines:
                if line != empty_line:
                    out_file.write(line)
                    out_file.write("\n")

    def get_lines(self) -> list:
        return self.return_lines


if __name__ == "__main__":
    x = ProcessData("training_data.txt", 2)
    x.modify_data()
    y = x.get_lines()
