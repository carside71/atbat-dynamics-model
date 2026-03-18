"""ロギングユーティリティ."""

import sys
from contextlib import contextmanager
from pathlib import Path


class TeeStream:
    """stdout/stderr を端末とファイルの両方に書き込むストリーム.

    tqdm 等の \\r による上書き更新はファイル側ではバッファリングし、
    最終状態のみを書き込むことでログの肥大化を防ぐ。
    """

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        self._line_buf = ""
        self._closed = False

    def write(self, data):
        self.stream.write(data)
        if self._closed:
            return
        # ファイル側: \r を考慮してバッファリング
        self._line_buf += data
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            # \r がある場合は最後の \r 以降だけ残す（上書き表現）
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            self.file.write(line + "\n")
            self.file.flush()
        # バッファ中に \r があれば先頭まで巻き戻す
        if "\r" in self._line_buf:
            self._line_buf = self._line_buf.rsplit("\r", 1)[-1]

    def flush(self):
        self.stream.flush()
        if not self._closed:
            self.file.flush()

    def close_log(self):
        """残バッファをフラッシュしてファイルを閉じる."""
        if self._closed:
            return
        if self._line_buf.strip():
            if "\r" in self._line_buf:
                self._line_buf = self._line_buf.rsplit("\r", 1)[-1]
            self.file.write(self._line_buf + "\n")
        self._line_buf = ""
        self.file.flush()
        self._closed = True

    def isatty(self):
        return self.stream.isatty()


@contextmanager
def tee_logging(log_path: Path):
    """stdout/stderr をファイルと端末の両方に出力するコンテキストマネージャ.

    Usage::

        with tee_logging(output_dir / "train.log"):
            print("This goes to both terminal and file")
    """
    with open(log_path, "w") as log_file:
        sys.stdout = TeeStream(log_file, sys.__stdout__)
        sys.stderr = TeeStream(log_file, sys.__stderr__)
        try:
            yield log_file
        finally:
            sys.stdout.close_log()
            sys.stderr.close_log()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
