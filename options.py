class Options:
    def __init__(self, data_dir: str, peaq_test_dir: str, watermark_length: int, bit_per_seq: int, isCode: bool) -> None:
        self.data_dir = data_dir
        self.peaq_test_dir = peaq_test_dir
        self.watermark_length = watermark_length
        self.bit_per_seq = bit_per_seq
        self.isCode = isCode