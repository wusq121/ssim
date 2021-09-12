class Options:
    """
    data_dir: 被嵌入音频文件的位置
    peaq_test_dir: 转换成48000采样率用于计算peaq
    watermark_length: 嵌入的水印长度（01格式）
    bits_per_seq: 每个序列表示多少位水印，间接决定了序列的长度
    isCode: 对接展示系统，需要用到水印编码器
    seed: 随即种子
    """
    def __init__(
          self, 
          data_dir: str, 
          out_dir: str,
          peaq_test_dir: str, 
          watermark_length: int, 
          sigma: int, 
          bits_per_seq: int, 
          logfile: str,
          isCode: bool, 
          seed: int = 5) -> None:
        self.data_dir = data_dir
        self.out_dir = out_dir,
        self.peaq_test_dir = peaq_test_dir
        self.watermark_length = watermark_length
        self.bits_per_seq = bits_per_seq
        self.isCode = isCode
        self.seed = seed
        self.sigma = sigma
        self.logfile = logfile