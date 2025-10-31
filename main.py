from argparse import ArgumentParser
from omegaconf import OmegaConf

from train import train_word_vectors, logger
from utils import clean_text, read_file, write_file

parser = ArgumentParser(description="Train Word Vectors")   
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the training configuration YAML file",
)
args = parser.parse_args()

config = OmegaConf.load(args.config)

# 中文语料（替换为完整zh.txt内容）
logger.info("Starting training for Chinese word vectors...")
text_zh = clean_text(read_file("data/zh.txt"))
zh_vectors = train_word_vectors(config, text_zh)

# 英文语料（替换为完整en.txt内容）
logger.info("Starting training for English word vectors...")
text_en = clean_text(read_file("data/en.txt"))
en_vectors = train_word_vectors(config, text_en)

# 保存到文件
write_file("word_vectors/zh_vectors.txt", zh_vectors)
write_file("word_vectors/en_vectors.txt", en_vectors)