import logging
from api import app
from chroma import vector_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-16s %(levelname)-8s %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S"
)

vector_repo.init_db("chroma/kakao_reference/")

if __name__ == "__main__":
    pass
