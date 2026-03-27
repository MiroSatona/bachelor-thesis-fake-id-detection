import sys
import numpy
from pathlib import Path
from .base_extractor import BaseExtractor, DEFAULT_USE_GPU, TRUFOR_STR, TRUFOR_SRC, TRUFOR_CONFIG
from external.TruFor.TruFor_train_test.lib.config  import config as trufor_config, update_config   
from external.TruFor.TruFor_train_test.lib.models.cmx.builder_np_conf import EncoderDecoder 


class TruForExtractor(BaseExtractor):

    def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
        super().__init__(gpu)
        self.root_dir: Path = self.root_dir / TRUFOR_STR
        self.src_dir: Path = self.root_dir / TRUFOR_SRC
    
    def load_extractor(self) -> None:

        if self.extractor_model is not None:
            return

        if not self.src_dir.exists():
            raise RuntimeError(f"Cannot find TruFor code in {self.src_dir}.")
        
        update_config(trufor_config, str(TRUFOR_CONFIG))


        self.extractor_model = EncoderDecoder(cfg=trufor_config)
        self.extractor_model.to(self.device)
        self.extractor_model.eval()
        

