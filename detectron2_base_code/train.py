import os
from config import get_configuration
from dataset import register_datasets
from trainer import MyTrainer

def main():
    register_datasets()
    cfg = get_configuration()
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()
