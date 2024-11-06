#by Gianmarco Foroni aka hobbit-linux-42
#tanks to the creators of BLIP
from posixpath import sep
import sys
import logging
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
logging.basicConfig(level=logging.INFO, format='%(message)s')

if os.name == "posix":
    dir_sep="/"
elif os.name == "nt":
    dir_sep="\\"
else:
    logging.fatal("[!] unsupported os")
    exit(1)

#load model
model_path = "Salesforce/blip-image-captioning-base"
cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
logging.info("loading BLIP model ...")
processor = BlipProcessor.from_pretrained(model_path, cache_dir=cache_dir)
model = BlipForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
logging.info("loaded")

def get_caption(img_path):
    try:
        img = Image.open(img_path)
    except:
        return None
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def onlyfiles(path:str, recursive=False):
    files=[]
    if os.path.isdir(path):
        if list(path)[-1] != dir_sep:
            path += dir_sep
        for i in os.listdir(path):
            f = path+i
            if os.path.isfile(f):
                files.append(f)
            elif os.path.isdir(f) and recursive:
                files += onlyfiles(f, True)
    elif os.path.isfile(path):
        files = [path]
    else:
        logging.fatal("[!] invalid file path")
        exit(2)
    return files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        also_childs = not (input("also use child directorys? [Y/n]").upper() in ["N", "NO"]) and os.path.isdir(sys.argv[1])
        imgs_paths=onlyfiles(sys.argv[1], also_childs)
        REDUCE = not (input("reduce names? [Y/n] ").upper() in ["N", "NO"])
        blacklist = []
        if REDUCE:
            with open("blacklist.txt", "r") as blf:
                blacklist = blf.read().split("\n")
        new_imgs_paths = []

        for img_path in imgs_paths:
            caption = get_caption(img_path)
            if caption != None:
                new_name = ""
                for word in caption.split(" "):
                    if word not in blacklist:
                        new_name += word+"_"
                n=""
                if new_name in new_imgs_paths:
                    n=str(new_imgs_paths.count(new_name)+1)+"_"
                new_imgs_paths.append(new_name)
                new_name += n+"."+img_path.split(".")[1]
                new_path = img_path.replace(img_path.split(dir_sep)[-1], new_name)
                os.rename(img_path, new_path)
            else:
                logging.exception("[!] invalid image")
    else:
        logging.exception("provide image/directory path")
        exit(3)
