"""Offline pretrained TrOCR and classical image preprocessing for multi-line answers."""
import time
import re
import threading
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.common import ROOT, read_json
from src.models.scorer import device

def segment_lines(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width * image.height > 25_000_000:
        image.thumbnail((4000,4000))
    gray = np.asarray(ImageOps.grayscale(image))
    # Otsu works for colored pen and gray strokes; remove tiny connected artifacts.
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    clean = np.zeros_like(ink)
    for i in range(1,n):
        x,y,w,h,area = stats[i]
        if area >= max(6, gray.size * .000002) and not (w > gray.shape[1]*.8 and h < 4):
            clean[labels == i] = 255
    projection = (clean > 0).sum(axis=1).astype(float)
    smooth = np.convolve(projection, np.ones(9)/9, mode="same")
    active = smooth > max(2, gray.shape[1] * .0025)
    transitions = np.diff(np.r_[False, active, False].astype(int))
    runs = list(zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0]))
    merged = []
    for y0,y1 in runs:
        if merged and y0 - merged[-1][1] < max(5, image.height*.004):
            merged[-1][1] = int(y1)
        else:
            merged.append([int(y0),int(y1)])
    # Touching ascenders/descenders can join adjacent lines into one active run.
    # Split at projection valleys between well-separated line-density peaks.
    letter_heights = [h for x,y,w,h,area in stats[1:] if area >= 50 and 6 <= h < image.height*.2 and w >= 3]
    character_height = float(np.median(letter_heights)) if letter_heights else max(12,image.height*.025)
    density = gaussian_filter1d(projection, max(2,character_height*.2))
    peaks,_ = find_peaks(density,distance=max(12,int(character_height*1.5)),
                        prominence=max(2,float(density.max())*.15))
    if len(peaks):
        boundaries = [0] + [int(np.argmin(density[a:b])+a) for a,b in zip(peaks,peaks[1:])] + [image.height]
        merged = []
        for y0,y1 in zip(boundaries,boundaries[1:]):
            occupied = np.where((projection[y0:y1] > max(2,image.width*.0025)) &
                                (density[y0:y1] > max(2,float(density[y0:y1].max())*.05)))[0]
            if len(occupied):
                merged.append([int(y0+occupied[0]),int(y0+occupied[-1]+1)])
    crops, boxes = [], []
    for y0,y1 in merged:
        if y1-y0 < 6:
            continue
        yy,xx = np.where(clean[y0:y1] > 0)
        if len(xx) < 15:
            continue
        pad = max(8, int((y1-y0)*.12))
        box = [max(0,int(xx.min())-pad), max(0,y0-pad), min(image.width,int(xx.max())+pad+1), min(image.height,y1+pad)]
        crops.append(ImageOps.autocontrast(image.crop(tuple(box))))
        boxes.append(box)
    return crops, boxes

class HandwritingOCR:
    def __init__(self):
        self.config = read_json(ROOT / "configs/ocr.json")
        path = ROOT / self.config["pretrained"]
        self.device = device()
        self.processor = TrOCRProcessor.from_pretrained(path, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def recognize(self, image):
        start = time.perf_counter()
        image = Image.open(image) if isinstance(image, (str, __import__("pathlib").Path)) else image
        crops, boxes = segment_lines(image)
        warnings = []
        if not crops:
            return {"text":"", "lines":[], "boxes":[], "confidence":0., "seconds":time.perf_counter()-start, "warnings":["未检测到清晰手写文本，请重新拍摄或手动输入。"]}
        maximum_lines = self.config["max_lines"]
        if len(crops) > maximum_lines:
            warnings.append("文本行数过多，图片可能包含复杂版面，建议人工复核。")
            crops, boxes = crops[:maximum_lines], boxes[:maximum_lines]
        lines, confidences = [], []
        with self.lock:
            batch_size = self.config["line_batch_size"]
            for i in range(0,len(crops),batch_size):
                pixels = self.processor(images=crops[i:i+batch_size], return_tensors="pt").pixel_values.to(self.device)
                out = self.model.generate(pixels, max_new_tokens=self.config["max_new_tokens"], num_beams=self.config["num_beams"], return_dict_in_generate=True, output_scores=True)
                lines.extend(self.processor.batch_decode(out.sequences, skip_special_tokens=True))
                scores = self.model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
                for row in scores:
                    valid = row[row < 0]
                    confidences.append(float(valid.mean().exp()) if len(valid) else 0.)
        text = "\n".join(x.strip() for x in lines)
        confidence = float(np.mean(confidences))
        if confidence < self.config["warning_signal_threshold"]:
            warnings.append("OCR 识别信号偏低，请核对识别文本。")
        if len(re.findall(r"[A-Za-z]",text)) < 3 or not text.strip():
            warnings.append("未识别到有效英文答案，请人工复核。")
        if any(len(x.split()) >= 3 and len(set(x.lower().split())) < len(x.split())*.25 for x in lines):
            warnings.append("识别文本可能异常重复，请人工复核。")
        if any(len(self.processor.tokenizer.encode(x)) >= self.config["max_new_tokens"]-2 for x in lines):
            warnings.append("部分文字行可能被截断，请核对原图。")
        return {"text":text, "lines":lines, "boxes":boxes, "confidence":confidence,
                "seconds":time.perf_counter()-start, "warnings":warnings, "model":"microsoft/trocr-small-handwritten"}
