import re
import cv2
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

app = FastAPI(title="Moikan OCR Service")

try:
    print("Initializing PaddleOCR...")
    # Disable angle classification to speed up scanning by 2-3x
    ocr = PaddleOCR(use_textline_orientation=False, lang='tr')
    print("PaddleOCR loaded successfully.")
except Exception as e:
    print(f"Error loading PaddleOCR: {e}")
    ocr = None


# Regex pattern to identify field labels on vehicle registration documents.
# These labels are excluded from being matched as extracted values.
_LABEL_RE = re.compile(
    r'''
    ^\s*\(                                      # Starts with open parenthesis
    | \bMARKASI\b | \bSOYADI\b | \bSOYAD\b
    | \bADRESİ\b  | \bADRESI\b
    | \bTESCİL\b  | \bTESCIL\b
    | \bTİPİ\b    | \bTIPİ\b    | \bTIPI\b
    | \bKİMLİK\b  | \bKIMLIK\b
    | \bVERGİ\b   | \bVERGI\b
    | \bSİLİNDİR\b| \bSILINDIR\b
    | \bMOTOR\s*GÜCÜ\b | \bMOTOR\s*GUCU\b
    | \bYAKIT\b
    | \bKULLANIM\b
    | \bARAÇ\s*SINIFI\b
    | \bMODEL\s*YILI\b
    | \bTİCARİ\s*ADI\b | \bTICARI\s*ADI\b
    | \bRENGİ\b | \bRENGI\b
    | \bAĞIRLIĞI\b | \bAGIRLIGI\b
    | \bKOLTUK\b | \bAYAKTA\b
    | \bNOTER\b
    | \bSATIŞ\b | \bSATIS\b
    | \bONAYLAYAN\b | \bVERİLDİĞİ\b
    | \bHAK\b.*\bMENFAAT\b
    | [A-Z]\.[0-9]\)         # e.g., D.1), C.1)
    | [A-Z]\.[0-9]\.[0-9]\)  # e.g., C.1.2)
    | \bP\.[0-9]\b            # e.g., P.1, P.5
    | [YZG]\.[0-9]            # e.g., Y.4, Z.1
    ''',
    re.IGNORECASE | re.VERBOSE
)


def is_field_label(line: str) -> bool:
    """Check if the text line is a registration document field label."""
    return bool(_LABEL_RE.search(line))


# Image preprocessing functions
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read image file.")
    h, w = img.shape[:2]
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif w < 1000:
        scale = 1200.0 / w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Contrast enhancement for blurry photos
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


# Field extraction logic
def extract_fields(text: str, lines: list) -> dict:

    def find_year_after(idx):
        for j in range(idx + 1, min(idx + 8, len(lines))):
            v = lines[j].strip()
            m = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', v)
            if m:
                return m.group(1)
        return ''

    def find_value_after(idx, stop_re=None, allow_skip_labels=False):
        for j in range(idx + 1, min(idx + 8, len(lines))):
            v = lines[j].strip()
            if not v:
                continue
            if stop_re and re.search(stop_re, v, re.IGNORECASE):
                break
            if re.match(r'^\(', v):
                if allow_skip_labels:
                    continue
                else:
                    break
            return v
        return ''

    # 1. Plate Number
    plaka = ''
    m = re.search(r'\b([0-9]{2}\s*[A-Z]{1,3}\s*[0-9]{2,4})\b', text, re.IGNORECASE)
    if m:
        plaka = re.sub(r'\s+', '', m.group(1)).upper()

    # 2. Chassis (VIN) Number
    sasi = ''
    for i, line in enumerate(lines):
        if re.search(r'ŞASE\s*N|SASE\s*N|\(E\)|E\)\s*S[AÂ]SE', line, re.IGNORECASE):
            # Inline detection: label and value on the same line
            inline = re.search(r'(?:ŞASE|SASE)\s*N[O0]?\s*([A-Z0-9]{15,20})', line, re.IGNORECASE)
            if inline:
                sasi = inline.group(1).strip().upper()
                break
            nxt = find_value_after(i, allow_skip_labels=True)
            if nxt and re.match(r'^[A-Z0-9]{15,20}$', nxt.strip(), re.IGNORECASE):
                sasi = nxt.strip().upper()
                break
    if not sasi:
        candidates = re.findall(r'\b([A-Z0-9]{17})\b', text, re.IGNORECASE)
        if candidates:
            sasi = candidates[0].upper()
    if sasi:
        sasi = sasi.replace('Q', '0')

    # 3. Engine Number
    motor = ''
    for i, line in enumerate(lines):
        if re.search(r'M[O0]T[O0]R\s*N|P\.?\s*5', line, re.IGNORECASE):
            # Case A: label and value on the same line (e.g. "P.5) MOTOR N350A10004375207")
            inline = re.search(
                r'(?:P[\.\s]*[5S][^\w]*)?M[O0]T[O0]R\s*N[O0]?\s*([A-Z0-9]{5,25})',
                line, re.IGNORECASE
            )

            if inline:
                candidate = inline.group(1).strip().upper()
                # Clean up leading 'N' from 'MOTOR NO' label if present
                candidate = re.sub(r'^N(?=[0-9A-Z]{4,})', '', candidate)
                if len(candidate) >= 5:
                    motor = candidate
                    break
            # Case B: value on the next line
            nxt = find_value_after(i, allow_skip_labels=True)
            if nxt and re.match(r'^[A-Z0-9]{5,}', nxt.strip(), re.IGNORECASE):
                motor = re.sub(r'\s+', '', nxt.strip()).upper()
                break
    if not motor:
        # Fallback search
        m = re.search(
            r'P[\.\s]*[5S][^\w]*(?:M[O0]T[O0]R\s*N[O0]?\s*)?([A-Z0-9]{5,25})',
            text, re.IGNORECASE
        )
        if m:
            motor = re.sub(r'\s+', '', m.group(1)).upper()
            motor = re.sub(r'^N(?=[0-9A-Z]{4,})', '', motor)

    # 4. Document Serial Number
    belge = ''
    for line in lines:
        m = re.search(r'[Ss]eri[:\s]*[A-Z]*\s*([0-9]{6})', line)
        if m:
            belge = m.group(1)
            break
    if not belge:
        m = re.search(r'(?:BELGE\s*SER[Iİ]|SER[Iİ]).{0,15}[A-Z]{0,3}\s*([0-9]{6})', text, re.IGNORECASE)
        if m:
            belge = m.group(1)
    if not belge:
        m = re.search(r'\bN[o0]?\s*([0-9]{6})\b', text, re.IGNORECASE)
        if m:
            belge = m.group(1)

    # 5. ID / Tax Number
    tc = ''
    m = re.search(r'\b([0-9]{10,11})\b', text)
    if m:
        tc = m.group(1)

    # 6. Model Year
    model_yili = ''
    for i, line in enumerate(lines):
        if re.search(r'D\.4|MODEL\s*YIL|MODED\s*YIL', line, re.IGNORECASE):
            inline = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', line)
            if inline:
                model_yili = inline.group(1)
            else:
                model_yili = find_year_after(i)
            if model_yili:
                break
    if not model_yili:
        clean_text = re.sub(r'(?:TIP\s*ONAY|MUA\.GEÇ|e9\s*\d{4}|2007/46)[^\n]*', '', text, flags=re.IGNORECASE)
        years = re.findall(r'\b(19[89]\d|20[0-2]\d)\b', clean_text)
        if years:
            model_yili = max(set(years), key=years.count)

    # 7. Brand / Make
    marka = ''
    for i, line in enumerate(lines):
        if re.search(r'D\.1|MARKASI', line, re.IGNORECASE):
            nxt = find_value_after(i, stop_re=r'D\.2|TİPİ|TIPI')
            if nxt and not is_field_label(nxt):
                clean = re.sub(r'[^A-ZÇĞİÖŞÜa-zçğışöü\s]', ' ', nxt).strip()
                clean = ' '.join(clean.split())
                if len(clean) >= 2:
                    marka = clean.upper()
            if marka:
                break
    if not marka:
        common = [
            'RENAULT', 'FIAT', 'VOLKSWAGEN', 'FORD', 'TOYOTA', 'HYUNDAI', 'PEUGEOT',
            'OPEL', 'CITROEN', 'HONDA', 'BMW', 'MERCEDES', 'AUDI', 'DACIA', 'SKODA',
            'SEAT', 'NISSAN', 'KIA', 'MAZDA', 'VOLVO', 'CHEVROLET', 'SUZUKI',
            'MITSUBISHI', 'ISUZU', 'TOFAS', 'TEMSA'
        ]
        for cb in common:
            if cb in text.upper():
                marka = cb
                break

    # 8. Model (Commercial Name)
    model = ''
    for i, line in enumerate(lines):
        if re.search(r'D\.3|T[İI]CAR[İI]\s*ADI|TICARIADI', line, re.IGNORECASE):
            # Try inline extraction first
            inline_model = re.search(
                r'(?:D\.3|T[İI]CAR[İI]\s*ADI)[)\s:]*'
                r'([A-ZÇĞİÖŞÜa-zçğışöü0-9][A-ZÇĞİÖŞÜa-zçğışöü0-9\s\-\.]{1,40})',
                line, re.IGNORECASE
            )

            if inline_model:
                cand = inline_model.group(1).strip()
                if len(cand) >= 2 and not is_field_label(cand):
                    model = re.sub(r'[^A-ZÇĞİÖŞÜa-zçğışöü0-9\s\-]', ' ', cand).strip()
                    model = ' '.join(model.split()).upper()
            if not model:
                for j in range(i + 1, min(i + 7, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if is_field_label(cand):
                        continue
                    if re.search(r'MAH\.|SK\.|NO:|İSTANBUL|ANKARA|İZMİR|ADRES|BAŞAKŞEHİR|BODRUM|MUĞLA', cand, re.IGNORECASE):
                        continue
                    if re.match(r'^\d{4}$', cand) or re.match(r'^[A-Z]$', cand):
                        continue
                    clean = re.sub(r'[^A-ZÇĞİÖŞÜa-zçğışöü0-9\s\-]', ' ', cand).strip()
                    clean = ' '.join(clean.split())
                    if clean and len(clean) >= 2:
                        model = clean.upper()
                        break
            if model:
                break

    if marka:
        marka = re.sub(r'^[^A-ZÇĞİÖŞÜa-zçğışöü0-9]+|[^A-ZÇĞİÖŞÜa-zçğışöü0-9]+$', '', marka)
    if model:
        model = re.sub(r'^[^A-ZÇĞİÖŞÜa-zçğışöü0-9]+|[^A-ZÇĞİÖŞÜa-zçğışöü0-9]+$', '', model)
        if model == marka:
            model = ''

    # 9. Brand Code
    marka_kodu = ''

    # 10. Vehicle Class / Usage Type
    use_type_id = ''
    cins_cand = ''
    # Try inline extraction on D.5 (CİNSİ) first
    for line in lines:
        cm2 = re.search(
            r'(?:D\.5|CİNS[İI]?)[^A-ZÇĞİÖŞÜ]*([A-ZÇĞİÖŞÜa-zçğışöü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğışöü]+)?)',
            line, re.IGNORECASE
        )
        if cm2:
            cins_cand = cm2.group(1).lower().strip()
            break
    if not cins_cand:
        cm = re.search(
            r'(?:CİNS[İI]?|ARAÇ\s*TİP[İI])[:\s.]*([A-Za-zÇĞİÖŞÜçğışöü0-9\s()+-]{3,30})',
            text, re.IGNORECASE
        )
        if cm:
            cins_cand = cm.group(1).lower()

    search_in = cins_cand if cins_cand else text.lower()

    use_map = [
        ('taksi', 2), ('minibüs', 3), ('minibus', 3), ('otobüs', 5), ('otobus', 5),
        ('kamyonet', 6), ('kamyon', 7), ('motosiklet', 11), ('çekici', 13), ('cekici', 13),
        ('traktör', 9), ('traktor', 9), ('tanker', 12), ('römork', 10), ('romork', 10),
        ('tarım', 15), ('tarim', 15), ('iş mak', 8), ('is mak', 8),
        ('özel amaç', 14), ('ozel amac', 14), ('otomobil', 1),
    ]
    for keyword, uid in use_map:
        if keyword in search_in:
            use_type_id = uid
            break

    # 11. Name & Surname
    NAME_NOISE = [
        'TICAR', 'TİCAR', 'TÍCAR', 'TCAR', 'UNVAN', 'ÜNVAN',
        'SOYAD', 'ADI', 'KIMLIK', 'TC', 'VERGI', 'PLAKA',
        'MARKAS', 'TİPİ', 'TIPI', 'MODEL', 'ARAÇ', 'MOTOR',
        'ŞASE', 'SASE', 'ADRES', 'NOTER', 'ONAY', 'BELGE',
        'TESCIL', 'TESCİL', 'BODRUM', 'ANKARA', 'İSTANBUL',
        'HUSUSI', 'NUSUSI', 'YOLCU', 'NAKLİ', 'NAKLI',
    ]

    def clean_name_words(raw_line):
        if is_field_label(raw_line):
            return []
        words = raw_line.split()
        clean_words = []
        for w in words:
            cw = re.sub(r'[^A-ZÇĞİÖŞÜa-zçğışöü]', '', w).upper()
            cw = cw.replace('Í', 'İ').replace('Î', 'İ').replace('Â', 'A')
            cw = re.sub(r'[^A-ZÇĞİÖŞÜ]', '', cw)
            if len(cw) <= 1:
                continue
            if any(nw in cw for nw in NAME_NOISE):
                continue
            clean_words.append(cw)
        return clean_words

    ad = ''
    soyad = ''

    for i, line in enumerate(lines):
        u = line.upper()

        # Surname label - skip if it looks like a commercial title
        if re.search(r'SOYAD|C\.L\.1|C\.1\.1', u) and not re.search(r'TİCARİ|TICARI|UNVAN|ÜNVAN', u):
            for j in range(i + 1, min(i + 8, len(lines))):
                cand = lines[j].strip()
                if is_field_label(cand):
                    break
                if re.search(r'\d|/', cand):
                    continue
                words = clean_name_words(cand)
                if words:
                    soyad = words[-1]
                    break

        # Name label
        if re.search(r'C\.1\.2', u) or (
            re.search(r'\bADI\b', u) and
            not re.search(r'SOYAD|TICARI|TİCARİ|TÍCARİ|TCARİ|NOTER|MARKAS', u)
        ):
            m3 = re.search(r'ADI\s+([A-ZÇĞİÖŞÜa-zçğışöü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğışöü]+)*)\s*$', line, re.IGNORECASE)
            if m3:
                words = clean_name_words(m3.group(1).strip())
                if words:
                    ad = ' '.join(words)
            if not ad:
                for j in range(i + 1, min(i + 6, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if is_field_label(cand):
                        break
                    if re.search(r'\d|/', cand):
                        continue
                    words = clean_name_words(cand)
                    if words:
                        ad = ' '.join(words)
                        break

    ad_soyad = f"{ad} {soyad}".strip()
    if ad_soyad:
        ad_soyad = re.sub(r'[^A-ZÇĞİÖŞÜ\s]', ' ', ad_soyad, flags=re.IGNORECASE)
        ad_soyad = ' '.join(ad_soyad.split())

    return {
        "car_number_plate": plaka,
        "car_sasi_no": sasi,
        "car_motor_no": motor,
        "car_file_number": belge,
        "consumer_id_number": tc,
        "consumer_name": ad_soyad,
        "car_model_year": model_yili,
        "car_brand_code": marka_kodu,
        "car_brand": marka,
        "car_model": model,
        "use_type_id": use_type_id
    }


# FastAPI Endpoints
@app.post("/ocr/ruhsat")
async def scan_ruhsat(ruhsat_photo: UploadFile = File(...)):
    if ocr is None:
        raise HTTPException(status_code=500, detail="OCR engine is not loaded.")

    try:
        t_start = time.time()
        contents = await ruhsat_photo.read()
        processed_img = preprocess_image(contents)
        ocr_results = ocr.ocr(processed_img)
        t_ocr = time.time()

        lines = []
        if ocr_results:
            for res in ocr_results:
                if res is None:
                    continue
                if isinstance(res, dict) and "rec_texts" in res:
                    lines.extend([str(t) for t in res["rec_texts"]])
                elif hasattr(res, "keys"):
                    try:
                        d = dict(res)
                        if "rec_texts" in d:
                            lines.extend([str(t) for t in d["rec_texts"]])
                    except Exception:
                        pass
                elif hasattr(res, "rec_texts"):
                    lines.extend([str(t) for t in res.rec_texts])
                elif isinstance(res, (list, tuple)):
                    for item in res:
                        try:
                            if isinstance(item, (list, tuple)) and len(item) > 1:
                                sub = item[1]
                                text_val = sub[0] if isinstance(sub, (list, tuple)) else sub
                                lines.append(str(text_val))
                        except Exception:
                            pass

        full_text = "\n".join(lines)
        parsed_data = extract_fields(full_text, lines)
        t_end = time.time()

        status = any([
            parsed_data["car_number_plate"],
            parsed_data["car_sasi_no"],
            parsed_data["consumer_id_number"]
        ])

        return JSONResponse(content={
            "status": status,
            "message": "Registration document successfully scanned." if status else "Could not read registration document fields. Please take a clearer photo.",
            "data": parsed_data,
            "raw_text": full_text
        })

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"status": False, "message": str(ve), "data": {}})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": False, "message": f"Server error: {str(e)}", "data": {}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ocr:app", host="127.0.0.1", port=8000, reload=False)
