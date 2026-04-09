# ============================================================
#  POSTCARGO — Renombrador de Guías  |  App Web Streamlit
#  Despliega gratis en streamlit.io
# ============================================================

# ── requirements.txt (archivo separado en el repo) ──────────
# streamlit
# pyzbar
# Pillow
# opencv-python-headless
# google-genai
# libzbar0  ← en Linux se instala vía packages.txt

# ── app.py ──────────────────────────────────────────────────
import io, re, shutil, zipfile, base64, threading, tempfile
import numpy as np
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pyzbar import pyzbar
from google import genai

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title = "Postcargo · Renombrador de Guías",
    page_icon  = "📦",
    layout     = "centered"
)

# ── CSS personalizado ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .pc-header   { background:#0f1923; border-radius:16px; padding:28px 32px;
                 display:flex; align-items:center; gap:20px; margin-bottom:24px; }
  .pc-logo     { width:48px; height:48px; background:#00e5a0; border-radius:12px;
                 display:flex; align-items:center; justify-content:center;
                 font-size:24px; flex-shrink:0; }
  .pc-title    { color:#fff; font-size:22px; font-weight:600;
                 margin:0; letter-spacing:-0.3px; }
  .pc-subtitle { color:#7a8a99; font-size:13px; margin:4px 0 0; }

  .stat-grid   { display:grid; grid-template-columns:repeat(4,1fr);
                 gap:10px; margin:16px 0; }
  .stat-box    { background:#f7f9fb; border-radius:10px; padding:14px 16px; }
  .stat-val    { font-size:26px; font-weight:600; line-height:1;
                 margin-bottom:4px; font-family:'DM Mono',monospace; }
  .stat-lbl    { font-size:11px; color:#7a8a99; font-weight:500; }
  .blue .stat-val  { color:#2563eb; }
  .green .stat-val { color:#00a86b; }
  .amber .stat-val { color:#e08c00; }
  .red .stat-val   { color:#d94040; }

  .badge-row   { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }
  .badge       { padding:6px 12px; border-radius:8px; font-size:12px;
                 font-weight:500; }
  .badge.teal  { background:#e6faf3; color:#007a52; }
  .badge.blue  { background:#eef3ff; color:#2563eb; }
  .badge.gray  { background:#f0f3f6; color:#5a6575; }

  .dl-box      { background:#0f1923; border-radius:12px; padding:16px 20px;
                 display:flex; align-items:center; gap:12px; margin-top:16px; }
  .dl-text     { color:#fff; font-size:14px; font-weight:500; }
  .dl-sub      { color:#7a8a99; font-size:12px; }

  div[data-testid="stButton"] > button {
    width:100%; height:48px; background:#0f1923; color:#00e5a0;
    border:none; border-radius:12px; font-size:15px; font-weight:600;
    cursor:pointer; margin-top:12px;
    transition: opacity 0.2s;
  }
  div[data-testid="stButton"] > button:hover { opacity:0.85; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="pc-header">
  <div class="pc-logo">📦</div>
  <div>
    <p class="pc-title">Postcargo · Renombrador de Guías</p>
    <p class="pc-subtitle">Procesamiento automático con código de barras e IA</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── API Key (se guarda en secrets de Streamlit en producción) ─
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
client         = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL   = "gemini-2.0-flash"
MAX_WORKERS    = 5
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

GEMINI_PROMPT = """
Analiza esta imagen de una guía de envío o etiqueta logística.
Extrae el NÚMERO DE GUÍA o NÚMERO DE SEGUIMIENTO.
Reglas:
- Busca códigos de barras, QR, o cualquier número prominente.
- El número suele tener entre 8 y 25 caracteres alfanuméricos.
- Devuelve ÚNICAMENTE el número, sin espacios ni explicaciones.
- Si no puedes identificarlo, responde exactamente: NO_DETECTADO
"""

# ── Funciones de procesamiento ───────────────────────────────

def preprocesar_imagen(imagen):
    variantes = []
    img_np    = np.array(imagen.convert("RGB"))
    gris      = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    variantes.append(Image.fromarray(gris))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    variantes.append(Image.fromarray(clahe.apply(gris)))
    variantes.append(Image.fromarray(cv2.adaptiveThreshold(
        gris,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)))
    variantes.append(Image.fromarray(cv2.adaptiveThreshold(
        gris,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)))
    w, h = imagen.size
    variantes.append(imagen.resize((w*2,h*2), Image.LANCZOS))
    gris_pil = Image.fromarray(gris)
    variantes.append(gris_pil.resize((w*2,h*2), Image.LANCZOS))
    blur = cv2.GaussianBlur(gris,(0,0),3)
    variantes.append(Image.fromarray(cv2.addWeighted(gris,1.5,blur,-0.5,0)))
    for ang in [90,180,270]:
        variantes.append(gris_pil.rotate(ang, expand=True))
    return variantes

def leer_pyzbar(imagen):
    for v in preprocesar_imagen(imagen):
        res = pyzbar.decode(v)
        if res:
            dato = res[0].data
            if isinstance(dato, bytes):
                dato = dato.decode("utf-8", errors="ignore")
            dato = dato.strip()
            if dato:
                return dato
    return None

def leer_gemini(imagen):
    buf = io.BytesIO()
    fmt = imagen.format or "PNG"
    imagen.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = {"PNG":"image/png","JPEG":"image/jpeg",
            "JPG":"image/jpeg","WEBP":"image/webp"}.get(fmt.upper(),"image/png")
    try:
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"parts":[
                {"inline_data":{"mime_type":mime,"data":b64}},
                {"text":GEMINI_PROMPT}
            ]}]
        )
        t = r.text.strip()
        if t and t.upper() != "NO_DETECTADO":
            n = re.sub(r"[^A-Za-z0-9\-]","",t)
            if len(n) >= 5:
                return n
    except Exception:
        pass
    return None

def sanitizar(nombre):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]',"_",nombre).strip()

def procesar_una(args):
    ruta, out_dir, no_det_dir = args
    res = {"archivo": ruta.name, "metodo": None, "numero": None}
    try:
        img = Image.open(ruta)
        img.load()
        ext = ruta.suffix.lower()

        numero = leer_pyzbar(img)
        if numero:
            res["metodo"] = "pyzbar"
        else:
            numero = leer_gemini(img)
            if numero:
                res["metodo"] = "gemini"

        if numero:
            res["numero"]  = sanitizar(numero)
            nombre_final   = f"{res['numero']}{ext}"
            destino        = out_dir / nombre_final
            c = 1
            while destino.exists():
                destino = out_dir / f"{res['numero']}_{c}{ext}"
                c += 1
            shutil.copy2(ruta, destino)
        else:
            res["metodo"] = "no_detectado"
            shutil.copy2(ruta, no_det_dir / ruta.name)
    except Exception as e:
        res["metodo"] = "error"
    return res

# ── UI principal ─────────────────────────────────────────────

archivo = st.file_uploader(
    "Sube tu archivo ZIP con las guías de envío",
    type=["zip"],
    help="El ZIP debe contener imágenes PNG o JPG de las guías."
)

if archivo is not None:
    if st.button("🚀  Iniciar procesamiento"):

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp      = Path(tmpdir)
            out_dir  = tmp / "renombradas"
            no_det   = tmp / "renombradas" / "NO_DETECTADOS"
            ext_dir  = tmp / "extraidas"
            out_dir.mkdir(parents=True)
            no_det.mkdir(parents=True)
            ext_dir.mkdir(parents=True)

            # Extracción
            st.info("📂 Extrayendo imágenes del ZIP…")
            with zipfile.ZipFile(io.BytesIO(archivo.read())) as zf:
                zf.extractall(ext_dir)

            imagenes = [
                p for p in ext_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            total = len(imagenes)

            if total == 0:
                st.error("No se encontraron imágenes válidas en el ZIP.")
                st.stop()

            # Procesamiento con barra de progreso
            st.write(f"🖼️ **{total}** imágenes encontradas. Procesando…")
            barra    = st.progress(0)
            estado   = st.empty()
            contadores = {"pyzbar":0,"gemini":0,"no_det":0,"err":0,"done":0}
            lock     = threading.Lock()

            args_list = [(img, out_dir, no_det) for img in imagenes]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futuros = {ex.submit(procesar_una, a): a for a in args_list}
                for futuro in as_completed(futuros):
                    res = futuro.result()
                    with lock:
                        contadores["done"] += 1
                        m = res.get("metodo")
                        if   m == "pyzbar":                   contadores["pyzbar"] += 1
                        elif m == "gemini":                   contadores["gemini"] += 1
                        elif m in ("no_det","no_detectado"):  contadores["no_det"] += 1
                        else:                                 contadores["err"]    += 1
                        done = contadores["done"]
                        det  = contadores["pyzbar"] + contadores["gemini"]

                    barra.progress(done / total)
                    estado.markdown(
                        f"**{done}/{total}** completadas · "
                        f"✅ {det} detectadas · "
                        f"❓ {contadores['no_det']} no detectadas"
                    )

            # Comprimir resultado
            st.info("📦 Generando ZIP de resultados…")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in out_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(out_dir))
            zip_buf.seek(0)

            # Reporte final
            det_total    = contadores["pyzbar"] + contadores["gemini"]
            no_det_total = contadores["no_det"]
            err_total    = contadores["err"]
            pct          = int((det_total / total) * 100) if total else 0

            st.success("✅ ¡Procesamiento completado!")
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-box blue">
                <div class="stat-val">{total}</div>
                <div class="stat-lbl">Total procesadas</div>
              </div>
              <div class="stat-box green">
                <div class="stat-val">{det_total}</div>
                <div class="stat-lbl">Detectadas ✓</div>
              </div>
              <div class="stat-box amber">
                <div class="stat-val">{no_det_total}</div>
                <div class="stat-lbl">No detectadas</div>
              </div>
              <div class="stat-box {'red' if err_total else ''}">
                <div class="stat-val">{err_total}</div>
                <div class="stat-lbl">Errores</div>
              </div>
            </div>
            <div class="badge-row">
              <span class="badge teal">📦 Código de barras: {contadores['pyzbar']}</span>
              <span class="badge blue">🤖 IA Gemini: {contadores['gemini']}</span>
              <span class="badge gray">❓ No detectadas: {no_det_total}</span>
            </div>
            """, unsafe_allow_html=True)

            st.progress(pct / 100)
            st.caption(f"Tasa de detección: {pct}%")

            # Botón de descarga nativo de Streamlit
            st.download_button(
                label     = "⬇️  Descargar guías renombradas (.zip)",
                data      = zip_buf,
                file_name = "postcargo_guias_renombradas.zip",
                mime      = "application/zip",
                use_container_width = True
            )
