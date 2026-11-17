"""لوحة تحكم NIDS - شغّلها بـ: streamlit run app.py
الملفات المطلوبة بجوارها: nids_model.joblib , demo_sample.csv"""
import time, joblib
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ نظام كشف التسلل في الشبكات (NIDS)")
st.caption("نموذج XGBoost مدرَّب على CICIDS2017 | الفئات: Benign, DoS_DDoS, PortScan, BruteForce, Botnet")

@st.cache_resource
def load_model():
    art = joblib.load("nids_model.joblib")
    model, le, feats = art["model"], art["label_encoder"], art["features"]
    return model, feats, list(le.classes_)

@st.cache_data
def load_demo():
    return pd.read_csv("demo_sample.csv")

try:
    model, FEATURES, CLASSES = load_model()
except Exception as e:
    st.error(f"تعذّر تحميل النموذج: {e}")
    st.stop()

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"ينقص الملف {len(missing)} عموداً، مثل: {missing[:5]}")
        st.stop()
    X = (df[FEATURES].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32"))
    proba = model.predict_proba(X)
    out = df.copy()
    out["Prediction"] = np.array(CLASSES)[proba.argmax(1)]
    out["Confidence"] = proba.max(1).round(4)
    return out

def health_status(attack_ratio: float):
    # مؤشر مبسّط لصحة الشبكة حسب نسبة الاتصالات المشبوهة في الدفعة (ليس تنبؤاً زمنياً)
    if attack_ratio < 0.01: return "🟢 سليمة", "success"
    if attack_ratio < 0.10: return "🟠 تحذير", "warning"
    return "🔴 خطرة", "error"

def show_results(res: pd.DataFrame, threshold: float):
    flagged = res[(res.Prediction != "Benign") & (res.Confidence >= threshold)]
    ratio = len(flagged) / max(len(res), 1)
    label, kind = health_status(ratio)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي السجلات", f"{len(res):,}")
    c2.metric("اتصالات مشبوهة", f"{len(flagged):,}")
    c3.metric("نسبة التهديد", f"{ratio:.2%}")
    c4.metric("حالة الشبكة", label)
    getattr(st, kind)(f"حالة الشبكة في هذه الدفعة: {label}")

    left, right = st.columns(2)
    counts = res.Prediction.value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    left.plotly_chart(px.bar(counts, x="Class", y="Count", color="Class", log_y=True,
                             title="توزيع التنبؤات (مقياس لوغاريتمي)"), use_container_width=True)
    right.plotly_chart(px.histogram(res, x="Confidence", color="Prediction", nbins=30,
                                    title="توزيع درجة الثقة"), use_container_width=True)

    st.subheader("التنبيهات")
    if len(flagged):
        show = ["Prediction", "Confidence"] + [c for c in ["Destination Port", "Flow Duration"] if c in res.columns]
        st.dataframe(flagged[show].sort_values("Confidence", ascending=False).head(500),
                     use_container_width=True)
    else:
        st.success("لا توجد تنبيهات.")

    if "Label" in res.columns:
        st.subheader("مقارنة مع التسميات الحقيقية (الملف يحتوي عمود Label)")
        truth = res["Label"].astype(str).str.strip()
        acc = (truth == res.Prediction).mean()
        st.write(f"الدقة على هذه العيّنة: **{acc:.2%}**")
        labels = [c for c in CLASSES if c in set(truth) | set(res.Prediction)]
        cm = confusion_matrix(truth, res.Prediction, labels=labels)
        st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                                  labels=dict(x="Predicted", y="True"), title="Confusion Matrix"),
                        use_container_width=True)

st.sidebar.header("الإعدادات")
threshold = st.sidebar.slider("حدّ الثقة لإصدار تنبيه", 0.50, 0.99, 0.70, 0.01)
st.sidebar.info("ملاحظة: النموذج يكشف الهجمات الخمس المذكورة فقط، وأداؤه على Botnet ضعيف نسبياً (Recall ≈ 49% في الاختبار).")

tab1, tab2, tab3 = st.tabs(["📂 فحص ملف سجلات", "🧪 عيّنة تجريبية", "📡 محاكاة تدفق مباشر"])

with tab1:
    up = st.file_uploader("ارفع ملف CSV بنفس أعمدة CICIDS2017", type="csv")
    if up:
        show_results(analyze(pd.read_csv(up, low_memory=False)), threshold)

with tab2:
    st.write("عيّنة صغيرة من بيانات الاختبار (مع التسميات الحقيقية للمقارنة).")
    if st.button("حلّل العيّنة التجريبية"):
        show_results(analyze(load_demo()), threshold)

with tab3:
    st.write("يعرض وصول اتصالات واحداً بعد الآخر من العيّنة التجريبية (محاكاة، وليست التقاطاً حقيقياً للحزم).")
    n = st.slider("عدد الاتصالات", 10, 100, 30)
    speed = st.slider("السرعة (ثانية بين اتصال وآخر)", 0.05, 1.0, 0.2)
    if st.button("▶️ ابدأ"):
        demo = load_demo().sample(n, random_state=int(time.time()) % 1000).reset_index(drop=True)
        res = analyze(demo)
        feed, bar, stat = st.empty(), st.progress(0), st.empty()
        log, attacks = [], 0
        for i, r in res.iterrows():
            is_att = r.Prediction != "Benign" and r.Confidence >= threshold
            attacks += int(is_att)
            log.insert(0, f"{'🚨' if is_att else '✅'} #{i+1}  {r.Prediction}  (ثقة {r.Confidence:.0%})")
            feed.code("\n".join(log[:12]))
            stat.write(f"المشبوه حتى الآن: **{attacks}** من {i+1}")
            bar.progress((i + 1) / n)
            time.sleep(speed)
