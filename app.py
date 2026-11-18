"""لوحة تحكم NIDS - شغّلها بـ: streamlit run app.py
الملفات المطلوبة بجوارها: nids_model.joblib , demo_realistic.csv , demo_balanced.csv , demo_timeline.csv"""
import os, time, joblib
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ نظام كشف التسلل والصيانة التنبؤية للشبكات")
st.caption("نموذج XGBoost مدرَّب على CICIDS2017 | الفئات: Benign, DoS_DDoS, PortScan, BruteForce, Botnet | Macro-F1 = 0.9172")

SAMPLES = {
    "واقعية (نِسَب الشبكة الحقيقية)": "demo_realistic.csv",
    "متوازنة (عدد متساوٍ من كل فئة)": "demo_balanced.csv",
}
TIMELINE = "demo_timeline.csv"


@st.cache_resource
def load_model():
    art = joblib.load("nids_model.joblib")
    return art["model"], art["features"], list(art["label_encoder"].classes_)


@st.cache_data
def load_csv(fname: str):
    return pd.read_csv(fname, low_memory=False)


try:
    model, FEATURES, CLASSES = load_model()
except Exception as e:
    st.error(f"تعذّر تحميل النموذج: {e}")
    st.stop()


def pick_sample(key: str):
    """يعرض خيار نوع العيّنة ويعيد اسم الملف، أو None إن لم يوجد أي ملف."""
    avail = {k: v for k, v in SAMPLES.items() if os.path.exists(v)}
    if not avail:
        st.error("لا يوجد ملف عيّنة. ارفع demo_realistic.csv و demo_balanced.csv بجوار app.py")
        return None
    choice = st.radio("نوع العيّنة", list(avail), key=key, horizontal=True)
    if "واقعية" in choice:
        st.caption("نِسَب مطابقة لبيانات الاختبار: أغلب الحركة طبيعية والهجمات قليلة، كما في شبكة حقيقية.")
    else:
        st.caption("عدد متساوٍ من كل فئة لعرض جميع أنواع الهجمات؛ نسبة الهجمات هنا أعلى من الواقع عمداً.")
    return avail[choice]


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
    # مؤشر مبسّط لصحة الشبكة حسب نسبة الاتصالات المشبوهة في الدفعة
    if attack_ratio < 0.01:
        return "🟢 سليمة", "success"
    if attack_ratio < 0.10:
        return "🟠 تحذير", "warning"
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
        st.subheader("مقارنة مع التسميات الحقيقية")
        truth = res["Label"].astype(str).str.strip()
        st.write(f"الدقة على هذه العيّنة: **{(truth == res.Prediction).mean():.2%}**")
        labels = [c for c in CLASSES if c in set(truth) | set(res.Prediction)]
        cm = confusion_matrix(truth, res.Prediction, labels=labels)
        st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                                  labels=dict(x="Predicted", y="True"), title="Confusion Matrix"),
                        use_container_width=True)


def show_health_timeline(res: pd.DataFrame, threshold: float, w: int):
    """يحسب صحة الشبكة لكل نافذة متتابعة ويرسمها عبر الزمن."""
    r = res.reset_index(drop=True).copy()
    r["win"] = r.index // w
    g = r.groupby("win")
    agg = pd.DataFrame({
        "n": g.size(),
        "attack_ratio": g.apply(lambda x: float(((x.Prediction != "Benign") &
                                                 (x.Confidence >= threshold)).mean())),
    }).reset_index()
    agg = agg[agg.n >= w // 2].reset_index(drop=True)
    if agg.empty:
        st.warning(f"الملف أصغر من نافذة واحدة ({w} اتصال). قلّل حجم النافذة.")
        return
    agg["health"] = (100 * (1 - agg.attack_ratio)).round(2)
    # التنبؤ بقاعدة الاستمرار: صحة النافذة القادمة = صحة الحالية
    agg["forecast"] = agg.health.shift(1)

    sane = int((agg.health >= 99).sum())
    warn = int(((agg.health >= 90) & (agg.health < 99)).sum())
    bad = int((agg.health < 90).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد النوافذ", len(agg))
    c2.metric("🟢 سليمة", sane)
    c3.metric("🟠 تحذير", warn)
    c4.metric("🔴 خطرة", bad)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg.win, y=agg.health, mode="lines", name="صحة الشبكة"))
    fig.add_trace(go.Scatter(x=agg.win, y=agg.forecast, mode="lines", name="التنبؤ (قاعدة الاستمرار)",
                             line=dict(dash="dash")))
    fig.add_hline(y=99, line_dash="dot", line_color="orange")
    fig.add_hline(y=90, line_dash="dot", line_color="red")
    fig.update_layout(title=f"صحة الشبكة عبر الزمن (نافذة = {w} اتصال)",
                      xaxis_title="رقم النافذة", yaxis_title="درجة الصحة",
                      yaxis_range=[0, 102], height=430)
    st.plotly_chart(fig, use_container_width=True)

    ok = agg.dropna(subset=["forecast"])
    if len(ok):
        mae = (ok.health - ok.forecast).abs().mean()
        base = (ok.health - ok.health.mean()).abs().mean()
        st.caption(f"خطأ التنبؤ بقاعدة الاستمرار: {mae:.2f} نقطة، مقابل {base:.2f} لمتوسط ثابت. "
                   "التجارب أظهرت أن نماذج التنبؤ الأعقد لم تتفوق على هذه القاعدة البسيطة، "
                   "لأن تدهور الشبكة يحدث فجأة.")

    worst = agg.nsmallest(8, "health")[["win", "health", "n"]]
    worst.columns = ["النافذة", "الصحة", "عدد الاتصالات"]
    st.write("أسوأ النوافذ:")
    st.dataframe(worst, use_container_width=True)


st.sidebar.header("الإعدادات")
threshold = st.sidebar.slider("حدّ الثقة لإصدار تنبيه", 0.50, 0.99, 0.70, 0.01)
st.sidebar.info("ملاحظة: النموذج يكشف الهجمات الخمس المذكورة فقط، وأداؤه على Botnet ضعيف نسبياً "
                "(Recall ≈ 94% لكن Precision ≈ 49% في الاختبار).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📂 فحص ملف سجلات", "🧪 عيّنة تجريبية", "📈 صحة الشبكة عبر الزمن", "📡 محاكاة تدفق مباشر"])

with tab1:
    up = st.file_uploader("ارفع ملف CSV بنفس أعمدة CICIDS2017", type="csv")
    if up:
        show_results(analyze(pd.read_csv(up, low_memory=False)), threshold)

with tab2:
    fname = pick_sample("s2")
    if fname and st.button("حلّل العيّنة"):
        show_results(analyze(load_csv(fname)), threshold)

with tab3:
    st.write("يقسّم الحركة إلى نوافذ متتابعة ويحسب درجة صحة لكل نافذة، "
             "فيُظهر متى تدهورت الشبكة. هذا هو الجانب التنبؤي في المشروع.")
    w = st.slider("حجم النافذة (عدد الاتصالات)", 100, 2000, 500, 100)
    src = st.radio("مصدر البيانات", ["عيّنة زمنية محفوظة", "ارفع ملفاً"], horizontal=True, key="s3src")
    if src == "عيّنة زمنية محفوظة":
        if not os.path.exists(TIMELINE):
            st.error(f"الملف {TIMELINE} غير موجود. ارفعه بجوار app.py.")
        else:
            st.caption("اتصالات متتابعة بترتيبها الأصلي من بيانات الاختبار، لا عيّنة عشوائية.")
            if st.button("احسب المؤشر", key="b3a"):
                show_health_timeline(analyze(load_csv(TIMELINE)), threshold, w)
    else:
        up3 = st.file_uploader("ملف CSV مرتب زمنياً", type="csv", key="u3")
        if up3:
            show_health_timeline(analyze(pd.read_csv(up3, low_memory=False)), threshold, w)

with tab4:
    st.write("يعرض وصول اتصالات واحداً بعد الآخر (محاكاة من عيّنة محفوظة، وليست التقاطاً حقيقياً للحزم).")
    fname = pick_sample("s4")
    n = st.slider("عدد الاتصالات", 10, 100, 30)
    speed = st.slider("السرعة (ثانية بين اتصال وآخر)", 0.05, 1.0, 0.2)
    if fname and st.button("▶️ ابدأ"):
        s = load_csv(fname)
        demo = s.sample(min(n, len(s)), random_state=int(time.time()) % 1000).reset_index(drop=True)
        res = analyze(demo)
        feed, bar, stat = st.empty(), st.progress(0), st.empty()
        log, attacks = [], 0
        for i, r in res.iterrows():
            is_att = r.Prediction != "Benign" and r.Confidence >= threshold
            attacks += int(is_att)
            log.insert(0, f"{'🚨' if is_att else '✅'} #{i+1}  {r.Prediction}  (ثقة {r.Confidence:.0%})")
            feed.code("\n".join(log[:12]))
            stat.write(f"المشبوه حتى الآن: **{attacks}** من {i+1}")
            bar.progress((i + 1) / len(res))
            time.sleep(speed)
