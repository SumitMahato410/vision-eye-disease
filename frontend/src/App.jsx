import { useEffect, useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [symptoms, setSymptoms] = useState([]);
  const [extra, setExtra] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("https://vision-eye-disease-ai2s.vercel.app/api/symptoms")
    .then(r => r.json())
    .then(setSymptoms)
      .catch(() => setError("Cannot reach backend. Make sure `python app.py` is running on port 5000."));
  }, []);

  const onFile = (e) => {
    const f = e.target.files[0];
    setResult(null); setError(""); setFile(f || null);
    setPreview(f ? URL.createObjectURL(f) : "");
  };
  const toggle = (k) => setExtra(s => s.includes(k) ? s.filter(x => x !== k) : [...s, k]);

  const predict = async () => {
    setError(""); setLoading(true); setResult(null);
    const fd = new FormData();
    fd.append("image", file); fd.append("symptoms", extra.join(","));
    try {
      const r = await fetch(
    "https://vision-eye-disease-ai2s.vercel.app/api/predict",
    {
        method: "POST",
        body: fd
    }
);
      const data = await r.json();
      if (!r.ok) setError(data.error || "Something went wrong."); else setResult(data);
    } catch { setError("Cannot reach backend. Is it running?"); }
    setLoading(false);
  };

  const im = result?.image;
  return (
    <div className="page">
      <header><h1>👁️ VISION</h1><p>Upload an eye photo → get predicted eye condition &amp; prevention</p></header>
      <main>
        <section className="card">
          <h2>1. Upload a close-up photo of one eye</h2>
          <p style={{ margin: "0 0 8px", color: "#475569" }}>Tip: eye open, good daylight, no flash, eye fills most of the photo.</p>
          <input type="file" accept="image/*" onChange={onFile} />
          {preview && <img className="preview" src={preview} alt="your eye" />}
          <details>
            <summary>Optional: add other symptoms for a better prediction</summary>
            <div className="grid small">
              {symptoms.map(s => (
                <label key={s.key} className={"chip" + (extra.includes(s.key) ? " on" : "")}>
                  <input type="checkbox" checked={extra.includes(s.key)} onChange={() => toggle(s.key)} /> {s.label}
                </label>
              ))}
            </div>
          </details>
          <div className="actions">
            <button className="primary" onClick={predict} disabled={!file || loading}>{loading ? "Analysing..." : "Predict Disease"}</button>
          </div>
          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className="card result">
            <h2>2. Result</h2>
            <div className={"banner " + result.urgency}>
              {result.urgency === "high" ? "⚠️ Please see an eye doctor soon" :
               result.urgency === "medium" ? "🩺 A doctor visit is recommended" : "✅ Usually mild - follow the prevention tips"}
            </div>
            <div className="two">
              <figure><img className="preview" src={im.overlay} alt="analysis" /><figcaption>Detected red areas highlighted in green</figcaption></figure>
              <div>
                <p><b>Image findings:</b></p>
                <p>Eye appearance: <b>{im.redness_level === "None" ? "Clear (no redness)" : "Red eye"}</b></p>
                <p>Redness: <b>{im.redness_percent}%</b> ({im.redness_level})</p>
                {im.yellowing && <p className="warn">Yellow tint detected - this can be a sign of jaundice. Please see a doctor.</p>}
              </div>
            </div>
            <h3>{result.top.disease}</h3>
            <div className="bar"><div style={{ width: result.top.confidence + "%" }} /></div>
            <p className="conf">Confidence: <b>{result.top.confidence}%</b></p>
            {result.uncertain && <p className="warn">Low confidence - add other symptoms (swelling, pain, discharge...) and predict again.</p>}
            <p>{result.description}</p>
            <h4>Other possibilities</h4>
            <ul className="plain">{result.others.map(o => <li key={o.disease}>{o.disease} — {o.confidence}%</li>)}</ul>
            <h4>Prevention &amp; care tips</h4>
            <ul>{result.prevention.map((p, i) => <li key={i}>{p}</li>)}</ul>
            <p className="disclaimer">{result.disclaimer}</p>
          </section>
        )}
      </main>
      <footer>Medicaps University · B.Tech CSE Project-1</footer>
    </div>
  );
}
