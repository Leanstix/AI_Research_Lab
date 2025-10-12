from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def render_pdf(report: dict, pdf_path: str):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    margin = 50
    x = margin
    y = h - margin

    def new_page():
        nonlocal y
        c.showPage()
        y = h - margin

    def write_heading(text, size=14):
        nonlocal y
        if y < 80: new_page()
        c.setFont("Helvetica-Bold", size)
        c.drawString(x, y, str(text))
        y -= 20

    def write_line(text, size=11, leading=14):
        nonlocal y
        if y < 60: new_page()
        c.setFont("Helvetica", size)
        c.drawString(x, y, str(text))
        y -= leading

    def wrap_text(text, width_chars=100, size=11, leading=14, bullet=None):
        """Very simple word wrap by character count (keeps it dependency-free)."""
        if text is None: return
        s = str(text)
        words = s.split()
        line = ""
        first = True
        for w_ in words:
            cand = w_ if not line else line + " " + w_
            if len(cand) <= width_chars:
                line = cand
            else:
                if first and bullet:
                    write_line(f"{bullet} {line}", size=size, leading=leading)
                    first = False
                else:
                    write_line(line, size=size, leading=leading)
                line = w_
        if line:
            if first and bullet:
                write_line(f"{bullet} {line}", size=size, leading=leading)
            else:
                write_line(line, size=size, leading=leading)

    def _short_hash(hv: str, n: int = 16) -> str:
        if not hv: return "N/A"
        hv = str(hv)
        return hv if len(hv) <= n else hv[:n] + "…"

    # ------------------------
    # Header
    # ------------------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "AMRRA — Research Report")
    y -= 28

    c.setFont("Helvetica", 11)
    run_id = report.get("run_id")
    ts = report.get("timestamp_unix")
    ts_iso = None
    try:
        if isinstance(ts, (int, float)):
            ts_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        ts_iso = None

    write_line(f"Run ID: {run_id}")
    write_line(f"Timestamp (UTC): {ts_iso or 'N/A'}")
    write_line(f"Report hash: {report.get('report_hash','N/A')}")

    # ------------------------
    # Request / Question
    # ------------------------
    write_heading("Request")
    q = report.get("request", {}).get("question", "N/A")
    wrap_text(f"Question: {q}", width_chars=95)

    # ------------------------
    # Hypotheses
    # ------------------------
    hyps = report.get("hypotheses") or []
    if hyps:
        write_heading("Hypotheses")
        for h_ in hyps[:6]:
            wrap_text(h_, width_chars=95, bullet="•")

    # ------------------------
    # Plan / Design
    # ------------------------
    plan = report.get("plan", {}) or {}
    write_heading("Plan / Design")
    design = plan.get("design", "N/A")
    write_line(f"Design: {design}")
    if "primary_metric" in plan:
        write_line(f"Primary metric: {plan.get('primary_metric')}")
    if "n" in plan:
        write_line(f"Sample size (n): {plan.get('n')}")
    if "alpha" in plan:
        write_line(f"Alpha: {plan.get('alpha')}")
    if "seed" in plan:
        write_line(f"Seed: {plan.get('seed')}")

    # ------------------------
    # Results (t-test or ML)
    # ------------------------
    res = report.get("results", {}) or {}

    # t-test branch
    if {"p", "effect_size", "ci"}.issubset(res.keys()):
        write_heading("Results (t-test)")
        write_line(f"p-value: {res.get('p')}")
        write_line(f"effect size: {res.get('effect_size')}")
        ci = res.get("ci") or []
        ci_txt = f"{ci[0]} to {ci[1]}" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "N/A"
        write_line(f"95% CI: {ci_txt}")

    # ML branch
    if "metrics" in res and "comparison" in res:
        mt = res.get("metrics", {})
        cmp_ = res.get("comparison", {})
        write_heading("Results (ML)")
        # Dataset summary (if present)
        ds = res.get("dataset", {})
        if ds:
            write_line(f"Dataset: {ds.get('name','N/A')}  |  "
                       f"samples={ds.get('n_samples','?')}, features={ds.get('n_features','?')}")

        # Metrics
        lr = mt.get("logreg", {})
        rf = mt.get("rf", {})
        write_line(f"LogReg — AUC: {lr.get('auc','?')}  Acc: {lr.get('accuracy','?')}  F1: {lr.get('f1','?')}")
        write_line(f"RandForest — AUC: {rf.get('auc','?')}  Acc: {rf.get('accuracy','?')}  F1: {rf.get('f1','?')}")

        # Confusion matrices (compact)
        cm_lr = lr.get("confusion_matrix")
        cm_rf = rf.get("confusion_matrix")
        if isinstance(cm_lr, list) and isinstance(cm_rf, list):
            write_line(f"Confusion (LR): {cm_lr}")
            write_line(f"Confusion (RF): {cm_rf}")

        # Comparison stats
        dauc = cmp_.get("delta_auc_rf_minus_lr")
        ci = cmp_.get("delta_auc_bootstrap_ci95", [])
        pperm = cmp_.get("permutation_test_p_greater")
        ci_txt = f"{ci[0]} to {ci[1]}" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "N/A"
        write_line(f"ΔAUC (RF - LR): {dauc}  |  CI95: {ci_txt}  |  Permutation p (RF>LR): {pperm}")

    # ------------------------
    # Retrieval / Sources (web + local)
    # ------------------------
    retr = (report.get("retrieval") or {}).get("sources") or []
    if retr:
        write_heading("Retrieval / Sources")
        for i, s in enumerate(retr[:6], start=1):
            title = s.get("title") or (s.get("source") or {}).get("name") or "Untitled"
            url = s.get("url") or (s.get("source") or {}).get("name")
            author = s.get("author")
            date = s.get("date")
            score = s.get("score")
            rank = s.get("rank")
            doc_hash = s.get("doc_hash")

            meta_bits = []
            if rank is not None: meta_bits.append(f"rank {rank}")
            if score is not None: meta_bits.append(f"score {round(score,4)}")
            if author: meta_bits.append(f"author {author}")
            if date: meta_bits.append(f"date {date}")
            meta_tail = ("  [" + "; ".join(meta_bits) + "]") if meta_bits else ""

            wrap_text(f"{i}. {title}{meta_tail}", width_chars=95)
            if url:
                wrap_text(f"URL: {url}", width_chars=95, size=9, leading=12)

            quote = s.get("quote")
            if quote:
                # normalize whitespace for nicer wrapping
                qnorm = " ".join(str(quote).split())
                wrap_text(f'“{qnorm}”', width_chars=95, size=10, leading=13)

            if doc_hash:
                write_line(f"doc_hash: {_short_hash(doc_hash)}", size=9, leading=12)

            y -= 6  # small spacer between sources

    # Source Summaries (LLM)
    summ = (report.get("retrieval") or {}).get("summaries") or []
    if summ:
        write_heading("Source Summaries")
        for i, it in enumerate(summ[:6], start=1):
            title = it.get("title") or "Untitled"
            url = it.get("url") or ""
            tldr = it.get("tldr") or ""
            wrap_text(f"{i}. {title}", width_chars=95)
            if url:
                wrap_text(f"URL: {url}", width_chars=95, size=9, leading=12)
            if tldr:
                wrap_text(f"TL;DR: {tldr}", width_chars=95, size=10, leading=13)
            y -= 6

    # ------------------------
    # Conclusion (LLM)
    # ------------------------
    conc = report.get("conclusion") or {}
    if conc.get("text"):
        write_heading("Conclusion")
        wrap_text(conc.get("text"), width_chars=95, size=11, leading=14)

        if conc.get("strength"):
            write_line(f"Strength of evidence: {conc['strength']}", size=11, leading=14)

        if isinstance(conc.get("limitations"), list) and conc["limitations"]:
            write_heading("Limitations", size=12)
            for item in conc["limitations"][:8]:
                wrap_text(f"• {item}", width_chars=95, size=10, leading=13)

        if isinstance(conc.get("next_steps"), list) and conc["next_steps"]:
            write_heading("Next steps", size=12)
            for item in conc["next_steps"][:8]:
                wrap_text(f"• {item}", width_chars=95, size=10, leading=13)

    # ------------------------
    # Environment
    # ------------------------
    env = report.get("environment", {}) or {}
    write_heading("Environment")
    write_line(f"Models: {env.get('models', {})}")
    write_line(f"Datasets: {env.get('datasets', [])}")
    code = env.get("code", {})
    if code:
        write_line(f"Code image: {code.get('image','N/A')}")

    # ------------------------
    # Artifacts (names only; no absolute paths)
    # ------------------------
    arts = report.get("artifacts", {}) or {}
    write_heading("Artifacts")
    # Support both new {json:{name}, pdf:{name}} and legacy {json_path:..., pdf_path:...}
    json_name = (arts.get("json") or {}).get("name") if isinstance(arts.get("json"), dict) else None
    pdf_name  = (arts.get("pdf") or {}).get("name") if isinstance(arts.get("pdf"), dict) else None
    legacy_json_path = arts.get("json_path")
    legacy_pdf_path  = arts.get("pdf_path")
    write_line(f"JSON: {json_name or legacy_json_path or 'N/A'}")
    write_line(f"PDF:  {pdf_name or legacy_pdf_path or 'N/A'}")

    # Footer
    y -= 8
    c.setFont("Helvetica-Oblique", 9)
    if y < 50: new_page()
    c.drawString(x, y, "Note: Report hash is anchored to Hedera Consensus Service for integrity verification.")
    y -= 14
    c.drawString(x, y, "This PDF is a human-readable view; the canonical JSON is the source of truth for hashing.")

    c.showPage()
    c.save()
