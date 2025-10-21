# app/core/pdfgen.py
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.units import inch

def render_pdf(report: dict, out_path: str):
    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER
    x = 1*inch
    y = height - 1*inch

    def write_line(text, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        lines = simpleSplit(text, "Helvetica", size, width - 2*inch)
        for line in lines:
            c.drawString(x, y, line)
            y -= leading
            if y < 1*inch:
                c.showPage()
                y = height - 1*inch

    write_line("AMRRA Research Report", size=16, leading=18)
    write_line("schema: " + str(report.get("schema","")))
    write_line("run_id: " + str(report.get("run_id","")))
    write_line("hash: " + str(report.get("report_hash","")))
    write_line(" ")

    req = report.get("request", {})
    write_line("Question: " + str(req.get("question","")))
    write_line("Mode: " + str(req.get("mode","")))
    write_line(" ")

    write_line("Sources:", size=13, leading=16)
    for s in (report.get("retrieval", {}).get("sources") or [])[:10]:
        title = s.get("title") or (s.get("source") or {}).get("name") or "Untitled"
        write_line(f"- {title}")
        if s.get("quote"):
            write_line(f"  > {s['quote'][:200]}")
    write_line(" ")

    c.showPage()
    c.save()