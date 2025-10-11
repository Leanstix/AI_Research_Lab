from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def render_pdf(report: dict, pdf_path: str):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    x, y = 50, h - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "AMRRA — Research Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Run ID: {report.get('run_id')}")
    y -= 18
    c.drawString(x, y, f"Question: {report.get('request',{}).get('question','N/A')}")
    y -= 18
    c.drawString(x, y, f"Design: {report.get('plan',{}).get('design','N/A')}")
    y -= 26

    res = report.get("results", {})
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Results:")
    y -= 18; c.setFont("Helvetica", 11)
    c.drawString(x, y, f"p-value: {res.get('p')}")
    y -= 16
    c.drawString(x, y, f"effect size: {res.get('effect_size')}")
    y -= 16
    ci = res.get("ci", [])
    c.drawString(x, y, f"CI: {ci[0]} to {ci[1]}" if len(ci)==2 else "CI: N/A")
    y -= 26

    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Environment:")
    y -= 18; c.setFont("Helvetica", 11)
    c.drawString(x, y, f"models: {report.get('environment',{}).get('models',{})}")
    y -= 16
    c.drawString(x, y, f"datasets: {report.get('environment',{}).get('datasets',[])}")
    y -= 26

    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Artifacts:")
    y -= 18; c.setFont("Helvetica", 11)
    c.drawString(x, y, f"JSON: {report.get('artifacts',{}).get('json_path','local')}")

    c.showPage(); c.save()
