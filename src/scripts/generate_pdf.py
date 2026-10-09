import os
import sys
import markdown
from xhtml2pdf import pisa

def convert_md_to_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        print(f"Erro: Arquivo {md_path} não encontrado.")
        sys.exit(1)
        
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    # Converter Markdown para HTML com suporte a tabelas e blocos de código
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])
    
    # Template HTML com CSS estilizado para relatório acadêmico
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4 portrait;
                margin: 2cm;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #2c3e50;
            }}
            h1 {{
                font-size: 18pt;
                color: #1a252f;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 6px;
                margin-top: 0;
            }}
            h2 {{
                font-size: 13pt;
                color: #2c3e50;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 4px;
                margin-top: 18px;
            }}
            h3 {{
                font-size: 11pt;
                color: #34495e;
                margin-top: 12px;
            }}
            p, li {{
                font-size: 10pt;
                text-align: justify;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th {{
                background-color: #34495e;
                color: #ffffff;
                font-weight: bold;
                padding: 6px;
                font-size: 8.5pt;
                text-align: left;
                border: 1px solid #34495e;
            }}
            td {{
                border: 1px solid #dcdde1;
                padding: 5px;
                font-size: 8pt;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            code {{
                font-family: Courier, monospace;
                background-color: #f1f2f6;
                padding: 2px 4px;
                font-size: 8.5pt;
                color: #c0392b;
            }}
            pre {{
                background-color: #f1f2f6;
                padding: 8px;
                font-family: Courier, monospace;
                font-size: 8pt;
                border: 1px solid #dcdde1;
                border-radius: 4px;
            }}
            blockquote {{
                background-color: #eef7fc;
                border-left: 4px solid #3498db;
                padding: 8px 12px;
                margin: 12px 0;
                color: #2980b9;
                font-size: 9.5pt;
            }}
            hr {{
                border: 0;
                height: 1px;
                background: #bdc3c7;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    with open(pdf_path, 'wb') as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
        
    if pisa_status.err:
        print(f"Erro ao gerar o PDF: {pisa_status.err}")
    else:
        print(f"✅ PDF gerado com sucesso em: {pdf_path}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    md_file = os.path.join(project_root, 'journal-docs/relatorio-orientador.md')
    pdf_file = os.path.join(project_root, 'journal-docs/relatorio-orientador.pdf')
    convert_md_to_pdf(md_file, pdf_file)
