import http.server
import socketserver
import os
import webbrowser
from urllib.parse import urlparse, parse_qs
import subprocess
import threading
import time

# Configurações do servidor
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class LandingPageHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Redirecionar para o Streamlit quando for requisitado
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/dashboard':
            self.send_response(302)
            self.send_header('Location', 'http://localhost:8501')
            self.end_headers()
            return
        
        # Caso contrário, comportamento padrão (servir arquivos estáticos)
        return super().do_GET()

def run_landing_server():
    """Inicia o servidor da landing page"""
    with socketserver.TCPServer(("", PORT), LandingPageHandler) as httpd:
        print(f"Serving landing page at http://localhost:{PORT}")
        httpd.serve_forever()

def check_streamlit_running():
    """Verifica se o Streamlit está rodando e inicia caso não esteja"""
    try:
        import requests
        response = requests.get("http://localhost:8501/_stcore/health")
        if response.status_code == 200:
            print("Streamlit já está rodando em http://localhost:8501")
            return True
    except:
        pass
    
    # Streamlit não está rodando, vamos iniciá-lo
    print("Iniciando Streamlit...")
    streamlit_path = os.path.join(os.path.dirname(DIRECTORY), "dashboard", "app.py")
    
    # Iniciar em um processo separado (não bloqueia)
    subprocess.Popen([
        "python", "-m", "streamlit", "run", 
        streamlit_path, 
        "--server.port=8501"
    ], 
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
    
    # Aguardar Streamlit iniciar
    print("Aguardando Streamlit iniciar...")
    for _ in range(15):  # Tenta por 15 segundos
        try:
            import requests
            response = requests.get("http://localhost:8501/_stcore/health")
            if response.status_code == 200:
                print("Streamlit iniciado com sucesso!")
                return True
        except:
            pass
        time.sleep(1)
    
    print("Não foi possível confirmar que o Streamlit iniciou, mas continuando...")
    return False

def open_browser():
    """Abre o navegador na landing page"""
    time.sleep(1)  # Aguarda o servidor iniciar
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    # Verificar se o Streamlit está rodando
    check_streamlit_running()
    
    # Abrir navegador
    threading.Thread(target=open_browser).start()
    
    # Iniciar servidor
    run_landing_server() 