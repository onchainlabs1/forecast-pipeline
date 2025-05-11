#!/usr/bin/env python
"""
Script para iniciar o sistema completo com landing page (lovable) e dashboard (Streamlit)
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading

def run_landing_page():
    """Executa o servidor da landing page"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "landing", "server.py")
    
    print("Iniciando landing page...")
    landing_process = subprocess.Popen([
        sys.executable, script_path
    ])
    return landing_process

def open_landing_page():
    """Abre o navegador na landing page após um breve atraso"""
    time.sleep(2)  # Espera o servidor iniciar
    webbrowser.open("http://localhost:8000")
    print("Landing page aberta em http://localhost:8000")

if __name__ == "__main__":
    # Iniciar a landing page (que também inicia o Streamlit)
    landing_process = run_landing_page()
    
    # Abrir o navegador
    threading.Thread(target=open_landing_page).start()
    
    try:
        # Manter o processo principal rodando
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nEncerrando aplicação...")
        landing_process.terminate()
        print("Landing page encerrada.")
        
        # Tentar encerrar o Streamlit também
        try:
            import signal
            import psutil
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'streamlit' in ' '.join(proc.info['cmdline'] or []):
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    print(f"Processo Streamlit (PID: {proc.info['pid']}) encerrado.")
        except Exception as e:
            print(f"Aviso: Não foi possível encerrar o Streamlit automaticamente: {e}")
            print("Você pode precisar encerrar o processo Streamlit manualmente.")
        
        sys.exit(0) 