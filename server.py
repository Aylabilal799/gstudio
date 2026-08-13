import http.server
import socketserver
import threading

PORT = 5454

def start_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving files on port {PORT}")
        httpd.serve_forever()

def run_server_in_background():
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
