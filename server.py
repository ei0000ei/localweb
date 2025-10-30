#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8002

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # 简化日志输出
        print(f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format % args}")
    
    def end_headers(self):
        # 添加CORS头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

print(f"启动HTTP服务器在端口 {PORT}")
print(f"服务目录: {os.getcwd()}")
print("文件列表:")
for file in os.listdir('.'):
    if file.endswith(('.html', '.csv', '.py')):
        print(f"  - {file}")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"请访问: http://localhost:{PORT} 或 http://127.0.0.1:{PORT}")
    print("按 Ctrl+C 停止服务器")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")