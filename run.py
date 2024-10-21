import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from flask import Flask
from App.routes import app

# Cấu hình logging
logging.basicConfig(filename='error.log', level=logging.INFO, format='%(message)s')

# Tạo logger cho Werkzeug
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)