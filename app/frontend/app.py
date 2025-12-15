# app.py
from flask import Flask, render_template, request, jsonify
import os
import sys
import tempfile
import requests
import traceback
import time
import threading
from queue import Queue
from werkzeug.utils import secure_filename
from pathlib import Path

# Optional: enable CORS if front-end served from different origin during dev
# from flask_cors import CORS

# Add project root if needed (adjust as appropriate)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

app = Flask(__name__)
# CORS(app)  # uncomment if necessary

# Increase timeout settings for Flask development server
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for debugging

# Add global timeout setting for slow requests (in seconds)
REQUEST_TIMEOUT = 180  # 3 minutes for video processing

# Allowed extensions helper
ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv'}

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze/file', methods=['POST'])
def analyze_video_file():
    app.logger.info('Incoming /api/analyze/file')
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Файл не предоставлен'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Файл не выбран'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Недопустимый формат файла'}), 400

        filename = secure_filename(file.filename)
        
        # Save to a named temp file with the same suffix so model can infer codec if needed
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        app.logger.info(f'File saved to {temp_path}')
        
        try:
            # Call model with timeout protection
            result = call_actual_ml_model_with_timeout(temp_path, REQUEST_TIMEOUT - 10)
            
            # Ensure response contains expected keys
            response = {
                'success': True,
                'result': result.get('prediction', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', None),
                'details': result.get('details', {})
            }
            
            total_time = time.time() - start_time
            app.logger.info(f'Analysis completed in {total_time:.2f} seconds')
            return jsonify(response)
            
        finally:
            # Don't delete temp file too early - model might still be using it
            # Wait a bit before cleanup
            time.sleep(0.5)
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    app.logger.debug(f'Removed temp file {temp_path}')
            except Exception:
                app.logger.exception('Failed to remove temp file')
                
    except TimeoutError as e:
        app.logger.error(f'Timeout in analyze_video_file: {str(e)}')
        return jsonify({'success': False, 'error': f'Таймаут обработки: {str(e)}'}), 408
    except Exception as e:
        app.logger.exception('Error in analyze_video_file')
        return jsonify({'success': False, 'error': f'Ошибка анализа: {str(e)}', 'trace': traceback.format_exc()}), 500

@app.route('/api/analyze/url', methods=['POST'])
def analyze_video_url():
    app.logger.info('Incoming /api/analyze/url')
    start_time = time.time()
    
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'url' not in data:
            return jsonify({'success': False, 'error': 'Ссылка не предоставлена'}), 400

        video_url = data['url']
        if not video_url:
            return jsonify({'success': False, 'error': 'Ссылка пуста'}), 400

        # Save remote file to temp file with increased timeout
        suffix = Path(video_url).suffix or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            # Increase download timeout
            with requests.get(video_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if time.time() - start_time > REQUEST_TIMEOUT - 30:
                        raise TimeoutError('Download timeout')
                    if chunk:
                        tmp.write(chunk)
                        downloaded += len(chunk)

        app.logger.info(f'Downloaded URL to {temp_path}')
        
        try:
            # Call model with timeout protection
            result = call_actual_ml_model_with_timeout(temp_path, REQUEST_TIMEOUT - 10)
            
            response = {
                'success': True,
                'result': result.get('prediction', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', None),
                'details': result.get('details', {})
            }
            
            total_time = time.time() - start_time
            app.logger.info(f'URL analysis completed in {total_time:.2f} seconds')
            return jsonify(response)
            
        finally:
            # Don't delete temp file too early
            time.sleep(0.5)
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    app.logger.debug(f'Removed temp file {temp_path}')
            except Exception:
                app.logger.exception('Failed to remove temp file')
                
    except TimeoutError as e:
        app.logger.error(f'Timeout in analyze_video_url: {str(e)}')
        return jsonify({'success': False, 'error': f'Таймаут обработки: {str(e)}'}), 408
    except Exception as e:
        app.logger.exception('Error in analyze_video_url')
        return jsonify({'success': False, 'error': f'Ошибка анализа: {str(e)}', 'trace': traceback.format_exc()}), 500

# ===========================
# Model integration section
# ===========================
def simulate_ml_analysis(video_path):
    """Fallback simulation — keep for dev"""
    import random
    time.sleep(1.5)
    is_violent = random.random() > 0.5
    confidence = round(80 + random.random() * 15, 1)
    return {
        'prediction': 'violent' if is_violent else 'nonviolent',
        'confidence': confidence,
        'processing_time': round(1.5 + random.random(), 2),
        'details': {
            'frames_analyzed': random.randint(50, 500),
            'video_duration': round(10 + random.random() * 20, 1)
        }
    }

def call_actual_ml_model_with_timeout(video_path, timeout_seconds=170):
    """
    Calls the real VideoClassifier model with timeout protection.
    Uses threading to enforce timeout limits.
    """
    result_queue = Queue()
    error_queue = Queue()
    
    def worker():
        try:
            from remote.evaluated import VideoClassifier
            
            # Use global classifier to avoid reloading model every request
            global classifier
            try:
                classifier
            except NameError:
                classifier = VideoClassifier()
            
            result = classifier.predict(video_path)
            result_queue.put(result)
        except Exception as e:
            error_queue.put(e)
    
    # Start model processing in a separate thread
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Daemon thread will be killed if main thread exits
    thread.start()
    
    # Wait for result with timeout
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running - timeout occurred
        raise TimeoutError(f'Model processing exceeded {timeout_seconds} seconds')
    
    # Check for errors
    if not error_queue.empty():
        error = error_queue.get()
        raise error
    
    # Get result
    if not result_queue.empty():
        result = result_queue.get()
        return {
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'processing_time': float(result['processing_time']),
            'details': result.get('details', {})
        }
    else:
        raise RuntimeError('No result from model processing')

def call_actual_ml_model(video_path):
    """
    Original function kept for compatibility.
    Now wraps the timeout-protected version.
    """
    return call_actual_ml_model_with_timeout(video_path, REQUEST_TIMEOUT - 10)

# Run server for debug (only use app.run in development)
if __name__ == '__main__':
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000, 
        use_reloader=False,
        threaded=True  # Enable threading for concurrent requests
    )