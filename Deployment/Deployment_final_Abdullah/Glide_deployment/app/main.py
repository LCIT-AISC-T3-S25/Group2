import io
import time
import threading
import queue
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
from app.model_loader import ModelLoader

app = Flask(__name__)

# Initialize model loader (singleton pattern)
model_loader = ModelLoader(model_path="glide_like_model_final.pt")

def generate_in_thread(prompt, result_queue, cancel_event):
    """Thread function for image generation"""
    try:
        if cancel_event.is_set():
            return
            
        # Generate image
        image_tensor = model_loader.generate(prompt)
        
        if cancel_event.is_set():
            return
            
        # Convert tensor to PIL Image
        image_np = (image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        image = Image.fromarray(image_np)
        
        # Convert to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        result_queue.put(img_io)
    except Exception as e:
        result_queue.put(e)

@app.route('/')
def index():
    """Render a simple HTML interface for testing"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """API endpoint to generate image from text prompt"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Early validation
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Set up threading
        cancel_event = threading.Event()
        result_queue = queue.Queue()
        
        # Generate in background thread
        thread = threading.Thread(
            target=generate_in_thread,
            args=(prompt, result_queue, cancel_event)
        )
        thread.start()
        
        # Wait with timeout (290 seconds)
        thread.join(timeout=290)
        
        if thread.is_alive():
            cancel_event.set()
            return jsonify({'error': 'Generation timed out'}), 504
            
        if result_queue.empty():
            return jsonify({'error': 'Generation failed'}), 500
            
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
            
        return send_file(
            result,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'generated_{int(time.time())}.png'
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)