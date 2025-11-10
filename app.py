from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, flash, send_file,send_from_directory

from face_recognition import capture_and_recognize, add_new_face, get_attendance_records, save_to_csv,save_captured_image,add_new_pass
import base64
import os




app = Flask(__name__)
app.secret_key = os.urandom(24) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mark_attendance')
def mark_attendance():
    return render_template('mark_attendance.html')

@app.route('/check_attendance')
def check_attendance():
    records = get_attendance_records()
    print(f"Records being passed to template: {records}")  # Debug print
    return render_template('check_attendance.html', records=records)


@app.route('/capture_face', methods=['GET', 'POST'])
def capture_face():
    if request.method == 'POST':
        name = request.form['name']
        manager = request.form['manager']
        group = request.form['group']
        add_new_face(name, manager, group)
        flash('New directory has been created', 'success')
        return redirect(url_for('capture_face'))
    return render_template('capture_face.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')
    capture_count = data.get('capture_count')

    if not name or not image_data:
        return jsonify({'status': 'error', 'message': 'Missing name or image data'})

    image_data = base64.b64decode(image_data.split(',')[1])
    save_captured_image(name, image_data, capture_count)

    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    return Response(capture_and_recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_recognized_face', methods=['POST'])
def save_recognized_face():
    print("Received request at /save_recognized_face")  # Debugging log
    name = request.form.get('name')
    manager = request.form.get('manager')
    group = request.form.get('group')

    if not name or not manager or not group:
        return jsonify({'status': 'error', 'message': 'Missing details'})

    # Save to CSV
    save_to_csv(name, manager, group)
    
    return jsonify({'status': 'success'})

@app.route('/create_pass', methods=['GET', 'POST'])
def create_pass():
    if request.method == 'POST':
        name = request.form['name']
        manager = request.form['manager']
        group = request.form['group']
        days_valid = int(request.form['days_valid'])
        purpose = request.form['purpose']

        add_new_pass(name, manager, group, days_valid, purpose)
        
    
        flash('Pass created successfully', 'success')
        return redirect(url_for('create_pass'))

    return render_template('create_pass.html')


@app.route('/get_pass_image')
def get_pass_image():
    name = request.args.get('name')
    image_path = f'database/{name}/{name}_01.jpg'  # Assuming the first image is used for the pass
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/print_pass')
def print_pass():
    return render_template('print_pass.html')

@app.route('/database/<path:filename>')
def serve_file(filename):
    return send_from_directory('database', filename)

if __name__ == "__main__":
    app.run(debug=True)
