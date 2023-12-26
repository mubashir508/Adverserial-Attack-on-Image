from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import time
from PIL import Image
from werkzeug.utils import secure_filename

from inference import inference, simulateAttack

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_IMAGE_PATH = 'uploads/uploaded.jpg'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<path:path>')
def send_js(path):
    return send_from_directory('uploads', path)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/upload1')
def upload1():
    return render_template('upload1.html')
@app.route('/features')
def features():
    return render_template('features.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/func')
def func():
    return render_template('func.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        timestamp = str(time.time())
        filename = secure_filename(f"uploaded_{timestamp}.jpg")


        if os.path.exists(app.config['UPLOAD_FOLDER']) == False:
            os.mkdir(app.config['UPLOAD_FOLDER'])

        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = f"uploads\{filename}"
        global UPLOAD_IMAGE_PATH 
        UPLOAD_IMAGE_PATH = file_path
        # print(UPLOAD_IMAGE_PATH)
        file.save(file_path)
        return render_template('upload.html', image_path=file_path)

    return redirect(request.url)


@app.route('/apply_attack', methods=['GET'])
def apply_attack():

    epsilon = float(request.args.get('epsilon'))
    print(epsilon)
    # epsilon = 0.02

    selected_attack = request.args.get('selected_attack')
    print(selected_attack)
    fgsm = selected_attack == 'fgsm'
    
    original_image_path = UPLOAD_IMAGE_PATH
    # print(original_image_path)
    prediction=inference(original_image_path)
    # print(prediction)
    attack_output = simulateAttack(original_image_path, fgsm=fgsm, label=prediction[-1],epsilon=epsilon)
    noiseTensor = attack_output[3]
    attack_image = attack_output[0]
    Image.fromarray(noiseTensor.astype('uint8')).save(f"uploads/noise_{'FGSM' if fgsm else 'PGD'}.png")
    Image.fromarray(attack_image.astype('uint8')).save(f"uploads/adv_{'FGSM' if fgsm else 'PGD'}.png")

    # print(attack_output)
    adversarial_image_path =f"uploads/adv_{'FGSM' if fgsm else 'PGD'}.png"
    noise_image_path =f"uploads/noise_{'FGSM' if fgsm else 'PGD'}.png"

    attack_conf = format(attack_output[2],'.2f')
    selected_attack=selected_attack.upper()

    return render_template('upload.html', attack_path=adversarial_image_path,org_path=original_image_path,prediction=prediction, attack_confidence=attack_conf, attack_class=attack_output[1], epsilon=epsilon, selected_attack=selected_attack, noise_path=noise_image_path)

if __name__=="__main__":
    app.run(debug=True,port=8000)